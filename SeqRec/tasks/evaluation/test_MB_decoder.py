import os
import torch
from loguru import logger
from typing import TYPE_CHECKING
from torch.utils.data import DataLoader

from SeqRec.tasks.evaluation.base import _BaseDecoderTestTask
from SeqRec.datasets.loaders.multi_behavior import load_MB_test_dataset, load_MB_valid_dataset
from SeqRec.datasets.multi_behavior import BaseMBDataset, EvaluationType
from SeqRec.datasets.collators.generative import EncoderDecoderTestCollator, DecoderOnlyTestCollator, EncoderDecoderCollator, DecoderOnlyCollator
from SeqRec.models.generative.registry import backbone_uses_actions, is_decoder_only_backbone
from SeqRec.tasks.evaluation.helpers import (
    build_behavior_prefix_fns,
    build_candidate_prefix_fn,
    build_generation_kwargs,
    get_generation_model,
    get_item_token_info,
)
from SeqRec.utils.args import SubParsersAction, parse_global_args, parse_dataset_args, parse_generation_eval_args
from SeqRec.utils.runtime import get_tqdm


if TYPE_CHECKING:
    from transformers import BatchEncoding
    from transformers.generation.utils import GenerateBeamOutput


class TestMBDecoder(_BaseDecoderTestTask):
    """
    Test a MB decoder for the SeqRec model.
    """

    @staticmethod
    def parser_name() -> str:
        return "test_MB_decoder"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        """
        Add subparsers for the TestMBDecoder task.
        """
        parser = sub_parsers.add_parser("test_MB_decoder", help="Train a MB decoder for SeqRec.")
        parser = parse_global_args(parser)
        parser = parse_dataset_args(parser)
        parse_generation_eval_args(
            parser,
            metrics="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
            include_filter=True,
            include_eval_types=True,
            include_valid_loss=True,
        )

    def check_collision_items(self, filter: bool = False) -> list[dict[str, int | float]]:
        ret_list = []
        for test_dataset in self.datasets:
            collision_cnt = 0
            new_inter_data = []
            for i, test_sample in enumerate(test_dataset):
                target_item = test_sample["labels"]
                if target_item in test_dataset.collision_items:
                    collision_cnt += 1
                else:
                    new_inter_data.append(test_dataset.inter_data[i])
            self.info([
                f"Total test data num: {len(test_dataset)}",
                f"Collision items num: {len(test_dataset.collision_items)}",
                f"Collision sample num: {collision_cnt}",
                f"Collision items ratio: {collision_cnt / len(test_dataset):.4f}",
            ])
            ret = {
                "total": len(test_dataset),
                "collision_items": len(test_dataset.collision_items),
                "collision_samples": collision_cnt,
                "collision_ratio": collision_cnt / len(test_dataset),
            }
            ret_list.append(ret)
            if filter:
                # Filter out the collision items from the test data
                test_dataset.inter_data = new_inter_data
                self.info(f"Filtered test data num: {len(test_dataset)}")
        return ret_list

    def test_single_type(self, loader: DataLoader, num_beams: int, eval_type: EvaluationType | None = None) -> dict[str, float]:
        results: dict[str, float] = {}
        total = 0
        pbar = get_tqdm(desc="Testing" if eval_type is None else f"Testing ({eval_type.value})", total=len(loader))

        user_metric_dict: dict[str, dict[int, float]] = {m: {} for m in self.metric_list}

        for batch in loader:
            batch: tuple["BatchEncoding", list[str]]
            inputs = batch[0].to(self.device)
            targets = batch[1]
            if eval_type in [EvaluationType.TARGET_BEHAVIOR, EvaluationType.BEHAVIOR_SPECIFIC]:
                behaviors: list[str] = inputs.pop("behavior", None)
                assert behaviors is not None, "behaviors should not be None"
                dataset: BaseMBDataset = loader.dataset
                behavior_tokens = [''.join(dataset.get_behavior_tokens(b)) for b in behaviors]
                behavior_tokens = self.tokenizer.batch_encode_plus(behavior_tokens, add_special_tokens=False)["input_ids"]
                decoder_input_ids = [[self.config.decoder_start_token_id] + tokens for tokens in behavior_tokens]
                if is_decoder_only_backbone(self.backbone):
                    # Get any item in all_items
                    max_new_tokens = self.sole_item_len
                    inputs.input_ids = inputs.input_ids[:, :-max_new_tokens]
                    inputs.attention_mask = inputs.attention_mask[:, :-max_new_tokens]
                if backbone_uses_actions(self.backbone):
                    action = [[dataset.behavior_level[u]] for u in behaviors]
                    inputs.actions = torch.cat([inputs.actions, torch.tensor(action, device=self.device)], dim=1)
                if eval_type == EvaluationType.TARGET_BEHAVIOR:
                    prefix_allowed_tokens_fn = self.prefix_allowed_tokens_by_behavior[dataset.target_behavior]
                else:
                    prefix_allowed_tokens_fn = self.prefix_allowed_tokens
            else:
                if is_decoder_only_backbone(self.backbone):
                    max_new_tokens = self.item_len
                    inputs.input_ids = inputs.input_ids[:, :-max_new_tokens]
                    inputs.attention_mask = inputs.attention_mask[:, :-max_new_tokens]
                decoder_input_ids = [[self.config.decoder_start_token_id] for _ in targets]
                prefix_allowed_tokens_fn = self.prefix_allowed_tokens
            batch_size = len(targets)

            if is_decoder_only_backbone(self.backbone):
                gen_kwargs = build_generation_kwargs(
                    backbone=self.backbone,
                    inputs=inputs,
                    max_new_tokens=self.sole_item_len if backbone_uses_actions(self.backbone) else max_new_tokens,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    num_beams=num_beams,
                    device=self.device,
                )
            else:
                gen_kwargs = build_generation_kwargs(
                    backbone=self.backbone,
                    inputs=inputs,
                    max_new_tokens=10,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    num_beams=num_beams,
                    device=self.device,
                    decoder_input_ids=decoder_input_ids,
                    decoder_start_token_id=self.config.decoder_start_token_id,
                )
            output: "GenerateBeamOutput" = get_generation_model(self.model).generate(**gen_kwargs)
            output_ids = output.sequences
            scores = output.sequences_scores

            if is_decoder_only_backbone(self.backbone):
                output_ids = output_ids[:, -self.item_len:]

            output_str = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            total += self._accumulate_batch_metrics(
                output_str=output_str,
                scores=scores,
                targets=targets,
                inputs=inputs,
                batch_size=batch_size,
                num_beams=num_beams,
                results=results,
                user_metric_dict=user_metric_dict,
            )
            self._pbar_step(pbar, results=results, total=total)

        save_path = os.path.join(
            self.results_file.replace(".json", ""),
            f"user_level_metrics_[{eval_type.value}].json",
        )
        self._finalize_loop_metrics(
            results=results, total=total,
            user_metric_dict=user_metric_dict,
            dataset_len=len(loader.dataset),
            save_path=save_path,
        )
        return results

    def test(self, num_beams: int) -> list[dict[str, float]]:
        results = []
        for eval_type in self.eval_types:
            result = self.test_single_type(self.loaders[1 if eval_type == EvaluationType.TARGET_BEHAVIOR else 0], num_beams, eval_type)
            result['eval_type'] = eval_type.value
            result['collision_info'] = self.collision_info[1 if eval_type == EvaluationType.TARGET_BEHAVIOR else 0]
            results.append(result)
        return results

    def invoke(
        self,
        # global arguments
        seed: int,
        backbone: str,
        base_model: str,  # unused in testing
        output_dir: str,  # unused in testing
        # dataset arguments
        data_path: str,
        tasks: str,  # unused in testing
        dataset: str,
        index_file: str,
        max_his_len: int,
        # testing arguments
        ckpt_path: str,
        results_file: str,
        test_batch_size: int,
        num_beams: int,
        metrics: str,
        test_task: str,
        filter: bool,
        eval_types: str,
        valid_loss: bool,
        *args,
        **kwargs
    ):
        """
        Test the MB decoder using the provided arguments.
        """
        self.init(seed, False)
        self.eval_types = eval_types.split(",")
        for eval_type in self.eval_types:
            assert eval_type in ["target_behavior", "behavior_specific", "behavior_item"], f"Invalid evaluation type: {eval_type}"
        self.eval_types = [EvaluationType(" ".join([e.capitalize() for e in eval_type.split("_")])) for eval_type in self.eval_types]
        logger.info(f"Evaluation types: {[e.value for e in self.eval_types]}")
        self._load_model_via_registry(backbone, ckpt_path)

        if valid_loss:
            self.datasets = [load_MB_valid_dataset(
                dataset,
                data_path,
                max_his_len,
                index_file,
                test_task,
            )]
        else:
            self.datasets = [load_MB_test_dataset(
                dataset,
                data_path,
                max_his_len,
                index_file,
                test_task,
            )]
            self.datasets.append(self.datasets[0].filter_by_behavior(self.datasets[0].target_behavior))

        self.samplers = self._setup_ddp_for_datasets(self.datasets)

        if valid_loss:
            if is_decoder_only_backbone(backbone):
                collator = DecoderOnlyCollator(self.tokenizer, only_train_response=True)
            else:
                collator = EncoderDecoderCollator(self.tokenizer)
        else:
            if is_decoder_only_backbone(backbone):
                collator = DecoderOnlyTestCollator(self.tokenizer)
            else:
                collator = EncoderDecoderTestCollator(self.tokenizer)

            for test_dataset in self.datasets:
                test_dataset.get_all_items()
            self.all_items = self.datasets[0].get_all_items()
            self.collision_info = self.check_collision_items(filter)

            self.all_behavior_items = self.datasets[0].get_all_items("all")
            item_reps = list(self.all_behavior_items)
            _, self.item_len, last_token_set = get_item_token_info(
                self.tokenizer, item_reps, self.config.pad_token_id,
            )
            self.sole_item_len = len(self.tokenizer.encode(next(iter(self.all_items)), add_special_tokens=False))
            self.info("Complete get all behavior items last token set.")

            self.prefix_allowed_tokens = build_candidate_prefix_fn(
                backbone=backbone,
                tokenizer=self.tokenizer,
                config=self.config,
                items=item_reps,
                last_token_set=last_token_set,
            )
            self.info("Complete building all behavior candidate trie for prefix allowed tokens function.")

            self.prefix_allowed_tokens_by_behavior = build_behavior_prefix_fns(
                backbone=backbone,
                tokenizer=self.tokenizer,
                config=self.config,
                dataset=self.datasets[0],
                last_token_set=last_token_set,
            )
            for behavior in self.datasets[0].behaviors:
                self.info(f"Complete building candidate trie for behavior {behavior} prefix allowed tokens function.")
            self.info("Complete building candidate trie for prefix allowed tokens function.")

        self.loaders = [DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            collate_fn=collator,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
        ) for sampler, test_dataset in zip(self.samplers, self.datasets)]
        self.info(["Complete loading datasets and collators."] + [
            f"Dataset {i} num: {len(test_dataset)}" for i, test_dataset in enumerate(self.datasets)
        ])

        self.model.eval()
        self.metric_list = metrics.split(",")
        self.backbone = backbone
        self.results_file = results_file

        if valid_loss:
            self.info("Testing valid dataset...")
            self.validation()
        else:
            results = self.test(num_beams)
            self._save_results_and_log(results, self.results_file, multiple=True)

        self.finish(False)
