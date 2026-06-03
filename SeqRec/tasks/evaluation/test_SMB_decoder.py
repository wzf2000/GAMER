import os
import numpy as np
import torch.distributed as dist
from loguru import logger
from typing import TYPE_CHECKING
from torch.utils.data import DataLoader

from SeqRec.tasks.evaluation.base import _BaseDecoderTestTask
from SeqRec.datasets.loaders.session_behavior import load_SMB_test_dataset, load_SMB_valid_dataset
from SeqRec.datasets.multi_behavior import EvaluationType
from SeqRec.datasets.session_behavior import BaseSMBDataset
from SeqRec.datasets.collators.generative import EncoderDecoderTestCollator, DecoderOnlyTestCollator, EncoderDecoderCollator, DecoderOnlyCollator
from SeqRec.evaluation.ranking import get_topk_results, get_metrics_results
from SeqRec.models.generative.registry import is_decoder_only_backbone
from SeqRec.tasks.evaluation.helpers import (
    build_behavior_prefix_fns,
    build_candidate_prefix_fn,
    build_generation_kwargs,
    get_generation_model,
    get_item_token_info,
    prepare_behavior_generation_prompt,
    slice_decoder_only_output,
)
from SeqRec.utils.args import SubParsersAction, parse_global_args, parse_dataset_args, parse_generation_eval_args
from SeqRec.utils.runtime import get_tqdm


if TYPE_CHECKING:
    from transformers import BatchEncoding
    from transformers.generation.utils import GenerateBeamOutput


class TestSMBDecoder(_BaseDecoderTestTask):
    """
    Test a SMB decoder for the SeqRec model.
    """

    @staticmethod
    def parser_name() -> str:
        return "test_SMB_decoder"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        """
        Add subparsers for the TestSMBDecoder task.
        """
        parser = sub_parsers.add_parser("test_SMB_decoder", help="Test a SMB decoder for SeqRec.")
        parser = parse_global_args(parser)
        parser = parse_dataset_args(parser)
        parse_generation_eval_args(
            parser,
            metrics="hit@1,hit@5,hit@10,recall@1,recall@5,recall@10,ndcg@5,ndcg@10",
            include_behaviors=True,
            include_valid_loss=True,
        )

    def check_collision_items(self) -> list[dict[str, int | float]]:
        ret_list = []
        for test_dataset in self.datasets:
            collision_cnt = 0
            for test_sample in test_dataset:
                target_items = test_sample["labels"]
                for target_item in target_items:
                    if target_item in test_dataset.collision_items:
                        collision_cnt += 1
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
        return ret_list

    def test_single_behavior(self, loader: DataLoader, num_beams: int, behavior: str) -> dict[str, float]:
        self.info(f"Start testing behavior {behavior} with {len(loader.dataset)} samples.")
        results: dict[str, float] = {}
        total = 0
        pbar = get_tqdm(desc=f"Testing ({EvaluationType.FIXED_BEHAVIOR.value} {behavior})", total=len(loader))

        user_metric_dict: dict[str, dict[int, float]] = {m: {} for m in self.metric_list}

        duplicate_ratios = []
        for batch in loader:
            batch: tuple["BatchEncoding", list[list[str]]]
            inputs = batch[0].to(self.device)
            targets = batch[1]
            batch_size = len(targets)
            behaviors: list[str] = [behavior for _ in range(batch_size)]
            dataset: BaseSMBDataset = loader.dataset
            decoder_input_ids, behavior_token_num = prepare_behavior_generation_prompt(
                inputs=inputs,
                tokenizer=self.tokenizer,
                dataset=dataset,
                behaviors=behaviors,
                device=self.device,
                is_decoder_only=is_decoder_only_backbone(self.backbone),
                decoder_start_token_id=self.config.decoder_start_token_id,
            )
            prefix_allowed_tokens_fn = self.prefix_allowed_tokens_by_behavior[behavior]

            gen_model = get_generation_model(self.model)
            gen_kwargs = build_generation_kwargs(
                backbone=self.backbone,
                inputs=inputs,
                max_new_tokens=self.sole_item_len,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                device=self.device,
                decoder_input_ids=decoder_input_ids if not is_decoder_only_backbone(self.backbone) else None,
            )
            output: "GenerateBeamOutput" = gen_model.generate(**gen_kwargs)
            output_ids = output.sequences
            scores = output.sequences_scores

            output_ids = slice_decoder_only_output(self.backbone, output_ids, self.item_len)

            output_str = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            if is_decoder_only_backbone(self.backbone):
                output_item_ids = output_ids[:, behavior_token_num:]  # Remove the behavior token if has
            else:
                output_item_ids = output_ids[:, behavior_token_num + 1:]  # Remove the decoder start token and behavior token if has
            output_items = self.tokenizer.batch_decode(output_item_ids, skip_special_tokens=True)
            output_items = [output_item.replace(' ', '') for output_item in output_items]
            # split the output items by num_beams
            output_items = [
                output_items[
                    i * num_beams: (i + 1) * num_beams
                ] for i in range(batch_size)
            ]
            history_items = inputs['inters_item_list']

            # count how many output items are in the history items for each sample
            duplicate_ratio = []
            for i in range(batch_size):
                output_item_set = set(output_items[i])
                history_item_set = set(history_items[i])
                intersection = output_item_set.intersection(history_item_set)
                duplicate_ratio.append(len(intersection) / len(output_item_set) if len(output_item_set) > 0 else 0)

            topk_res = get_topk_results(
                output_str,
                scores,
                targets,
                num_beams,
            )

            total += self._gather_sum(batch_size)
            topk_res = self._gather_concat(topk_res)
            targets = self._gather_concat(targets)
            duplicate_ratio = self._gather_concat(duplicate_ratio)
            if 'uid' in inputs:
                uid = self._gather_concat(inputs['uid'])

            if 'uid' in inputs:
                batch_metrics_res = get_metrics_results(topk_res, self.metric_list, targets, list_output=True)
                for m in batch_metrics_res:
                    for i in range(len(uid)):
                        user_metric_dict[m][uid[i]] = batch_metrics_res[m][i]
                batch_metrics_res = {
                    m: sum(batch_metrics_res[m]) for m in batch_metrics_res
                }
            else:
                batch_metrics_res = get_metrics_results(topk_res, self.metric_list, targets, list_output=False)
            for m, res in batch_metrics_res.items():
                if m not in results:
                    results[m] = res
                else:
                    results[m] += res
            duplicate_ratios.extend(duplicate_ratio)

            if self.local_rank == 0:
                show_metric_keys = self.metric_list[:2]  # Show only the first two metrics
                show_metric_dict = {
                    m: f"{results[m] / total:.4f}" for m in show_metric_keys if m in results
                }
                show_metric_dict["Avg. Duplicate Ratio"] = f"{np.mean(duplicate_ratios):.4f}"
                pbar.set_postfix(show_metric_dict)
                pbar.update(1)
            if self.ddp:
                dist.barrier()
        if pbar:
            pbar.close()

        self.info(f"Finished testing behavior {behavior} with {total} samples.")
        for m in results:
            results[m] = results[m] / total
        results["Avg. Duplicate Ratio"] = np.mean(duplicate_ratios)

        save_path = os.path.join(
            self.results_file.replace(".json", ""),
            f"user_level_metrics_{behavior}.json",
        )
        self._save_user_metrics(user_metric_dict, len(loader.dataset), save_path, results)

        return results

    def test(self, num_beams: int) -> list[dict[str, float]]:
        results = []
        merge_results = {m: 0.0 for m in self.metric_list}
        total = 0
        for i, behavior in enumerate(self.behaviors):
            result = self.test_single_behavior(self.loaders[i], num_beams, behavior)
            result['eval_type'] = f"Behavior {behavior}"
            result['collision_info'] = self.collision_info[i]
            results.append(result)
            for m in self.metric_list:
                assert m in result, f"Metric {m} not found in results for behavior {behavior}."
                merge_results[m] += result[m] * len(self.loaders[i].dataset)
            total += len(self.loaders[i].dataset)
        for m in merge_results:
            merge_results[m] /= total
        merge_results['eval_type'] = "Merged Behavior"
        results.append(merge_results)
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
        behaviors: list[str] | None,
        valid_loss: bool,
        *args,
        **kwargs
    ):
        """
        Test the SMB decoder using the provided arguments.
        """
        self.init(seed, False)
        self._load_model_via_registry(backbone, ckpt_path)
        # output the parameters of the model
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.success(f"Model {backbone} has {total_params} parameters, {trainable_params} of them are trainable.")

        if valid_loss:
            self.valid_dataset = load_SMB_valid_dataset(
                dataset,
                data_path,
                max_his_len,
                index_file,
                test_task,
            )
            self.datasets: list[BaseSMBDataset] = [self.valid_dataset]
        else:
            self.base_dataset = load_SMB_test_dataset(
                dataset,
                data_path,
                max_his_len,
                index_file,
                test_task,
            )
            self.datasets: list[BaseSMBDataset] = []
            if behaviors is None:
                self.behaviors = self.base_dataset.behaviors
            else:
                self.behaviors = behaviors
            for behavior in self.behaviors:
                self.datasets.append(self.base_dataset.filter_by_behavior(behavior))
                self.info(f"Loaded dataset for behavior {behavior} with {len(self.datasets[-1])} samples.")

        self.samplers = self._setup_ddp_for_datasets(self.datasets)

        if valid_loss:
            behavior_tokens: list[str] = []
            for behavior in self.datasets[0].behaviors:
                behavior_tokens.extend(self.datasets[0].get_behavior_tokens(behavior))
            behavior_tokens = [
                self.tokenizer.encode(b, add_special_tokens=False)[0]
                for b in behavior_tokens
            ]
            if is_decoder_only_backbone(backbone):
                collator = DecoderOnlyCollator(self.tokenizer, ignore_behavior_tokens=behavior_tokens)
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
            self.collision_info = self.check_collision_items()

            self.all_behavior_items = self.datasets[0].get_all_items("all")
            item_reps = list(self.all_behavior_items)
            items_tokens, self.item_len, last_token_set = get_item_token_info(
                self.tokenizer,
                item_reps,
                self.config.pad_token_id,
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
            for behavior in self.behaviors:
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
