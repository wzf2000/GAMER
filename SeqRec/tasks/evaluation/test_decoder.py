import os
import torch
import torch.distributed as dist
from typing import TYPE_CHECKING
from torch.utils.data import DataLoader

from SeqRec.tasks.evaluation.base import _BaseDecoderTestTask
from SeqRec.datasets.loaders.sequential import load_test_dataset
from SeqRec.datasets.collators.generative import EncoderDecoderTestCollator, DecoderOnlyTestCollator
from SeqRec.tasks.evaluation.helpers import (
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


class TestDecoder(_BaseDecoderTestTask):
    """
    Test a decoder for the SeqRec model.
    """

    @staticmethod
    def parser_name() -> str:
        return "test_decoder"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        """
        Add subparsers for the TestDecoder task.
        """
        parser = sub_parsers.add_parser("test_decoder", help="Train a decoder for SeqRec.")
        parser = parse_global_args(parser)
        parser = parse_dataset_args(parser)
        parse_generation_eval_args(
            parser,
            metrics="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
            include_filter=True,
        )

    def check_collision_items(self, filter: bool = False) -> dict[str, int | float]:
        collision_cnt = 0
        new_inter_data = []
        for i, test_sample in enumerate(self.dataset):
            target_item = test_sample["labels"]
            if target_item in self.dataset.collision_items:
                collision_cnt += 1
            else:
                new_inter_data.append(self.dataset.inter_data[i])
        self.info([
            f"Total test data num: {len(self.dataset)}",
            f"Collision items num: {len(self.dataset.collision_items)}",
            f"Collision sample num: {collision_cnt}",
            f"Collision items ratio: {collision_cnt / len(self.dataset):.4f}",
        ])
        ret = {
            "total": len(self.dataset),
            "collision_items": len(self.dataset.collision_items),
            "collision_samples": collision_cnt,
            "collision_ratio": collision_cnt / len(self.dataset),
        }
        if filter:
            # Filter out the collision items from the test data
            self.dataset.inter_data = new_inter_data
            self.info(f"Filtered test data num: {len(self.dataset)}")
        return ret

    def test_single_type(self, loader: DataLoader, num_beams: int) -> dict[str, float]:
        results: dict[str, float] = {}
        total = 0
        pbar = get_tqdm(desc="Testing", total=len(loader))

        user_metric_dict: dict[str, dict[int, float]] = {m: {} for m in self.metric_list}

        for batch in loader:
            batch: tuple["BatchEncoding", list[str], torch.LongTensor]
            inputs = batch[0].to(self.device)
            targets = batch[1]
            if self.backbone == 'Qwen3':
                max_new_tokens = self.item_len
                inputs.input_ids = inputs.input_ids[:, :-max_new_tokens]
                inputs.attention_mask = inputs.attention_mask[:, :-max_new_tokens]
            else:
                max_new_tokens = 10
            batch_size = len(targets)

            gen_kwargs = build_generation_kwargs(
                backbone=self.backbone,
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                prefix_allowed_tokens_fn=self.prefix_allowed_tokens,
                num_beams=num_beams,
                device=self.device,
                decoder_start_token_id=self.config.decoder_start_token_id,
            )
            output: "GenerateBeamOutput" = get_generation_model(self.model).generate(**gen_kwargs)
            output_ids = output.sequences
            scores = output.sequences_scores

            if self.backbone == 'Qwen3':
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

            if self.local_rank == 0:
                show_metric_keys = self.metric_list[:2]  # Show only the first two metrics
                show_metric_dict = {
                    m: f"{results[m] / total:.4f}" for m in show_metric_keys if m in results
                }
                pbar.set_postfix(show_metric_dict)
                pbar.update(1)
            if self.ddp:
                dist.barrier()

        if self.ddp:
            dist.barrier()
        for m in results:
            results[m] = results[m] / total

        save_path = os.path.join(self.results_file.replace(".json", ""), "user_level_metrics.json")
        self._save_user_metrics(user_metric_dict, len(loader.dataset), save_path, results)

        return results

    def test(self, num_beams: int) -> dict[str, float]:
        results = self.test_single_type(self.loader, num_beams)
        results['collision_info'] = self.collision_info
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
        *args,
        **kwargs
    ):
        """
        Test the decoder using the provided arguments.
        """
        self.init(seed, False)
        self._load_model_via_registry(backbone, ckpt_path)
        self.dataset = load_test_dataset(
            dataset,
            data_path,
            max_his_len,
            index_file,
            test_task,
        )
        self.sampler = self._setup_ddp_for_datasets([self.dataset])[0]

        if backbone == 'Qwen3':
            collator = DecoderOnlyTestCollator(self.tokenizer)
        else:
            collator = EncoderDecoderTestCollator(self.tokenizer)

        self.all_items = self.dataset.get_all_items()
        self.collision_info = self.check_collision_items(filter)

        item_reps = list(self.all_items)
        _, self.item_len, last_token_set = get_item_token_info(
            self.tokenizer, item_reps, self.config.pad_token_id,
        )
        self.prefix_allowed_tokens = build_candidate_prefix_fn(
            backbone=backbone,
            tokenizer=self.tokenizer,
            config=self.config,
            items=item_reps,
            last_token_set=last_token_set,
        )
        self.info("Complete building candidate trie for prefix allowed tokens function.")

        self.loader = DataLoader(
            self.dataset,
            batch_size=test_batch_size,
            collate_fn=collator,
            sampler=self.sampler,
            num_workers=2,
            pin_memory=True,
        )
        self.info([
            "Complete loading test datasets and collators.",
            f"Dataset num: {len(self.dataset)}",
        ])

        self.model.eval()
        self.metric_list = metrics.split(",")
        self.backbone = backbone
        self.results_file = results_file

        results = self.test(num_beams)
        self._save_results_and_log(results, self.results_file, multiple=False)

        self.finish(False)
