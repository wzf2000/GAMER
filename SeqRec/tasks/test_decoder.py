import os
import json
import torch
import torch.distributed as dist
from tqdm import tqdm
from loguru import logger
from typing import Callable
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BatchEncoding, T5Config, T5Tokenizer
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerateBeamOutput

from SeqRec.tasks.multi_gpu import MultiGPUTask
from SeqRec.datasets.loading import load_test_dataset
from SeqRec.datasets.MB_dataset import BaseMBDataset, EvaluationType
from SeqRec.datasets.collator import TestCollator
from SeqRec.models.TIGER import TIGER
from SeqRec.models.PBATransformers import PBATransformerConfig, PBATransformersForConditionalGeneration
from SeqRec.evaluation.ranking import get_topk_results, get_metrics_results
from SeqRec.generation.trie import Trie, prefix_allowed_tokens_fn
from SeqRec.utils.futils import ensure_dir
from SeqRec.utils.parse import SubParsersAction, parse_global_args, parse_dataset_args


class TestDecoder(MultiGPUTask):
    """
    Test a decoder for the SeqRec model.
    """

    @staticmethod
    def parser_name() -> str:
        return "test_decoder"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        """
        Add subparsers for the TrainDecoder task.
        """
        parser = sub_parsers.add_parser("test_decoder", help="Train a decoder for SeqRec.")
        parser = parse_global_args(parser)
        parser = parse_dataset_args(parser)
        parser.add_argument("--ckpt_path", type=str, default="./checkpoint", help="The checkpoint path")
        parser.add_argument(
            "--results_file",
            type=str,
            default="./results/test.json",
            help="result output path",
        )
        parser.add_argument("--test_batch_size", type=int, default=16)
        parser.add_argument("--num_beams", type=int, default=20)
        parser.add_argument(
            "--metrics",
            type=str,
            default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
            help="test metrics, separate by comma",
        )
        parser.add_argument("--test_task", type=str, default="SeqRec")
        parser.add_argument(
            "--filter",
            action="store_true",
            help="Filter out the collision items from the test data",
        )

    @property
    def multi_behavior(self) -> bool:
        assert hasattr(self, "datasets") and len(self.datasets) > 0, "Test data is not initialized."
        return isinstance(self.datasets[0], BaseMBDataset)

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
            if self.local_rank == 0:
                logger.info(f"Total test data num: {len(test_dataset)}")
                logger.info(f"Collision items num: {len(test_dataset.collision_items)}")
                logger.info(f"Collision sample num: {collision_cnt}")
                logger.info(f"Collision items ratio: {collision_cnt / len(test_dataset)}")
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
                if self.local_rank == 0:
                    logger.info(f"Filtered test data num: {len(test_dataset)}")
        return ret_list

    def test_single_type(self, loader: DataLoader, num_beams: int, eval_type: EvaluationType | None = None) -> dict[str, float]:
        results: dict[str, float] = {}
        total = 0
        if self.local_rank == 0:
            pbar = tqdm(total=len(loader), desc="Testing" if eval_type is None else f"Testing ({eval_type.value})")

        for batch in loader:
            batch: tuple[BatchEncoding, list[str], torch.LongTensor]
            inputs = batch[0].to(self.device)
            targets = batch[1]
            label_ids = batch[2].to(self.device)
            if eval_type in [EvaluationType.TARGET_BEHAVIOR, EvaluationType.BEHAVIOR_SPECIFIC]:
                assert self.multi_behavior, "Multi-behavior dataset is required for target behavior evaluation."
                behaviors: list[str] = inputs.pop("target_behavior", None)
                dataset: BaseMBDataset = loader.dataset
                behavior_tokens = [''.join(dataset.get_behavior_tokens(b)) for b in behaviors]
                decoder_input_ids = [[self.config.decoder_start_token_id] + self.tokenizer.encode(behavior_token_str, add_special_tokens=False) for behavior_token_str in behavior_tokens]
                if eval_type == EvaluationType.TARGET_BEHAVIOR:
                    prefix_allowed_tokens_fn = self.prefix_allowed_tokens_by_behavior[dataset.target_behavior]
                else:
                    prefix_allowed_tokens_fn = self.prefix_allowed_tokens
            else:
                decoder_input_ids = [[self.config.decoder_start_token_id] for _ in targets]
                prefix_allowed_tokens_fn = self.prefix_allowed_tokens
            batch_size = len(targets)

            output: GenerateBeamOutput = (
                self.model
                if isinstance(self.model, GenerationMixin)
                else
                self.model.module
            ).generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                decoder_input_ids=torch.tensor(decoder_input_ids, device=self.device),
                max_new_tokens=10,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
            )
            output_ids = output.sequences
            scores = output.sequences_scores

            output_str = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            topk_res = get_topk_results(
                output_str,
                scores,
                targets,
                num_beams,
            )

            if self.ddp:
                batch_size_gather_list = [None for _ in range(self.world_size)]
                dist.all_gather_object(obj=batch_size, object_list=batch_size_gather_list)
                total += sum(batch_size_gather_list)
                res_gather_list = [None for _ in range(self.world_size)]
                dist.all_gather_object(obj=topk_res, object_list=res_gather_list)

                all_device_topk_res = []
                for ga_res in res_gather_list:
                    all_device_topk_res += ga_res
                topk_res = all_device_topk_res
            else:
                total += batch_size

            batch_metrics_res = get_metrics_results(topk_res, self.metric_list)
            for m, res in batch_metrics_res.items():
                if m not in results:
                    results[m] = res
                else:
                    results[m] += res

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

        return results

    def test(self, num_beams: int) -> dict[str, float] | list[dict[str, float]]:
        if not self.multi_behavior:
            results = self.test_single_type(self.loaders[0], num_beams)
            results['collision_info'] = self.collision_info[0]
            return results
        results = []
        result = self.test_single_type(self.loaders[1], num_beams, EvaluationType.TARGET_BEHAVIOR)
        result['eval_type'] = "Target Behavior"
        result['collision_info'] = self.collision_info[1]
        results.append(result)
        result = self.test_single_type(self.loaders[0], num_beams, EvaluationType.BEHAVIOR_SPECIFIC)
        result['eval_type'] = "Behavior Specific"
        result['collision_info'] = self.collision_info[0]
        results.append(result)
        result = self.test_single_type(self.loaders[0], num_beams, EvaluationType.BEHAVIOR_ITEM)
        result['eval_type'] = "Behavior Item"
        result['collision_info'] = self.collision_info[0]
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
        *args,
        **kwargs
    ):
        """
        Test the decoder using the provided arguments.
        """
        # Implementation of the training logic goes here.
        self.init(seed, False)
        if backbone == 'TIGER':
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(ckpt_path, legacy=True)
            self.model = TIGER.from_pretrained(ckpt_path).to(self.device)
            self.config: T5Config = self.model.config
        elif backbone == 'PBATransformers':
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(ckpt_path, legacy=True)
            self.model = PBATransformersForConditionalGeneration.from_pretrained(ckpt_path).to(self.device)
            self.config: PBATransformerConfig = self.model.config
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        assert isinstance(self.model, GenerationMixin), "Model must be a generation model."
        self.datasets = [load_test_dataset(
            dataset,
            data_path,
            max_his_len,
            index_file,
            test_task,
        )]
        if self.multi_behavior:
            self.datasets.append(self.datasets[0].filter_by_behavior(self.datasets[0].target_behavior))
        if self.ddp:
            self.samplers = [DistributedSampler(
                test_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
            ) for test_dataset in self.datasets]
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            self.samplers = [None] * len(self.datasets)
        collator = TestCollator(self.tokenizer)
        for test_dataset in self.datasets:
            test_dataset.get_all_items()
        self.all_items = self.datasets[0].get_all_items()
        self.collision_info = self.check_collision_items(filter)
        if self.multi_behavior:
            assert isinstance(self.datasets[0], BaseMBDataset), "Expected a multi-behavior dataset."
            self.all_behavior_items = self.datasets[0].get_all_items("all")
            candidate_trie = Trie(
                [[self.config.decoder_start_token_id] + self.tokenizer.encode(candidate) for candidate in self.all_behavior_items]
            )
            self.prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
            self.prefix_allowed_tokens_by_behavior: dict[str, Callable[[int, torch.Tensor], list[int]]] = {}
            behaviors = self.datasets[0].behaviors
            for behavior in behaviors:
                all_items = self.datasets[0].get_all_items(behavior)
                behavior_trie = Trie(
                    [[self.config.decoder_start_token_id] + self.tokenizer.encode(candidate) for candidate in all_items]
                )
                self.prefix_allowed_tokens_by_behavior[behavior] = prefix_allowed_tokens_fn(behavior_trie)
        else:
            candidate_trie = Trie(
                [[self.config.decoder_start_token_id] + self.tokenizer.encode(candidate) for candidate in self.all_items]
            )
            self.prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
        self.loaders = [DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            collate_fn=collator,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
        ) for sampler, test_dataset in zip(self.samplers, self.datasets)]
        if self.local_rank == 0:
            for i, test_dataset in enumerate(self.datasets):
                logger.info(f"Dataset {i} num: {len(test_dataset)}")

        self.model.eval()
        self.metric_list = metrics.split(",")
        results = self.test(num_beams)
        if self.local_rank == 0:
            logger.success("======================================================")
            logger.success("Results:")
            if self.multi_behavior:
                for res in results:
                    logger.success("======================================================")
                    logger.success(f"{res['eval_type']} results:")
                    for m in res:
                        if isinstance(res[m], float):
                            logger.success(f"\t{m} = {res[m]:.4f}")
            else:
                for m in results:
                    logger.success(f"\t{m} = {results[m]:.4f}")
            logger.success("======================================================")
            ensure_dir(os.path.dirname(results_file))
            with open(results_file, "w") as f:
                json.dump(results, f, indent=4)
            logger.success(f"Results saved to {results_file}.")

        self.finish(False)
