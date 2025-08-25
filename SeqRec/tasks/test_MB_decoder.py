import os
import json
import torch
import torch.distributed as dist
from loguru import logger
from typing import Callable
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BatchEncoding, T5Config, T5Tokenizer, Qwen3Config, Qwen2Tokenizer
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerateBeamOutput

from SeqRec.tasks.multi_gpu import MultiGPUTask
from SeqRec.datasets.loading_MB import load_MB_test_dataset
from SeqRec.datasets.MB_dataset import BaseMBDataset, EvaluationType
from SeqRec.datasets.collator import EncoderDecoderTestCollator, DecoderOnlyTestCollator
from SeqRec.models.TIGER import TIGER
from SeqRec.models.PBATransformers import PBATransformerConfig, PBATransformersForConditionalGeneration
from SeqRec.models.Qwen import Qwen3WithTemperature
from SeqRec.evaluation.ranking import get_topk_results, get_metrics_results
from SeqRec.generation.trie import Trie, prefix_allowed_tokens_fn, prefix_allowed_tokens_fn_by_last_token
from SeqRec.utils.futils import ensure_dir
from SeqRec.utils.parse import SubParsersAction, parse_global_args, parse_dataset_args
from SeqRec.utils.pipe import get_tqdm


class TestMBDecoder(MultiGPUTask):
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

        for batch in loader:
            batch: tuple[BatchEncoding, list[str], torch.LongTensor]
            inputs = batch[0].to(self.device)
            targets = batch[1]
            if eval_type in [EvaluationType.TARGET_BEHAVIOR, EvaluationType.BEHAVIOR_SPECIFIC]:
                behaviors: list[str] = inputs.pop("target_behavior", None)
                dataset: BaseMBDataset = loader.dataset
                behavior_tokens = [''.join(dataset.get_behavior_tokens(b)) for b in behaviors]
                behavior_tokens = self.tokenizer.batch_encode_plus(behavior_tokens, add_special_tokens=False)["input_ids"]
                decoder_input_ids = [[self.config.decoder_start_token_id] + tokens for tokens in behavior_tokens]
                if self.backbone == 'Qwen3':
                    # Get any item in all_items
                    max_new_tokens = self.sole_item_len
                    inputs.input_ids = inputs.input_ids[:, :-max_new_tokens]
                    inputs.attention_mask = inputs.attention_mask[:, :-max_new_tokens]
                if eval_type == EvaluationType.TARGET_BEHAVIOR:
                    prefix_allowed_tokens_fn = self.prefix_allowed_tokens_by_behavior[dataset.target_behavior]
                else:
                    prefix_allowed_tokens_fn = self.prefix_allowed_tokens
            else:
                if self.backbone == 'Qwen3':
                    max_new_tokens = self.item_len
                    inputs.input_ids = inputs.input_ids[:, :-max_new_tokens]
                    inputs.attention_mask = inputs.attention_mask[:, :-max_new_tokens]
                decoder_input_ids = [[self.config.decoder_start_token_id] for _ in targets]
                prefix_allowed_tokens_fn = self.prefix_allowed_tokens
            batch_size = len(targets)

            if self.backbone == 'Qwen3':
                output: GenerateBeamOutput = (
                    self.model
                    if isinstance(self.model, GenerationMixin)
                    else
                    self.model.module
                ).generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )
            else:
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

            if self.backbone == 'Qwen3':
                output_ids = output_ids[:, -self.item_len:]

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

    def test(self, num_beams: int) -> list[dict[str, float]]:
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
        Test the MB decoder using the provided arguments.
        """
        self.init(seed, False)
        if backbone == 'TIGER':
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(ckpt_path, legacy=True)
            self.model = TIGER.from_pretrained(ckpt_path).to(self.device)
            self.config: T5Config = self.model.config
        elif backbone == 'PBATransformers':
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(ckpt_path, legacy=True)
            self.model = PBATransformersForConditionalGeneration.from_pretrained(ckpt_path).to(self.device)
            self.config: PBATransformerConfig = self.model.config
        elif backbone == 'Qwen3':
            self.tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(ckpt_path)
            self.model = Qwen3WithTemperature.from_pretrained(ckpt_path).to(self.device)
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False)[0]
            self.config: Qwen3Config = self.model.config
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        assert isinstance(self.model, GenerationMixin), "Model must be a generation model."

        self.datasets = [load_MB_test_dataset(
            dataset,
            data_path,
            max_his_len,
            index_file,
            test_task,
        )]
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

        if backbone == 'Qwen3':
            collator = DecoderOnlyTestCollator(self.tokenizer)
        else:
            collator = EncoderDecoderTestCollator(self.tokenizer)

        for test_dataset in self.datasets:
            test_dataset.get_all_items()
        self.all_items = self.datasets[0].get_all_items()
        self.collision_info = self.check_collision_items(filter)

        self.all_behavior_items = self.datasets[0].get_all_items("all")
        item_reps = list(self.all_behavior_items)
        items_tokens = self.tokenizer.batch_encode_plus(item_reps, add_special_tokens=False)["input_ids"]
        self.item_len = len(items_tokens[0])
        self.sole_item_len = len(self.tokenizer.encode(next(iter(self.all_items)), add_special_tokens=False))
        last_token_set: set[int] = set([tokens[-1] for tokens in items_tokens])
        last_token_set.add(self.config.pad_token_id)  # Ensure pad token is included
        self.info("Complete get all behavior items last token set.")
        if backbone == 'Qwen3':
            candidate_trie = Trie(items_tokens)
            self.prefix_allowed_tokens = prefix_allowed_tokens_fn_by_last_token(candidate_trie, last_token_set)
        else:
            candidate_tokens = self.tokenizer.batch_encode_plus(list(self.all_behavior_items))["input_ids"]
            # Add decoder start token id to each candidate
            candidate_tokens = [[self.config.decoder_start_token_id] + tokens for tokens in candidate_tokens]
            candidate_trie = Trie(candidate_tokens)
            self.prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
        self.info("Complete building all behavior candidate trie for prefix allowed tokens function.")
        self.prefix_allowed_tokens_by_behavior: dict[str, Callable[[int, torch.Tensor], list[int]]] = {}
        behaviors = self.datasets[0].behaviors
        for behavior in behaviors:
            all_items = self.datasets[0].get_all_items(behavior)
            if backbone == 'Qwen3':
                candidate_tokens = self.tokenizer.batch_encode_plus(list(all_items), add_special_tokens=False)["input_ids"]
                behavior_trie = Trie(candidate_tokens)
                self.prefix_allowed_tokens_by_behavior[behavior] = prefix_allowed_tokens_fn_by_last_token(behavior_trie, last_token_set)
            else:
                candidate_tokens = self.tokenizer.batch_encode_plus(list(all_items))["input_ids"]
                # Add decoder start token id to each candidate
                candidate_tokens = [[self.config.decoder_start_token_id] + tokens for tokens in candidate_tokens]
                behavior_trie = Trie(candidate_tokens)
                self.prefix_allowed_tokens_by_behavior[behavior] = prefix_allowed_tokens_fn(behavior_trie)
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
        self.info(["Complete loading test datasets and collators."] + [
            f"Dataset {i} num: {len(test_dataset)}" for i, test_dataset in enumerate(self.datasets)
        ])

        self.model.eval()
        self.metric_list = metrics.split(",")
        self.backbone = backbone
        results = self.test(num_beams)
        logger.success("======================================================")
        logger.success("Results:")
        for res in results:
            logger.success("======================================================")
            logger.success(f"{res['eval_type']} results:")
            for m in res:
                if isinstance(res[m], float):
                    logger.success(f"\t{m} = {res[m]:.4f}")
        logger.success("======================================================")
        if self.local_rank == 0:
            ensure_dir(os.path.dirname(results_file))
            with open(results_file, "w") as f:
                json.dump(results, f, indent=4)
        logger.success(f"Results saved to {results_file}.")

        self.finish(False)
