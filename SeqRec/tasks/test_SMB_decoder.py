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
from SeqRec.datasets.loading_SMB import load_SMB_test_dataset
from SeqRec.datasets.MB_dataset import EvaluationType
from SeqRec.datasets.SMB_dataset import BaseSMBDataset
from SeqRec.datasets.collator import EncoderDecoderTestCollator, DecoderOnlyTestCollator
from SeqRec.models.TIGER import TIGER
from SeqRec.models.PBATransformers import PBATransformerConfig, PBATransformersForConditionalGeneration
from SeqRec.models.PBATransformers_session import PBATransformerConfigSession, PBATransformersForConditionalGenerationSession
from SeqRec.models.Qwen import Qwen3WithTemperature
from SeqRec.models.Qwen_Moe import Qwen3WithTemperatureMoe
from SeqRec.models.Qwen_session import Qwen3SessionWithTemperature
from SeqRec.evaluation.ranking import get_topk_results, get_metrics_results
from SeqRec.generation.trie import Trie, prefix_allowed_tokens_fn, prefix_allowed_tokens_fn_by_last_token
from SeqRec.utils.futils import ensure_dir
from SeqRec.utils.parse import SubParsersAction, parse_global_args, parse_dataset_args
from SeqRec.utils.pipe import get_tqdm


class TestSMBDecoder(MultiGPUTask):
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
            default="hit@1,hit@5,hit@10,recall@1,recall@5,recall@10,ndcg@5,ndcg@10",
            help="test metrics, separate by comma",
        )
        parser.add_argument("--test_task", type=str, default="SeqRec")

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

        for batch in loader:
            batch: tuple[BatchEncoding, list[list[str]], torch.LongTensor]
            inputs = batch[0].to(self.device)
            targets = batch[1]
            batch_size = len(targets)
            behaviors: list[str] = [behavior for _ in range(batch_size)]
            dataset: BaseSMBDataset = loader.dataset
            behavior_tokens = [''.join(dataset.get_behavior_tokens(b)) for b in behaviors]
            behavior_tokens = self.tokenizer.batch_encode_plus(behavior_tokens, add_special_tokens=False)
            bahavior_attention_mask = behavior_tokens["attention_mask"]
            behavior_tokens = behavior_tokens["input_ids"]
            if self.backbone == 'Qwen3' or self.backbone == "Qwen3Moe":
                inputs.input_ids = torch.cat([inputs.input_ids, torch.tensor(behavior_tokens, device=self.device)], dim=1)
                inputs.attention_mask = torch.cat([inputs.attention_mask, torch.tensor(bahavior_attention_mask, device=self.device)], dim=1)
            else:
                decoder_input_ids = [[self.config.decoder_start_token_id] + tokens for tokens in behavior_tokens]
            prefix_allowed_tokens_fn = self.prefix_allowed_tokens_by_behavior[behavior]

            self.info("Start generating items for the batch.")
            if self.backbone == 'Qwen3' or self.backbone == "Qwen3Moe":
                output: GenerateBeamOutput = (
                    self.model
                    if isinstance(self.model, GenerationMixin)
                    else
                    self.model.module
                ).generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=self.sole_item_len,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )
            elif self.backbone in ["PBATransformers_session", "PBATransformers_time"]:
                output: GenerateBeamOutput = (
                    self.model
                    if isinstance(self.model, GenerationMixin)
                    else
                    self.model.module
                ).generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    decoder_input_ids=torch.tensor(decoder_input_ids, device=self.device),
                    max_new_tokens=self.sole_item_len,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                    session_ids=inputs.session_ids,
                    time=inputs.time,
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
                    max_new_tokens=self.sole_item_len,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )
            self.info("Finished generating items for the batch.")
            output_ids = output.sequences
            scores = output.sequences_scores

            if self.backbone == 'Qwen3' or self.backbone == "Qwen3Moe":
                output_ids = output_ids[:, -self.item_len:]

            output_str = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            self.info("Finished decoding output ids to strings.")

            topk_res = get_topk_results(
                output_str,
                scores,
                targets,
                num_beams,
            )
            self.info(f"Got top-{num_beams} results for the batch with {len(topk_res)} items.")

            if self.ddp:
                batch_size_gather_list = [None for _ in range(self.world_size)]
                dist.all_gather_object(obj=batch_size, object_list=batch_size_gather_list)
                total += sum(batch_size_gather_list)
                res_gather_list = [None for _ in range(self.world_size)]
                dist.all_gather_object(obj=topk_res, object_list=res_gather_list)
                targets_gather_list = [None for _ in range(self.world_size)]
                dist.all_gather_object(obj=targets, object_list=targets_gather_list)

                all_device_topk_res = []
                for ga_res in res_gather_list:
                    all_device_topk_res += ga_res
                topk_res = all_device_topk_res

                all_device_targets = []
                for ga_targets in targets_gather_list:
                    all_device_targets += ga_targets
                targets = all_device_targets
            else:
                total += batch_size
            self.info(f"Gathered top-{num_beams} results from all devices, total samples: {total}.")

            batch_metrics_res = get_metrics_results(topk_res, self.metric_list, targets)
            for m, res in batch_metrics_res.items():
                if m not in results:
                    results[m] = res
                else:
                    results[m] += res
            self.info(f"Calculated metrics for the batch: {batch_metrics_res}.")

            if self.local_rank == 0:
                show_metric_keys = self.metric_list[:2]  # Show only the first two metrics
                show_metric_dict = {
                    m: f"{results[m] / total:.4f}" for m in show_metric_keys if m in results
                }
                pbar.set_postfix(show_metric_dict)
                pbar.update(1)
            self.info(f"Updated progress bar for behavior {behavior} with {batch_size} samples.")
            if self.ddp:
                dist.barrier()
            self.info(f"Finished processing batch with {batch_size} samples for behavior {behavior}.")
        if pbar:
            pbar.close()

        self.info(f"Finished testing behavior {behavior} with {total} samples.")
        for m in results:
            results[m] = results[m] / total

        return results

    def test(self, num_beams: int) -> list[dict[str, float]]:
        results = []
        merge_results = {m: 0.0 for m in self.metric_list}
        total = 0
        for i, behavior in enumerate(self.base_dataset.behaviors):
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
        *args,
        **kwargs
    ):
        """
        Test the SMB decoder using the provided arguments.
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
        elif backbone in ['PBATransformers_session', 'PBATransformers_time']:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(ckpt_path, legacy=True)
            self.model = PBATransformersForConditionalGenerationSession.from_pretrained(ckpt_path).to(self.device)
            self.config: PBATransformerConfigSession = self.model.config
        elif backbone == 'Qwen3':
            self.tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(ckpt_path)
            self.model = Qwen3WithTemperature.from_pretrained(ckpt_path).to(self.device)
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False)[0]
            self.config: Qwen3Config = self.model.config
        elif backbone == 'Qwen3Moe':
            self.tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(ckpt_path)
            self.model = Qwen3WithTemperatureMoe.from_pretrained(ckpt_path).to(self.device)
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False)[0]
            self.config: Qwen3Config = self.model.config
        elif backbone == 'Qwen3Session':
            self.tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(ckpt_path)
            self.model = Qwen3SessionWithTemperature.from_pretrained(ckpt_path).to(self.device)
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False)[0]
            self.config: Qwen3Config = self.model.config
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        assert isinstance(self.model, GenerationMixin), "Model must be a generation model."

        self.base_dataset = load_SMB_test_dataset(
            dataset,
            data_path,
            max_his_len,
            index_file,
            test_task,
        )
        self.datasets: list[BaseSMBDataset] = []
        for behavior in self.base_dataset.behaviors:
            self.datasets.append(self.base_dataset.filter_by_behavior(behavior))
            self.info(f"Loaded dataset for behavior {behavior} with {len(self.datasets[-1])} samples.")

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

        if backbone in ['Qwen3', 'Qwen3Moe', 'Qwen3Session']:
            collator = DecoderOnlyTestCollator(self.tokenizer)
        else:
            collator = EncoderDecoderTestCollator(self.tokenizer)

        for test_dataset in self.datasets:
            test_dataset.get_all_items()
        self.all_items = self.datasets[0].get_all_items()
        self.collision_info = self.check_collision_items()

        self.all_behavior_items = self.datasets[0].get_all_items("all")
        item_reps = list(self.all_behavior_items)
        items_tokens = self.tokenizer.batch_encode_plus(item_reps, add_special_tokens=False)["input_ids"]
        self.item_len = len(items_tokens[0])
        self.sole_item_len = len(self.tokenizer.encode(next(iter(self.all_items)), add_special_tokens=False))

        last_token_set: set[int] = set([tokens[-1] for tokens in items_tokens])
        last_token_set.add(self.config.pad_token_id)  # Ensure pad token is included
        self.info("Complete get all behavior items last token set.")

        if backbone in ['Qwen3', 'Qwen3Moe', 'Qwen3Session']:
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
            if backbone in ['Qwen3', 'Qwen3Moe', 'Qwen3Session']:
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
        if self.local_rank == 0:
            logger.success("======================================================")
            logger.success("Results:")
            for res in results:
                logger.success("======================================================")
                logger.success(f"{res['eval_type']} results:")
                for m in res:
                    if isinstance(res[m], float):
                        logger.success(f"\t{m} = {res[m]:.4f}")
            logger.success("======================================================")
            ensure_dir(os.path.dirname(results_file))
            with open(results_file, "w") as f:
                json.dump(results, f, indent=4)
            logger.success(f"Results saved to {results_file}.")

        self.finish(False)
