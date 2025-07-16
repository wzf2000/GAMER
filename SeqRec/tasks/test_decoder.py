import os
import json
import torch
import torch.distributed as dist
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import T5Tokenizer, T5ForConditionalGeneration, BatchEncoding
from transformers.generation.utils import GenerateBeamOutput

from SeqRec.tasks.multi_gpu import MultiGPUTask
from SeqRec.datasets.seq_dataset import load_test_dataset
from SeqRec.datasets.collator import TestCollator
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

    def check_collision_items(self, filter: bool = False) -> dict[str, int | float]:
        collision_cnt = 0
        new_inter_data = []
        for i, test_sample in enumerate(self.test_data):
            target_item = test_sample["labels"]
            if target_item in self.test_data.collision_items:
                collision_cnt += 1
            else:
                new_inter_data.append(self.test_data.inter_data[i])
        if self.local_rank == 0:
            logger.info(f"Total test data num: {len(self.test_data)}")
            logger.info(f"Collision items num: {len(self.test_data.collision_items)}")
            logger.info(f"Collision sample num: {collision_cnt}")
            logger.info(f"Collision items ratio: {collision_cnt / len(self.test_data)}")
        ret = {
            "total": len(self.test_data),
            "collision_items": len(self.test_data.collision_items),
            "collision_samples": collision_cnt,
            "collision_ratio": collision_cnt / len(self.test_data),
        }
        if filter:
            # Filter out the collision items from the test data
            self.test_data.inter_data = new_inter_data
            if self.local_rank == 0:
                logger.info(f"Filtered test data num: {len(self.test_data)}")
        return ret

    def test(self, num_beams: int) -> dict[str, float]:
        results: dict[str, float] = {}
        total = 0
        if self.local_rank == 0:
            pbar = tqdm(total=len(self.loader), desc="Testing")

        for batch in self.loader:
            batch: tuple[BatchEncoding, list[str]]
            inputs = batch[0].to(self.device)
            targets = batch[1]
            batch_size = len(targets)

            output: GenerateBeamOutput = (
                self.model
                if isinstance(self.model, T5ForConditionalGeneration)
                else
                self.model.module
            ).generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=10,
                prefix_allowed_tokens_fn=self.prefix_allowed_tokens,
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
            dist.barrier()

        dist.barrier()
        for m in results:
            results[m] = results[m] / total

        return results

    def invoke(
        self,
        # global arguments
        seed: int,
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
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(ckpt_path, legacy=True)
        self.model = T5ForConditionalGeneration.from_pretrained(ckpt_path).to(self.device)
        self.test_data = load_test_dataset(
            dataset,
            data_path,
            max_his_len,
            index_file,
            test_task,
        )
        if self.ddp:
            self.sampler = DistributedSampler(
                self.test_data,
                num_replicas=self.world_size,
                rank=self.local_rank,
            )
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            self.sampler = None
        collator = TestCollator(self.tokenizer)
        self.all_items = self.test_data.get_all_items()
        self.collision_info = self.check_collision_items(filter)
        candidate_trie = Trie(
            [[0] + self.tokenizer.encode(candidate) for candidate in self.all_items]
        )
        self.prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
        self.loader = DataLoader(
            self.test_data,
            batch_size=test_batch_size,
            collate_fn=collator,
            sampler=self.sampler,
            num_workers=2,
            pin_memory=True,
        )
        if self.local_rank == 0:
            logger.info(f"Data num: {len(self.test_data)}")

        self.model.eval()
        self.metric_list = metrics.split(",")
        results = self.test(num_beams)
        if self.local_rank == 0:
            logger.success("======================================================")
            logger.success("Results:")
            for m in results:
                logger.success(f"\t{m} = {results[m]:.4f}")
            logger.success("======================================================")
            results['collision_info'] = self.collision_info
            ensure_dir(os.path.dirname(results_file))
            with open(results_file, "w") as f:
                json.dump(results, f, indent=4)
            logger.success(f"Results saved to {results_file}.")

        self.finish(False)
