import os
import json
import torch
import numpy as np
from loguru import logger

from SeqRec.tasks.base import Task
from SeqRec.datasets.loading_SMB import load_SMB_test_dataset
from SeqRec.datasets.SMB_dataset import BaseSMBDataset
from SeqRec.evaluation.ranking import get_topk_results, get_metrics_results
from SeqRec.utils.futils import ensure_dir
from SeqRec.utils.parse import SubParsersAction, parse_global_args, parse_dataset_args
from SeqRec.utils.pipe import get_tqdm


class TestSMBRule(Task):
    """
    Test a rule-based predictor for SMB SeqRec tasks.
    """

    @staticmethod
    def parser_name() -> str:
        return "test_SMB_rule"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        """
        Add subparsers for the TestSMBRule task.
        """
        parser = sub_parsers.add_parser("test_SMB_rule", help="Test a rule-based predictor for SMB SeqRec tasks.")
        parser = parse_global_args(parser)
        parser = parse_dataset_args(parser)
        parser.add_argument(
            "--results_file",
            type=str,
            default="./results/test.json",
            help="result output path",
        )
        parser.add_argument(
            "--num_items",
            type=int,
            default=20,
            help="Number of items to predict for each sample.",
        )
        parser.add_argument(
            "--unique",
            action="store_true",
            help="Whether to ensure unique items in the prediction. Default is False.",
        )
        parser.add_argument(
            "--metrics",
            type=str,
            default="hit@1,hit@5,hit@10,recall@1,recall@5,recall@10,ndcg@5,ndcg@10",
            help="test metrics, separate by comma",
        )
        parser.add_argument("--test_task", type=str, default="SeqRec")

    def check_collision_items(self) -> list[dict[str, int | float]]:
        ret_list = []
        for behavior in self.dataset.behaviors:
            collision_cnt = 0
            for test_sample in self.dataset:
                if test_sample["behavior"] != behavior:
                    continue
                target_items = test_sample["labels"]
                for target_item in target_items:
                    if target_item in self.dataset.collision_items:
                        collision_cnt += 1
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
            ret_list.append(ret)
        return ret_list

    def test_all_behaviors(self, dataset: BaseSMBDataset, num_items: int, unique: bool) -> dict[str, dict[str, float]]:
        self.info(f"Start testing all behaviors with {len(dataset)} samples.")
        behavior_results: dict[dict[str, float]] = {}
        pbar = get_tqdm(desc="Testing", total=len(dataset))

        for sample in dataset:
            behaviors = sample["behavior"]
            behaviors_set = set(behaviors)
            behaviors_array = np.array(behaviors)
            for behavior in behaviors_set:
                indices = np.where(behaviors_array == behavior)[0]
                if behavior not in behavior_results:
                    behavior_results[behavior] = {m: 0.0 for m in self.metric_list}
                    behavior_results[behavior]['cnt'] = 0
                behavior_results[behavior]['cnt'] += 1
                targets = [[sample["labels"][indice] for indice in indices]]
                scores = torch.arange(num_items, 0, -1)
                if unique:
                    item_list = sample["inters_item_list"][::-1]
                    history: list[str] = []
                    for item_rep in item_list:
                        if item_rep not in history:
                            history.append(item_rep)
                    history = history[:num_items]
                else:
                    history: list[str] = sample["inters_item_list"][::-1][:num_items]  # Get the last num_items items in reverse order
                output_str = [dataset.get_behavior_item(item, behavior) for item in history]

                topk_res = get_topk_results(
                    output_str,
                    scores,
                    targets,
                    num_items,
                )

                batch_metrics_res = get_metrics_results(topk_res, self.metric_list, targets)
                for m, res in batch_metrics_res.items():
                    behavior_results[behavior][m] += res
            pbar.update(1)
        if pbar:
            pbar.close()

        for behavior in behavior_results:
            for m in self.metric_list:
                behavior_results[behavior][m] /= behavior_results[behavior]['cnt']

        return behavior_results

    def test(self, num_items: int, unique: bool) -> list[dict[str, float]]:
        results = []
        behavior_results = self.test_all_behaviors(self.dataset, num_items, unique)
        merge_results = {m: 0.0 for m in self.metric_list}
        total = 0
        for i, behavior in enumerate(self.dataset.behaviors):
            result = behavior_results[behavior]
            result['eval_type'] = f"Behavior {behavior}"
            result['collision_info'] = self.collision_info[i]
            results.append(result)
            for m in self.metric_list:
                assert m in result, f"Metric {m} not found in results for behavior {behavior}."
                merge_results[m] += result[m] * result['cnt']
            total += result['cnt']
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
        results_file: str,
        num_items: int,
        unique: bool,
        metrics: str,
        test_task: str,
        *args,
        **kwargs
    ):
        """
        Test the SMB decoder using the provided arguments.
        """
        self.dataset = load_SMB_test_dataset(
            dataset,
            data_path,
            max_his_len,
            index_file,
            test_task,
        )
        self.all_items = self.dataset.get_all_items()
        self.collision_info = self.check_collision_items()

        self.metric_list = metrics.split(",")
        results = self.test(num_items, unique)
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
