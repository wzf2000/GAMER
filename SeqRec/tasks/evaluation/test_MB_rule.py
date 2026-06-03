import os
import json
import torch
import numpy as np
from loguru import logger

from SeqRec.tasks.base import Task
from SeqRec.datasets.loaders.multi_behavior import load_MB_test_dataset
from SeqRec.datasets.multi_behavior import BaseMBDataset
from SeqRec.evaluation.ranking import get_topk_results, get_metrics_results
from SeqRec.utils.fs import ensure_dir
from SeqRec.utils.args import SubParsersAction, parse_global_args, parse_dataset_args
from SeqRec.utils.runtime import get_tqdm


class TestMBRule(Task):
    """
    Test a rule-based predictor for MB SeqRec tasks.

    The rule is: predict the last ``num_items`` items in the user's interaction
    history (optionally deduplicated), wrapped with the sample's target behavior,
    as the ranked candidate list.
    """

    @staticmethod
    def parser_name() -> str:
        return "test_MB_rule"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        parser = sub_parsers.add_parser("test_MB_rule", help="Test a rule-based predictor for MB SeqRec tasks.")
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
        parser.add_argument("--test_task", type=str, default="mb_explicit")

    def check_collision_items(self) -> list[dict[str, int | float]]:
        ret_list = []
        for behavior in self.dataset.behaviors:
            collision_cnt = 0
            total_cnt = 0
            for test_sample in self.dataset:
                if test_sample["behavior"] != behavior:
                    continue
                total_cnt += 1
                target_item = test_sample["labels"]
                if target_item in self.dataset.collision_items:
                    collision_cnt += 1
            self.info([
                f"Behavior: {behavior}",
                f"Total test data num: {total_cnt}",
                f"Collision items num: {len(self.dataset.collision_items)}",
                f"Collision sample num: {collision_cnt}",
                f"Collision items ratio: {collision_cnt / total_cnt:.4f}" if total_cnt > 0 else "N/A",
            ])
            ret_list.append({
                "behavior": behavior,
                "total": total_cnt,
                "collision_items": len(self.dataset.collision_items),
                "collision_samples": collision_cnt,
                "collision_ratio": collision_cnt / total_cnt if total_cnt > 0 else 0.0,
            })
        return ret_list

    def test_all_behaviors(self, dataset: BaseMBDataset, num_items: int, unique: bool) -> dict[str, dict[str, float]]:
        self.info(f"Start testing all behaviors with {len(dataset)} samples.")
        behavior_results: dict[str, dict[str, float]] = {}
        pbar = get_tqdm(desc="Testing", total=len(dataset))

        for sample in dataset:
            behavior: str = sample["behavior"]
            uid: str | None = sample.get("uid")
            target: str = sample["labels"]

            if uid is None:
                pbar.update(1)
                continue

            if behavior not in behavior_results:
                behavior_results[behavior] = {m: 0.0 for m in self.metric_list}
                behavior_results[behavior]["cnt"] = 0
            behavior_results[behavior]["cnt"] += 1

            # Raw history items (all items stored for this user, last one is the test target)
            raw_items: list[str] = dataset.remapped_inters[uid][:-1]
            history = raw_items[::-1]  # most recent first

            if unique:
                seen: list[str] = []
                for item in history:
                    if item not in seen:
                        seen.append(item)
                history = seen

            history = history[:num_items]
            output_str = [dataset.get_behavior_item(item, behavior) for item in history]

            scores = torch.arange(num_items, 0, -1, dtype=torch.float)
            targets = [[target]]

            topk_res = get_topk_results(output_str, scores, targets, num_items)
            batch_metrics_res = get_metrics_results(topk_res, self.metric_list, targets)
            for m, res in batch_metrics_res.items():
                behavior_results[behavior][m] += res

            pbar.update(1)

        if pbar:
            pbar.close()

        for behavior in behavior_results:
            cnt = behavior_results[behavior]["cnt"]
            for m in self.metric_list:
                behavior_results[behavior][m] /= cnt if cnt > 0 else 1

        return behavior_results

    def test(self, num_items: int, unique: bool) -> list[dict[str, float]]:
        results = []
        behavior_results = self.test_all_behaviors(self.dataset, num_items, unique)
        merge_results = {m: 0.0 for m in self.metric_list}
        total = 0

        for i, behavior in enumerate(self.dataset.behaviors):
            if behavior not in behavior_results:
                continue
            result = behavior_results[behavior]
            result["eval_type"] = f"Behavior {behavior}"
            result["collision_info"] = self.collision_info[i]
            results.append(result)
            for m in self.metric_list:
                assert m in result, f"Metric {m} not found in results for behavior {behavior}."
                merge_results[m] += result[m] * result["cnt"]
            total += result["cnt"]

        if total > 0:
            for m in merge_results:
                merge_results[m] /= total
        merge_results["eval_type"] = "Merged Behavior"
        results.append(merge_results)
        return results

    def invoke(
        self,
        # global arguments
        seed: int,
        backbone: str,
        base_model: str,
        output_dir: str,
        # dataset arguments
        data_path: str,
        tasks: str,
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
        **kwargs,
    ):
        self.dataset = load_MB_test_dataset(
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
