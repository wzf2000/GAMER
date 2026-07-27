import os
import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from SeqRec.datasets.collators.generative import DecoderOnlyRankingCollator
from SeqRec.datasets.loaders.session_behavior import load_SMB_test_dataset, load_SMB_valid_dataset
from SeqRec.datasets.session_behavior import SMBRankingDatasetForDecoder
from SeqRec.evaluation.ranking import BINARY_METRICS, DEFAULT_BINARY_METRICS, BinaryMetricAccumulator, get_metrics_results, get_ranked_item_hits, rank_items_by_scores
from SeqRec.tasks.evaluation.base import _BaseDecoderTestTask
from SeqRec.utils.args import SubParsersAction, parse_dataset_args, parse_generation_eval_args, parse_global_args
from SeqRec.utils.runtime import get_tqdm


class TestSMBRankingDecoder(_BaseDecoderTestTask):
    @staticmethod
    def parser_name() -> str:
        return "test_SMB_ranking_decoder"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        parser = sub_parsers.add_parser(
            "test_SMB_ranking_decoder",
            help="Test a SMB decoder ranking scorer.",
        )
        parser = parse_global_args(parser)
        parser = parse_dataset_args(parser)
        parse_generation_eval_args(
            parser,
            metrics=",".join(DEFAULT_BINARY_METRICS),
            include_behaviors=True,
            include_valid_loss=True,
        )

    @staticmethod
    def _max_metric_k(metrics: list[str]) -> int:
        metric_ks = [int(metric.split("@")[1]) for metric in metrics if "@" in metric]
        return max(metric_ks) if metric_ks else 0

    def _align_sequence(self, sequence: list[int], length: int, pad_value: int = 0) -> list[int]:
        if len(sequence) > length:
            sequence = sequence[-length:]
        if len(sequence) < length:
            sequence = sequence + [pad_value] * (length - len(sequence))
        return sequence

    def _score_candidate_batch(
        self,
        *,
        sample: dict,
        candidates: list[str],
        item_len: int,
    ) -> torch.Tensor:
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"
        texts = [sample["input_ids"] + candidate for candidate in candidates]
        inputs = self.tokenizer(
            text=texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        max_length = inputs["input_ids"].shape[1]
        relation_actions = [
            self._align_sequence(sample["relation_actions"] + [0] * item_len, max_length)
            for _candidate in candidates
        ]
        session_id = sample["target_session_id"]
        session_ids = [
            self._align_sequence(sample["session_ids"] + [session_id] * item_len, max_length)
            for _candidate in candidates
        ]
        next_extended = (max(sample["extended_session_ids"]) + 1) if sample["extended_session_ids"] else 0
        extended_session_ids = [
            self._align_sequence(
                sample["extended_session_ids"] + [next_extended + index for index in range(item_len)],
                max_length,
            )
            for _candidate in candidates
        ]

        model_inputs = {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device),
            "relation_actions": torch.tensor(relation_actions, dtype=torch.long, device=self.device),
            "actions": torch.tensor(relation_actions, dtype=torch.long, device=self.device),
            "session_ids": torch.tensor(session_ids, dtype=torch.long, device=self.device),
            "extended_session_ids": torch.tensor(extended_session_ids, dtype=torch.long, device=self.device),
            "user_id": torch.tensor([sample["user_id"]] * len(candidates), dtype=torch.long, device=self.device),
            "use_cache": False,
        }
        if self.ranking_score_type in {"hidden_head", "llm_pair"}:
            model_inputs["use_ranking_head"] = True
        else:
            model_inputs["logits_to_keep"] = 1
        with torch.no_grad():
            output = self.model(**model_inputs)
            if self.ranking_score_type in {"hidden_head", "llm_pair"}:
                return output.logits.squeeze(-1).detach().cpu()
            return output.logits[:, -1, self.ranking_target_token_id].detach().cpu()

    def _score_sample(
        self,
        *,
        sample: dict,
        item_len: int,
        candidate_batch_size: int,
    ) -> torch.Tensor:
        scores = []
        for start in range(0, len(self.all_items), candidate_batch_size):
            candidates = self.all_items[start : start + candidate_batch_size]
            scores.append(
                self._score_candidate_batch(
                    sample=sample,
                    candidates=candidates,
                    item_len=item_len,
                )
            )
        return torch.cat(scores, dim=0)

    def test_single_behavior(
        self,
        loader: DataLoader,
        behavior: str,
        candidate_batch_size: int,
    ) -> dict[str, float]:
        if len(loader.dataset) == 0:
            return {metric: 0.0 for metric in self.metric_list}
        dataset: SMBRankingDatasetForDecoder = loader.dataset
        if behavior != dataset.target_behavior:
            raise ValueError("The ranking head scores only the target/max-level behavior.")
        item_len = len(self.tokenizer.encode(self.all_items[0], add_special_tokens=False))
        max_k = self._max_metric_k(self.metric_list)
        results: dict[str, float] = {}
        user_metric_dict: dict[str, dict[str, float]] = {metric: {} for metric in self.metric_list}
        total = 0
        pbar = get_tqdm(desc=f"Ranking behavior {behavior}", total=len(loader))

        for batch in loader:
            for sample in batch:
                scores = self._score_sample(
                    sample=sample,
                    item_len=item_len,
                    candidate_batch_size=candidate_batch_size,
                )
                ranked_items = rank_items_by_scores(self.all_items, scores)
                hit_limit = None if "auc" in {metric.lower() for metric in self.metric_list} else max_k
                topk_hits = [get_ranked_item_hits(ranked_items, sample["labels"], hit_limit)]
                sample_metrics = get_metrics_results(
                    topk_hits,
                    self.metric_list,
                    [sample["labels"]],
                    list_output=True,
                )
                uid = sample.get("uid", str(total))
                for metric, values in sample_metrics.items():
                    user_metric_dict[metric][uid] = values[0]
                    results[metric] = results.get(metric, 0.0) + values[0]
                total += 1

            if pbar:
                show = {
                    metric: f"{results[metric] / max(total, 1):.4f}"
                    for metric in self.metric_list[:2]
                    if metric in results
                }
                pbar.set_postfix(show)
                pbar.update(1)
            if self.ddp:
                dist.barrier()

        if pbar:
            pbar.close()
        total = self._gather_sum(total)
        for metric in list(results):
            results[metric] = self._gather_sum(results[metric]) / total

        gathered_user_metrics = self._gather_concat([user_metric_dict])
        merged_user_metrics: dict[str, dict[str, float]] = {metric: {} for metric in self.metric_list}
        for one_rank_metrics in gathered_user_metrics:
            for metric, metric_by_uid in one_rank_metrics.items():
                merged_user_metrics[metric].update(metric_by_uid)

        save_path = os.path.join(
            self.results_file.replace(".json", ""),
            f"user_level_metrics_{behavior}.json",
        )
        self._save_user_metrics(merged_user_metrics, len(loader.dataset), save_path, results)
        return results

    def test(self, candidate_batch_size: int) -> list[dict[str, float]]:
        results = []
        merged = {metric: 0.0 for metric in self.metric_list}
        total = 0
        for loader, behavior in zip(self.loaders, self.behaviors):
            result = self.test_single_behavior(loader, behavior, candidate_batch_size)
            result["eval_type"] = f"Behavior {behavior}"
            results.append(result)
            dataset_len = len(loader.dataset)
            for metric in self.metric_list:
                merged[metric] += result[metric] * dataset_len
            total += dataset_len
        for metric in merged:
            merged[metric] = merged[metric] / total if total else 0.0
        merged["eval_type"] = "Merged Behavior"
        results.append(merged)
        return results

    def _is_cvr_auc_eval(self) -> bool:
        return all(m in BINARY_METRICS for m in self.metric_list)

    def _setup_cvr_samplers(self):
        if not self.ddp:
            return [None] * len(self.datasets)
        return [
            DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=False,
            )
            for dataset in self.datasets
        ]

    def test_cvr_auc(self, loader: DataLoader, candidate_batch_size: int) -> list[dict[str, float]]:
        dataset: SMBRankingDatasetForDecoder = loader.dataset
        target_behavior = dataset.target_behavior
        if len(dataset) == 0:
            result = {
                "eval_type": f"CVR {target_behavior}",
                "positive": 0.0,
                "negative": 0.0,
                "num_examples": 0.0,
            }
            for metric in self.metric_list:
                result[metric.lower()] = 0.0
            return [result]
        item_len = len(self.tokenizer.encode(dataset[0]["target_item"][0], add_special_tokens=False))
        records = []
        accumulator = BinaryMetricAccumulator(self.metric_list)
        pbar = get_tqdm(desc=f"CVR AUC {target_behavior}", total=len(loader))

        for batch in loader:
            for sample in batch:
                candidates = sample["target_item"]
                behaviors = sample["behavior"]
                for start in range(0, len(candidates), candidate_batch_size):
                    chunk_candidates = candidates[start : start + candidate_batch_size]
                    chunk_labels = [
                        1 if dataset.behavior_level[behavior] == dataset.max_behavior_level else 0
                        for behavior in behaviors[start : start + candidate_batch_size]
                    ]
                    scores = self._score_candidate_batch(
                        sample=sample,
                        candidates=chunk_candidates,
                        item_len=item_len,
                    )
                    chunk_scores = scores.tolist()
                    chunk_uids = [sample["uid"]] * len(chunk_labels)
                    accumulator.update(chunk_labels, chunk_scores, chunk_uids)
                    for offset, score in enumerate(chunk_scores):
                        item_index = start + offset
                        label = chunk_labels[offset]
                        key = (sample["uid"], sample["target_session_id"], item_index)
                        records.append((key, label, score))
            if pbar:
                progress = accumulator.compute(include_rank_metrics=False)
                show = {
                    m.lower(): f"{progress[m.lower()]:.4f}"
                    for m in self.metric_list
                    if m.lower() in progress
                }
                show["scored"] = len(records)
                pbar.set_postfix(show)
                pbar.update(1)

        records = self._gather_concat(records)
        deduped = {}
        for key, label, score in records:
            deduped[key] = (label, score)
        uid_list = [key[0] for key in deduped.keys()]
        labels = [label for label, _score in deduped.values()]
        scores = [score for _label, score in deduped.values()]
        result: dict = {
            "eval_type": f"CVR {target_behavior}",
        }
        final_accumulator = BinaryMetricAccumulator(self.metric_list)
        final_accumulator.update(labels, scores, uid_list)
        result.update(final_accumulator.compute())
        return [result]

    def invoke(
        self,
        seed: int,
        backbone: str,
        base_model: str,
        output_dir: str,
        data_path: str,
        tasks: str,
        dataset: str,
        index_file: str,
        max_his_len: int,
        ckpt_path: str,
        results_file: str,
        test_batch_size: int,
        num_beams: int,
        metrics: str,
        test_task: str,
        behaviors: list[str] | None,
        valid_loss: bool,
        *args,
        **kwargs,
    ):
        if not backbone.startswith("Qwen3TemporalHierarchical"):
            raise ValueError("SMB ranking decoder requires a Qwen3TemporalHierarchical backbone.")
        self.init(seed, False)
        self._load_model_via_registry(backbone, ckpt_path)
        self.ranking_score_type = getattr(
            self.model.config,
            "ranking_score_type",
            "hidden_head" if hasattr(self.model, "ranking_head") else "lm_target_token",
        )
        self.ranking_target_token_id = getattr(self.model.config, "ranking_target_token_id", None)
        if self.ranking_score_type in {"hidden_head", "llm_pair"} and not hasattr(self.model, "ranking_head"):
            raise ValueError("SMB ranking decoder checkpoint does not contain a ranking head. Please retrain it.")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.success(f"Model {backbone} has {total_params} parameters, {trainable_params} of them are trainable.")

        if valid_loss:
            self.datasets = [
                load_SMB_valid_dataset(dataset, data_path, max_his_len, index_file, test_task)
            ]
            collator = DecoderOnlyRankingCollator(self.tokenizer)
        else:
            self.base_dataset = load_SMB_test_dataset(dataset, data_path, max_his_len, index_file, test_task)
            if self.ranking_score_type == "lm_target_token" and self.ranking_target_token_id is None:
                target_behavior_tokens = self.base_dataset.get_behavior_tokens(self.base_dataset.target_behavior)
                if len(target_behavior_tokens) != 1:
                    raise ValueError("SMB ranking decoder requires exactly one target behavior token.")
                self.ranking_target_token_id = self.tokenizer.encode(
                    target_behavior_tokens[0],
                    add_special_tokens=False,
                )[0]
            self.metric_list = [metric.strip().lower() for metric in metrics.split(",") if metric.strip()]
            self.cvr_auc_eval = self._is_cvr_auc_eval() and behaviors is None
            if self.cvr_auc_eval:
                self.behaviors = [self.base_dataset.target_behavior]
                self.datasets = [self.base_dataset]
                self.info(
                    "Using fast CVR AUC evaluation: "
                    f"positive=max behavior level ({self.base_dataset.target_behavior}), "
                    "negative=other target-session behaviors."
                )
            elif behaviors is None:
                self.behaviors = [self.base_dataset.target_behavior]
                self.datasets = [
                    self.base_dataset.filter_by_behavior(behavior)
                    for behavior in self.behaviors
                ]
            else:
                self.behaviors = behaviors
                self.datasets = [
                    self.base_dataset.filter_by_behavior(behavior)
                    for behavior in self.behaviors
                ]
            if not self.cvr_auc_eval:
                for behavior, behavior_dataset in zip(self.behaviors, self.datasets):
                    self.info(f"Loaded ranking dataset for behavior {behavior} with {len(behavior_dataset)} samples.")
                self.all_items = sorted(self.base_dataset.get_all_items())
            collator = lambda batch: batch

        self.samplers = self._setup_cvr_samplers() if getattr(self, "cvr_auc_eval", False) else self._setup_ddp_for_datasets(self.datasets)
        loader_batch_size = test_batch_size if valid_loss else 1
        self.loaders = [
            DataLoader(
                test_dataset,
                batch_size=loader_batch_size,
                collate_fn=collator,
                sampler=sampler,
                num_workers=0,
                pin_memory=True,
            )
            for sampler, test_dataset in zip(self.samplers, self.datasets)
        ]

        self.model.eval()
        if not hasattr(self, "metric_list"):
            self.metric_list = metrics.split(",")
        self.results_file = results_file

        if valid_loss:
            self.validation()
        elif getattr(self, "cvr_auc_eval", False):
            results = self.test_cvr_auc(self.loaders[0], max(1, test_batch_size))
            self._save_results_and_log(results, results_file, multiple=True)
        else:
            results = self.test(max(1, test_batch_size))
            self._save_results_and_log(results, results_file, multiple=True)

        self.finish(False)
