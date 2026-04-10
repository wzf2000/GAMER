"""
Sparse-target-behavior comparative analysis.

Two modes in one task:
1. **Interesting-user discovery** — find users with very few target-behavior
   (e.g. conversion) interactions in history but rich auxiliary behaviors,
   where *our* model ranks high but the *baseline* ranks low.
2. **Bucketed statistics** — group all target-behavior test users by their
   history target-behavior count, compute HR@K / NDCG@K per bucket for both
   models and report the per-bucket delta.
"""

import os
import copy
import json
import torch
import numpy as np
from loguru import logger
from typing import Callable, TYPE_CHECKING
from torch.utils.data import DataLoader

from SeqRec.tasks.multi_gpu import MultiGPUTask
from SeqRec.datasets.loading_SMB import load_SMB_test_dataset
from SeqRec.datasets.SMB_dataset import BaseSMBDataset
from SeqRec.datasets.collator import DecoderOnlyTestCollator, EncoderDecoderTestCollator
from SeqRec.generation.trie import Trie, prefix_allowed_tokens_fn, prefix_allowed_tokens_fn_by_last_token
from SeqRec.evaluation.ranking import get_topk_results, get_metrics_results
from SeqRec.utils.futils import ensure_dir
from SeqRec.utils.parse import SubParsersAction, parse_global_args, parse_dataset_args
from SeqRec.utils.pipe import get_tqdm

if TYPE_CHECKING:
    from transformers.generation.utils import GenerateBeamOutput
    from transformers import BatchEncoding


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _hr_at_k(rank: int | None, k: int) -> float:
    return 1.0 if (rank is not None and rank <= k) else 0.0


def _ndcg_at_k(rank: int | None, k: int) -> float:
    return (1.0 / np.log2(rank + 1)) if (rank is not None and rank <= k) else 0.0


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class AnalyzeSparseTargetBehavior(MultiGPUTask):
    """
    Compare *our model* vs *baseline* on users whose history contains only a
    small number of target-behavior interactions (sparse conversion signal).
    """

    @staticmethod
    def parser_name() -> str:
        return "analyze_sparse_behavior"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        parser = sub_parsers.add_parser(
            "analyze_sparse_behavior",
            help="Sparse-target-behavior comparison: our model vs baseline.",
        )
        parser = parse_global_args(parser)
        parser = parse_dataset_args(parser)
        # Our model
        parser.add_argument("--ckpt_path", type=str, required=True,
                            help="Checkpoint path for our model.")
        # Baseline
        parser.add_argument("--baseline_ckpt_path", type=str, required=True,
                            help="Checkpoint path for the baseline model.")
        parser.add_argument("--baseline_backbone", type=str, default=None,
                            help="Backbone type for baseline (default: same as --backbone).")
        parser.add_argument("--baseline_max_his_len", type=int, default=None, required=True,
                            help="the max number of items in history sequence for baseline, -1 means no limit, required if --baseline_backbone is not None")
        # Dataset / inference
        parser.add_argument("--test_task", type=str, default="smb_explicit")
        parser.add_argument("--target_behavior", type=str, default=None,
                            help="Behavior used for evaluation (default: dataset max-level behavior).")
        parser.add_argument("--test_batch_size", type=int, default=16)
        parser.add_argument("--num_beams", type=int, default=20)
        # Analysis parameters
        parser.add_argument("--metrics", type=str, default="hit@10,ndcg@10",
                            help="Metrics for bucketed statistics.")
        parser.add_argument("--bucket_thresholds", type=str, default="3,6",
                            help="Comma-separated thresholds that define bucket edges. "
                                 "'3,6' → [0-2], [3-5], [6+].")
        parser.add_argument("--max_sparse_count", type=int, default=2,
                            help="Max history target-behavior count to be considered 'sparse'.")
        parser.add_argument("--max_interesting_users", type=int, default=20,
                            help="Max interesting-user examples to record in detail.")
        parser.add_argument("--interesting_top_k", type=int, default=10,
                            help="Our model must rank <= this AND baseline > this to be interesting.")
        # Output
        parser.add_argument("--results_file", type=str,
                            default="./results/sparse_behavior_analysis.json")

    # ------------------------------------------------------------------
    # Model loading (same pattern as test_SMB_decoder)
    # ------------------------------------------------------------------

    def _load_model_and_tokenizer(self, backbone: str, ckpt_path: str):
        if backbone == "TIGER":
            from transformers import T5Tokenizer
            from SeqRec.models.generative.TIGER import TIGER
            self.tokenizer = T5Tokenizer.from_pretrained(ckpt_path, legacy=True)
            self.model = TIGER.from_pretrained(ckpt_path).to(self.device)
        elif backbone == "PBATransformer":
            from transformers import T5Tokenizer
            from SeqRec.models.generative.PBATransformer import PBATransformerForConditionalGeneration
            self.tokenizer = T5Tokenizer.from_pretrained(ckpt_path, legacy=True)
            self.model = PBATransformerForConditionalGeneration.from_pretrained(ckpt_path).to(self.device)
        elif backbone == "Qwen3":
            from transformers import Qwen2Tokenizer
            from SeqRec.models.generative.Qwen3 import Qwen3WithTemperature
            self.tokenizer = Qwen2Tokenizer.from_pretrained(ckpt_path)
            self.model = Qwen3WithTemperature.from_pretrained(ckpt_path).to(self.device)
        elif backbone == "Qwen3Session":
            from transformers import Qwen2Tokenizer
            from SeqRec.models.generative.Qwen3Session import Qwen3SessionWithTemperature
            self.tokenizer = Qwen2Tokenizer.from_pretrained(ckpt_path)
            self.model = Qwen3SessionWithTemperature.from_pretrained(ckpt_path).to(self.device)
        elif backbone == "Qwen3Multi":
            from transformers import Qwen2Tokenizer
            from SeqRec.models.generative.Qwen3Multi import Qwen3MultiWithTemperature
            self.tokenizer = Qwen2Tokenizer.from_pretrained(ckpt_path)
            self.model = Qwen3MultiWithTemperature.from_pretrained(ckpt_path).to(self.device)
        elif backbone == "Qwen3SessionMulti":
            from transformers import Qwen2Tokenizer
            from SeqRec.models.generative.Qwen3SessionMulti import Qwen3SessionMultiWithTemperature
            self.tokenizer = Qwen2Tokenizer.from_pretrained(ckpt_path)
            self.model = Qwen3SessionMultiWithTemperature.from_pretrained(ckpt_path).to(self.device)
        elif backbone == "LlamaMulti":
            from transformers import Qwen2Tokenizer
            from SeqRec.models.generative.LlamaMulti import LlamaMultiWithTemperature
            self.tokenizer = Qwen2Tokenizer.from_pretrained(ckpt_path)
            self.model = LlamaMultiWithTemperature.from_pretrained(ckpt_path).to(self.device)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if hasattr(self.model, "config") and self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.encode(
                self.tokenizer.pad_token, add_special_tokens=False
            )[0]
        self.config = self.model.config

    # ------------------------------------------------------------------
    # Trie / prefix-constrained decoding
    # ------------------------------------------------------------------

    @staticmethod
    def _is_decoder_only_backbone(backbone: str) -> bool:
        return backbone in ("Qwen3", "Qwen3Session", "Qwen3Multi", "Qwen3SessionMulti", "LlamaMulti")

    def _build_tries(self, base_dataset: BaseSMBDataset, is_decoder_only: bool):
        all_beh_items = base_dataset.get_all_items("all")
        item_reps = list(all_beh_items)
        items_tokens = self.tokenizer.batch_encode_plus(item_reps, add_special_tokens=False)["input_ids"]
        self.item_len = len(items_tokens[0])
        self.sole_item_len = len(
            self.tokenizer.encode(next(iter(base_dataset.get_all_items())), add_special_tokens=False)
        )
        last_token_set = {tok[-1] for tok in items_tokens}
        last_token_set.add(self.config.pad_token_id)

        self.prefix_fn_by_behavior: dict[str, Callable] = {}
        for beh in base_dataset.behaviors:
            beh_items = list(base_dataset.get_all_items(beh))
            if is_decoder_only:
                cand_tokens = self.tokenizer.batch_encode_plus(beh_items, add_special_tokens=False)["input_ids"]
                trie = Trie(cand_tokens)
                self.prefix_fn_by_behavior[beh] = prefix_allowed_tokens_fn_by_last_token(trie, last_token_set)
            else:
                cand_tokens = self.tokenizer.batch_encode_plus(beh_items)["input_ids"]
                cand_tokens = [[self.config.decoder_start_token_id] + t for t in cand_tokens]
                trie = Trie(cand_tokens)
                self.prefix_fn_by_behavior[beh] = prefix_allowed_tokens_fn(trie)

    # ------------------------------------------------------------------
    # Single-batch beam-search inference
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        inputs: "BatchEncoding",
        behavior: str,
        dataset: BaseSMBDataset,
        num_beams: int,
        is_decoder_only: bool,
        backbone: str,
    ) -> tuple[list[str], torch.Tensor]:
        """Append behavior token, run beam search; return (output_str, scores)."""
        from transformers.generation import GenerationMixin

        batch_size = inputs.input_ids.shape[0]
        beh_str = "".join(dataset.get_behavior_tokens(behavior))
        beh_enc = self.tokenizer.batch_encode_plus(
            [beh_str] * batch_size, add_special_tokens=False
        )
        beh_ids = beh_enc["input_ids"]
        beh_mask = beh_enc["attention_mask"]

        inp = copy.copy(inputs)
        gen_model = self.model if isinstance(self.model, GenerationMixin) else self.model.module

        if is_decoder_only:
            inp.input_ids = torch.cat(
                [inp.input_ids, torch.tensor(beh_ids, device=self.device)], dim=1
            )
            inp.attention_mask = torch.cat(
                [inp.attention_mask, torch.tensor(beh_mask, device=self.device)], dim=1
            )
            beh_action = [[dataset.behavior_level[behavior]]] * batch_size
            inp.actions = torch.cat(
                [inp.actions, torch.tensor(beh_action, device=self.device)], dim=1
            )

        gen_kwargs: dict = dict(
            input_ids=inp.input_ids,
            attention_mask=inp.attention_mask,
            max_new_tokens=self.sole_item_len,
            prefix_allowed_tokens_fn=self.prefix_fn_by_behavior[behavior],
            num_beams=num_beams,
            num_return_sequences=num_beams,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True,
        )
        if backbone in ("Qwen3Session", "Qwen3Multi", "Qwen3SessionMulti", "LlamaMulti"):
            gen_kwargs["session_ids"] = inp.session_ids
            gen_kwargs["extended_session_ids"] = inp.extended_session_ids
        if backbone in ("Qwen3Multi", "Qwen3SessionMulti", "LlamaMulti"):
            gen_kwargs["actions"] = inp.actions
        if not is_decoder_only:
            gen_kwargs["decoder_input_ids"] = torch.tensor([[self.config.decoder_start_token_id] + tokens for tokens in beh_ids], device=self.device)

        output: "GenerateBeamOutput" = gen_model.generate(**gen_kwargs)
        out_ids = output.sequences
        scores = output.sequences_scores
        if is_decoder_only:
            out_ids = out_ids[:, -self.item_len:]
        output_str = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        return output_str, scores

    # ------------------------------------------------------------------
    # Full inference pass → per-uid result dict
    # ------------------------------------------------------------------

    def _collect_results(
        self,
        filtered_dataset: BaseSMBDataset,
        collator,
        target_behavior: str,
        num_beams: int,
        is_decoder_only: bool,
        backbone: str,
        batch_size: int,
        desc: str,
    ) -> dict[str, dict]:
        """
        Run beam-search over the whole filtered dataset and return a dict
            uid → {rank, score, top_preds, targets, topk_result}
        where `topk_result` is the raw list[int] from get_topk_results (length=num_beams).
        """
        loader = DataLoader(
            filtered_dataset,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=False,
            num_workers=0,
        )
        uid_results: dict[str, dict] = {}

        for batch in get_tqdm(loader, desc=desc):
            inputs: "BatchEncoding" = batch[0].to(self.device)
            targets: list[list[str]] = batch[1]
            uids: list[str] = inputs.get("uid", [None] * len(targets))

            output_str, scores = self._run_inference(
                inputs, target_behavior, filtered_dataset, num_beams, is_decoder_only, backbone
            )

            # Rank extraction (1-indexed; num_beams+1 if not found)
            for i, (uid, tgt_list) in enumerate(zip(uids, targets)):
                if uid is None:
                    continue
                cands = [s.replace(" ", "") for s in output_str[i * num_beams: (i + 1) * num_beams]]
                cand_scores = scores[i * num_beams: (i + 1) * num_beams]
                tgt_set = {t.replace(" ", "") for t in tgt_list}

                best_rank, best_score = num_beams + 1, None
                for rank, (cand, sc) in enumerate(zip(cands, cand_scores), start=1):
                    if cand in tgt_set:
                        best_rank = rank
                        best_score = float(sc)
                        break

                uid_results[uid] = {
                    "rank": best_rank if best_rank <= num_beams else None,
                    "score": best_score,
                    "top_preds": cands[:5],
                    "targets": [t.replace(" ", "") for t in tgt_list],
                }

        return uid_results

    # ------------------------------------------------------------------
    # Bucket statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _bucket_label(count: int, thresholds: list[int]) -> str:
        for i, thr in enumerate(thresholds):
            if count < thr:
                lo = thresholds[i - 1] if i > 0 else 0
                return f"{lo}-{thr - 1}"
        lo = thresholds[-1]
        return f"{lo}+"

    def _compute_bucket_stats(
        self,
        uid_target_counts: dict[str, int],
        results_ours: dict[str, dict],
        results_base: dict[str, dict],
        thresholds: list[int],
        metric_list: list[str],
        num_beams: int,
    ) -> list[dict]:
        # Parse K values from metric names (e.g. hit@10 → k=10)
        def _parse_k(m: str) -> int:
            return int(m.split("@")[1])

        all_uids = [uid for uid in uid_target_counts if uid in results_ours and uid in results_base]

        # group uids by bucket
        bucket_labels = []
        lo = 0
        for thr in thresholds:
            bucket_labels.append(f"{lo}-{thr - 1}")
            lo = thr
        bucket_labels.append(f"{lo}+")

        def _bucket_idx(count):
            for i, thr in enumerate(thresholds):
                if count < thr:
                    return i
            return len(thresholds)

        buckets: list[list[str]] = [[] for _ in range(len(bucket_labels))]
        for uid in all_uids:
            buckets[_bucket_idx(uid_target_counts[uid])].append(uid)

        rows = []
        for label, uids_in_bucket in zip(bucket_labels, buckets):
            if not uids_in_bucket:
                rows.append({
                    "label": label,
                    "n_users": 0,
                    **{f"model_{m}": None for m in metric_list},
                    **{f"baseline_{m}": None for m in metric_list},
                    **{f"delta_{m}": None for m in metric_list},
                })
                continue

            row: dict = {"label": label, "n_users": len(uids_in_bucket)}
            for m in metric_list:
                k = _parse_k(m)
                metric_fn = _hr_at_k if m.startswith("hit") else _ndcg_at_k

                ours_vals = [metric_fn(results_ours[u]["rank"], k) for u in uids_in_bucket]
                base_vals = [metric_fn(results_base[u]["rank"], k) for u in uids_in_bucket]
                ours_avg = float(np.mean(ours_vals))
                base_avg = float(np.mean(base_vals))
                row[f"model_{m}"] = ours_avg
                row[f"baseline_{m}"] = base_avg
                row[f"delta_{m}"] = ours_avg - base_avg
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # Interesting-user selection
    # ------------------------------------------------------------------

    def _find_interesting_users(
        self,
        uid_target_counts: dict[str, int],
        results_ours: dict[str, dict],
        results_base: dict[str, dict],
        base_dataset: BaseSMBDataset,
        target_behavior: str,
        max_sparse_count: int,
        interesting_top_k: int,
        max_users: int,
    ) -> list[dict]:
        candidates = []
        for uid, tgt_count in uid_target_counts.items():
            if tgt_count > max_sparse_count:
                continue
            if uid not in results_ours or uid not in results_base:
                continue
            r_ours = results_ours[uid]["rank"]   # None means not in top-beam
            r_base = results_base[uid]["rank"]
            if r_ours is None or r_ours > interesting_top_k:
                continue
            if r_base is None or r_base <= interesting_top_k:
                continue
            # improvement = baseline_rank - our_rank  (higher = more interesting)
            improvement = (r_base if r_base is not None else 9999) - r_ours
            candidates.append((uid, improvement))

        # Sort by largest improvement first
        candidates.sort(key=lambda x: -x[1])

        records = []
        all_behaviors = base_dataset.behaviors
        for uid, _ in candidates[:max_users]:
            test_pos = base_dataset.test_pos[uid]
            hist_beh = base_dataset.history_behaviors[uid][:test_pos]
            aux_counts = {
                b: hist_beh.count(b) for b in all_behaviors if b != target_behavior
            }
            records.append({
                "uid": uid,
                "target_behavior_count_in_history": uid_target_counts[uid],
                "auxiliary_behavior_counts": aux_counts,
                "target_items": results_ours[uid]["targets"],
                "model_rank": results_ours[uid]["rank"],
                "model_score": results_ours[uid]["score"],
                "model_top5": results_ours[uid]["top_preds"],
                "baseline_rank": results_base[uid]["rank"],
                "baseline_score": results_base[uid]["score"],
                "baseline_top5": results_base[uid]["top_preds"],
            })
        return records

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

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
        baseline_ckpt_path: str,
        baseline_backbone: str | None,
        baseline_max_his_len: int,
        test_task: str,
        target_behavior: str | None,
        test_batch_size: int,
        num_beams: int,
        metrics: str,
        bucket_thresholds: str,
        max_sparse_count: int,
        max_interesting_users: int,
        interesting_top_k: int,
        results_file: str,
        *args,
        **kwargs,
    ):
        self.init(seed, False)

        if baseline_backbone is None:
            baseline_backbone = backbone
        metric_list = [m.strip() for m in metrics.split(",")]
        thresholds = [int(t.strip()) for t in bucket_thresholds.split(",")]

        # ---- Load dataset (once; shared between both models) ----
        base_dataset: BaseSMBDataset = load_SMB_test_dataset(
            dataset, data_path, max_his_len, index_file, test_task
        )
        baseline_base_dataset: BaseSMBDataset = load_SMB_test_dataset(
            dataset, data_path, baseline_max_his_len, index_file, test_task
        )
        base_dataset.get_all_items()
        baseline_base_dataset.get_all_items()
        eval_behavior = target_behavior if target_behavior else base_dataset.target_behavior
        all_behaviors = base_dataset.behaviors
        logger.info(
            f"Dataset behaviors: {all_behaviors} | "
            f"eval behavior: {eval_behavior} | "
            f"metrics: {metric_list} | "
            f"bucket thresholds: {thresholds}"
        )

        # Pre-compute history target-behavior count for every test user
        filtered_dataset = base_dataset.filter_by_behavior(eval_behavior)
        baseline_filtered_dataset = baseline_base_dataset.filter_by_behavior(eval_behavior)
        uid_target_counts: dict[str, int] = {}
        for sample in filtered_dataset.inter_data:
            uid = sample.get("uid")
            if uid is None:
                continue
            test_pos = base_dataset.test_pos[uid]
            uid_target_counts[uid] = base_dataset.history_behaviors[uid][:test_pos].count(eval_behavior)

        logger.info(
            f"Test users with {eval_behavior} in history: "
            f"{sum(1 for c in uid_target_counts.values() if c > 0)} / {len(uid_target_counts)}"
        )

        # ================================================================
        # Run our model
        # ================================================================
        logger.info(f"Loading our model: {backbone} from {ckpt_path}")
        ours_is_dec = self._is_decoder_only_backbone(backbone)
        self._load_model_and_tokenizer(backbone, ckpt_path)
        self._build_tries(base_dataset, ours_is_dec)
        ours_collator = (
            DecoderOnlyTestCollator(self.tokenizer)
            if ours_is_dec
            else EncoderDecoderTestCollator(self.tokenizer)
        )
        self.model.eval()
        with torch.no_grad():
            results_ours = self._collect_results(
                filtered_dataset, ours_collator, eval_behavior,
                num_beams, ours_is_dec, backbone, test_batch_size,
                desc=f"Our model [{backbone}] inference",
            )
        del self.model
        torch.cuda.empty_cache()
        logger.info(f"Our model: {len(results_ours)} users processed.")

        # ================================================================
        # Run baseline model
        # ================================================================
        logger.info(f"Loading baseline model: {baseline_backbone} from {baseline_ckpt_path}")
        base_is_dec = self._is_decoder_only_backbone(baseline_backbone)
        self._load_model_and_tokenizer(baseline_backbone, baseline_ckpt_path)
        self._build_tries(baseline_base_dataset, base_is_dec)
        base_collator = (
            DecoderOnlyTestCollator(self.tokenizer)
            if base_is_dec
            else EncoderDecoderTestCollator(self.tokenizer)
        )
        self.model.eval()
        with torch.no_grad():
            results_base = self._collect_results(
                baseline_filtered_dataset, base_collator, eval_behavior,
                num_beams, base_is_dec, baseline_backbone, test_batch_size,
                desc=f"Baseline [{baseline_backbone}] inference",
            )
        del self.model
        torch.cuda.empty_cache()
        logger.info(f"Baseline: {len(results_base)} users processed.")

        # ================================================================
        # Bucketed statistics
        # ================================================================
        bucket_rows = self._compute_bucket_stats(
            uid_target_counts, results_ours, results_base,
            thresholds, metric_list, num_beams,
        )

        # ================================================================
        # Interesting sparse users
        # ================================================================
        interesting_users = self._find_interesting_users(
            uid_target_counts, results_ours, results_base,
            base_dataset, eval_behavior,
            max_sparse_count, interesting_top_k, max_interesting_users,
        )
        logger.info(f"Interesting sparse users found: {len(interesting_users)}")

        # ================================================================
        # Save & print
        # ================================================================
        output = {
            "model": {"backbone": backbone, "ckpt_path": ckpt_path},
            "baseline": {"backbone": baseline_backbone, "ckpt_path": baseline_ckpt_path},
            "dataset": dataset,
            "test_task": test_task,
            "target_behavior": eval_behavior,
            "metrics": metric_list,
            "bucket_thresholds": thresholds,
            "num_beams": num_beams,
            "bucket_statistics": bucket_rows,
            "interesting_sparse_users": interesting_users,
        }

        ensure_dir(os.path.dirname(results_file))
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        self._print_summary(bucket_rows, interesting_users, metric_list, eval_behavior, dataset)
        logger.success(f"Results saved to {results_file}.")
        self.finish(False)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    @staticmethod
    def _print_summary(
        bucket_rows: list[dict],
        interesting_users: list[dict],
        metric_list: list[str],
        target_behavior: str,
        dataset_name: str,
    ):
        logger.success("=" * 72)
        logger.success(f"Sparse-Target-Behavior Analysis — {dataset_name} ({target_behavior})")
        logger.success("")

        # Bucketed statistics table
        header = f"{'Bucket':<12} {'N':>5}"
        for m in metric_list:
            header += f"  {'Ours_' + m:>14}  {'Base_' + m:>14}  {'Δ' + m:>8}"
        logger.success(header)
        logger.success("-" * 72)
        for row in bucket_rows:
            line = f"{row['label']:<12} {row['n_users']:>5}"
            for m in metric_list:
                o = row.get(f"model_{m}")
                b = row.get(f"baseline_{m}")
                d = row.get(f"delta_{m}")
                o_str = f"{o:.4f}" if o is not None else "  N/A "
                b_str = f"{b:.4f}" if b is not None else "  N/A "
                d_str = f"{d:+.4f}" if d is not None else "  N/A "
                line += f"  {o_str:>14}  {b_str:>14}  {d_str:>8}"
            logger.success(line)

        # Interesting users summary
        logger.success("")
        logger.success(
            f"Interesting sparse users (our model in top-K, baseline outside top-K): "
            f"{len(interesting_users)}"
        )
        for u in interesting_users[:5]:
            logger.success(
                f"  uid={u['uid']:<12}  hist_conv={u['target_behavior_count_in_history']}  "
                f"aux={u['auxiliary_behavior_counts']}  "
                f"our_rank={u['model_rank']}  base_rank={u['baseline_rank']}"
            )
        logger.success("=" * 72)
