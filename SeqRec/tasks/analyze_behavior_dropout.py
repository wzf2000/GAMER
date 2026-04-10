"""
Behavior-dropout ablation analysis task.

Selects HR@1 "typical users" from the test set (all behavior types present in
history), then re-runs inference with every non-empty subset of behavior types
retained in the history.  Records the rank and beam score of each target item
under each setting so we can see which behaviors drive the prediction.
"""

import os
import copy
import json
import itertools
import torch
import numpy as np
from loguru import logger
from typing import Callable, TYPE_CHECKING
from torch.utils.data import DataLoader, Dataset

from SeqRec.tasks.multi_gpu import MultiGPUTask
from SeqRec.datasets.loading_SMB import load_SMB_test_dataset
from SeqRec.datasets.SMB_dataset import BaseSMBDataset
from SeqRec.datasets.collator import DecoderOnlyTestCollator, EncoderDecoderTestCollator
from SeqRec.generation.trie import Trie, prefix_allowed_tokens_fn, prefix_allowed_tokens_fn_by_last_token
from SeqRec.utils.futils import ensure_dir
from SeqRec.utils.parse import SubParsersAction, parse_global_args, parse_dataset_args
from SeqRec.utils.pipe import get_tqdm

if TYPE_CHECKING:
    from transformers.generation.utils import GenerateBeamOutput
    from transformers import BatchEncoding


# ---------------------------------------------------------------------------
# Tiny helper dataset for single-user batch inference
# ---------------------------------------------------------------------------

class _SampleDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Main task
# ---------------------------------------------------------------------------

class AnalyzeBehaviorDropout(MultiGPUTask):
    """
    Post-hoc analysis: for each HR@1 test user (with all behavior types in
    history), sweep over every non-empty kept-behavior subset, run beam-search
    inference, and record the rank / score of the target item(s).
    """

    @staticmethod
    def parser_name() -> str:
        return "analyze_behavior_dropout"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        parser = sub_parsers.add_parser(
            "analyze_behavior_dropout",
            help="Behavior-dropout ablation study on test users that hit HR@1.",
        )
        parser = parse_global_args(parser)
        parser = parse_dataset_args(parser)
        parser.add_argument("--ckpt_path", type=str, default="./checkpoint")
        parser.add_argument(
            "--results_file", type=str, default="./results/behavior_dropout.json"
        )
        parser.add_argument("--test_task", type=str, default="smb_explicit")
        parser.add_argument("--test_batch_size", type=int, default=16)
        parser.add_argument("--num_beams", type=int, default=20)
        parser.add_argument(
            "--max_users",
            type=int,
            default=20,
            help="Maximum number of typical users to analyse.",
        )
        parser.add_argument(
            "--target_behavior",
            type=str,
            default=None,
            help="Behavior used for HR@1 selection (default: dataset max-level behavior).",
        )

    # -----------------------------------------------------------------------
    # Model / tokenizer loading  (mirrors test_SMB_decoder)
    # -----------------------------------------------------------------------

    def _load_model_and_tokenizer(self, backbone: str, ckpt_path: str):
        if backbone == "TIGER":
            from transformers import T5Tokenizer
            from SeqRec.models.generative.TIGER import TIGER
            self.tokenizer = T5Tokenizer.from_pretrained(ckpt_path, legacy=True)
            self.model = TIGER.from_pretrained(ckpt_path).to(self.device)
        elif backbone == "PBATransformer":
            from transformers import T5Tokenizer
            from SeqRec.models.generative.PBATransformer import (
                PBATransformerForConditionalGeneration,
            )
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
            from SeqRec.models.generative.Qwen3SessionMulti import (
                Qwen3SessionMultiWithTemperature,
            )
            self.tokenizer = Qwen2Tokenizer.from_pretrained(ckpt_path)
            self.model = Qwen3SessionMultiWithTemperature.from_pretrained(ckpt_path).to(self.device)
        elif backbone == "LlamaMulti":
            from transformers import Qwen2Tokenizer
            from SeqRec.models.generative.LlamaMulti import LlamaMultiWithTemperature
            self.tokenizer = Qwen2Tokenizer.from_pretrained(ckpt_path)
            self.model = LlamaMultiWithTemperature.from_pretrained(ckpt_path).to(self.device)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if (
            hasattr(self.model, "config")
            and self.model.config.pad_token_id is None
        ):
            self.model.config.pad_token_id = self.tokenizer.encode(
                self.tokenizer.pad_token, add_special_tokens=False
            )[0]
        self.config = self.model.config

    # -----------------------------------------------------------------------
    # Constrained-decoding trie construction
    # -----------------------------------------------------------------------

    def _build_tries(self, base_dataset: BaseSMBDataset):
        """Pre-build per-behavior candidate tries for prefix-constrained decoding."""
        all_behavior_items = base_dataset.get_all_items("all")
        item_reps = list(all_behavior_items)
        items_tokens = self.tokenizer.batch_encode_plus(
            item_reps, add_special_tokens=False
        )["input_ids"]
        self.item_len = len(items_tokens[0])
        self.sole_item_len = len(
            self.tokenizer.encode(
                next(iter(base_dataset.get_all_items())), add_special_tokens=False
            )
        )
        last_token_set = {tokens[-1] for tokens in items_tokens}
        last_token_set.add(self.config.pad_token_id)

        self.prefix_fn_by_behavior: dict[str, Callable] = {}
        for beh in base_dataset.behaviors:
            beh_items = list(base_dataset.get_all_items(beh))
            if self._is_decoder_only:
                cand_tokens = self.tokenizer.batch_encode_plus(
                    beh_items, add_special_tokens=False
                )["input_ids"]
                trie = Trie(cand_tokens)
                self.prefix_fn_by_behavior[beh] = prefix_allowed_tokens_fn_by_last_token(
                    trie, last_token_set
                )
            else:
                cand_tokens = self.tokenizer.batch_encode_plus(beh_items)["input_ids"]
                cand_tokens = [
                    [self.config.decoder_start_token_id] + t for t in cand_tokens
                ]
                trie = Trie(cand_tokens)
                self.prefix_fn_by_behavior[beh] = prefix_allowed_tokens_fn(trie)

    # -----------------------------------------------------------------------
    # Single-batch beam-search inference
    # -----------------------------------------------------------------------

    def _run_inference(
        self,
        inputs: "BatchEncoding",
        behavior: str,
        dataset: BaseSMBDataset,
        num_beams: int,
    ) -> tuple[list[str], torch.Tensor, int]:
        """
        Append the behavior token to *a copy of* inputs, run beam search,
        and return (output_str, scores, behavior_token_num).

        output_str[i*num_beams + j] is the j-th candidate (full decoded string,
        including behavior token) for sample i.
        """
        from transformers.generation import GenerationMixin

        # --- encode behavior token for each sample ---
        batch_size = inputs.input_ids.shape[0]
        beh_str = "".join(dataset.get_behavior_tokens(behavior))
        beh_enc = self.tokenizer.batch_encode_plus(
            [beh_str] * batch_size, add_special_tokens=False
        )
        beh_token_ids = beh_enc["input_ids"]           # list of lists
        beh_attn_mask = beh_enc["attention_mask"]
        beh_token_num = len(beh_token_ids[0])

        # --- work on a copy so the original tensor is not mutated ---
        inp = copy.copy(inputs)

        gen_model = (
            self.model if isinstance(self.model, GenerationMixin) else self.model.module
        )

        if self._is_decoder_only:
            inp.input_ids = torch.cat(
                [inp.input_ids, torch.tensor(beh_token_ids, device=self.device)], dim=1
            )
            inp.attention_mask = torch.cat(
                [inp.attention_mask, torch.tensor(beh_attn_mask, device=self.device)],
                dim=1,
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
        if self.backbone in ("Qwen3Session", "Qwen3Multi", "Qwen3SessionMulti", "LlamaMulti"):
            gen_kwargs["session_ids"] = inp.session_ids
            gen_kwargs["extended_session_ids"] = inp.extended_session_ids
        if self.backbone in ("Qwen3Multi", "Qwen3SessionMulti", "LlamaMulti"):
            gen_kwargs["actions"] = inp.actions

        output: "GenerateBeamOutput" = gen_model.generate(**gen_kwargs)
        output_ids = output.sequences
        scores = output.sequences_scores

        if self._is_decoder_only:
            output_ids = output_ids[:, -self.item_len:]

        output_str = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return output_str, scores, beh_token_num

    # -----------------------------------------------------------------------
    # Rank extraction
    # -----------------------------------------------------------------------

    @staticmethod
    def _compute_ranks(
        output_str: list[str],
        scores: torch.Tensor,
        targets: list[list[str]],
        num_beams: int,
    ) -> list[tuple[int, float | None]]:
        """
        For each sample find the best (lowest) rank of its target items among the
        beam candidates and the corresponding beam score.
        Returns list of (rank, score); rank is 1-indexed, num_beams+1 if not found.
        """
        results = []
        for i, target_list in enumerate(targets):
            cands = [
                s.replace(" ", "")
                for s in output_str[i * num_beams: (i + 1) * num_beams]
            ]
            cand_scores = scores[i * num_beams: (i + 1) * num_beams]
            target_set = {t.replace(" ", "") for t in target_list}
            best_rank, best_score = num_beams + 1, None
            for rank, (cand, sc) in enumerate(zip(cands, cand_scores), start=1):
                if cand in target_set:
                    best_rank = rank
                    best_score = float(sc)
                    break  # candidates are sorted by score; first hit == best rank
            results.append((best_rank, best_score))
        return results

    # -----------------------------------------------------------------------
    # Build a test-style sample dict with behavior dropout
    # -----------------------------------------------------------------------

    def _build_sample(
        self,
        dataset: BaseSMBDataset,
        uid: str,
        kept_behaviors: set[str],
    ) -> dict | None:
        """
        Build a sample dict (same schema as BaseSMBDataset.__getitem__) for the
        given user with only `kept_behaviors` retained in the history.
        Returns None if the filtered history would be empty (collator cannot
        handle all-empty session_ids / actions tensors).
        """
        test_pos = dataset.test_pos[uid]
        items = dataset.remapped_inters[uid]
        behaviors = dataset.history_behaviors[uid]
        sids = dataset.session[uid]
        times = dataset.time[uid]

        # Filter history to kept behaviors
        hist = [
            (it, beh, sid, t)
            for it, beh, sid, t in zip(
                items[:test_pos], behaviors[:test_pos],
                sids[:test_pos], times[:test_pos],
            )
            if beh in kept_behaviors
        ]
        if not hist:
            return None  # empty history → skip

        hist_items, hist_behaviors, hist_sids, hist_times = map(list, zip(*hist))

        # Target session items (unchanged, only history is modified)
        session_items = [
            dataset.get_behavior_item(items[i], behaviors[i])
            for i in range(test_pos, len(items))
        ]
        session_behaviors = list(behaviors[test_pos:])

        return {
            "input_ids": dataset._get_inters(hist_items, hist_behaviors),
            "labels": session_items,
            "behavior": session_behaviors,
            "session_ids": dataset._generate_session_ids(hist_sids),
            "extended_session_ids": dataset._generate_extended_session_ids(hist_sids),
            "actions": dataset._generate_actions(hist_behaviors),
            "time": dataset._generate_times(hist_times + [times[test_pos]]),
            "inters_item_list": dataset._get_inters_with_only_items(hist_items),
            "split": "test",
            "uid": uid,
        }

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def invoke(
        self,
        # global args
        seed: int,
        backbone: str,
        base_model: str,
        output_dir: str,
        # dataset args
        data_path: str,
        tasks: str,
        dataset: str,
        index_file: str,
        max_his_len: int,
        # task-specific args
        ckpt_path: str,
        results_file: str,
        test_task: str,
        test_batch_size: int,
        num_beams: int,
        max_users: int,
        target_behavior: str | None,
        *args,
        **kwargs,
    ):
        self.init(seed, False)
        self._is_decoder_only = backbone in (
            "Qwen3", "Qwen3Session", "Qwen3Multi", "Qwen3SessionMulti", "LlamaMulti"
        )
        self.backbone = backbone

        self._load_model_and_tokenizer(backbone, ckpt_path)

        # ---- Load test dataset ----
        base_dataset: BaseSMBDataset = load_SMB_test_dataset(
            dataset, data_path, max_his_len, index_file, test_task
        )
        all_behaviors = base_dataset.behaviors
        eval_behavior = target_behavior if target_behavior else base_dataset.target_behavior
        logger.info(
            f"Dataset behaviors: {all_behaviors} | target_behavior for HR@1: {eval_behavior}"
        )

        base_dataset.get_all_items()
        self._build_tries(base_dataset)

        collator = (
            DecoderOnlyTestCollator(self.tokenizer)
            if self._is_decoder_only
            else EncoderDecoderTestCollator(self.tokenizer)
        )

        # ---- Step 1: Full-sequence inference on target-behavior test split ----
        filtered_dataset = base_dataset.filter_by_behavior(eval_behavior)
        loader = DataLoader(
            filtered_dataset,
            batch_size=test_batch_size,
            collate_fn=collator,
            shuffle=False,
            num_workers=0,
        )

        self.model.eval()
        hr1_uids: list[str] = []

        with torch.no_grad():
            for batch in get_tqdm(loader, desc="Full-sequence inference (HR@1 selection)"):
                inputs: "BatchEncoding" = batch[0].to(self.device)
                targets: list[list[str]] = batch[1]
                uids: list[str] = inputs.get("uid", [None] * len(targets))

                output_str, scores, _ = self._run_inference(
                    inputs, eval_behavior, filtered_dataset, num_beams
                )
                ranks = self._compute_ranks(output_str, scores, targets, num_beams)

                for uid, (rank, _) in zip(uids, ranks):
                    if uid is not None and rank == 1:
                        hr1_uids.append(uid)

        logger.info(f"HR@1 users found: {len(hr1_uids)}")

        # ---- Step 2: Keep users where ALL behavior types appear in history ----
        typical_uids: list[str] = []
        all_behaviors_set = set(all_behaviors)
        for uid in hr1_uids:
            test_pos = base_dataset.test_pos[uid]
            hist_beh_set = set(base_dataset.history_behaviors[uid][:test_pos])
            if all_behaviors_set == hist_beh_set:
                typical_uids.append(uid)
            if len(typical_uids) >= max_users:
                break

        if not typical_uids:
            logger.warning(
                "No user has all behavior types in history; "
                "relaxing to all HR@1 users."
            )
            typical_uids = hr1_uids[:max_users]

        logger.info(f"Typical users selected: {len(typical_uids)}")

        # ---- Step 3: Enumerate all non-empty kept-behavior subsets ----
        # Sort behaviors by level (ascending) for consistent subset naming
        behaviors_by_level = sorted(all_behaviors, key=lambda b: base_dataset.behavior_level[b])
        dropout_settings: list[tuple[str, set[str]]] = []
        for r in range(1, len(all_behaviors) + 1):
            for kept in itertools.combinations(behaviors_by_level, r):
                name = "keep_" + "_".join(kept)
                dropout_settings.append((name, set(kept)))

        logger.info(
            f"Dropout settings ({len(dropout_settings)}): "
            + ", ".join(s[0] for s in dropout_settings)
        )

        # ---- Step 4: Per-user dropout ablation ----
        user_records: list[dict] = []

        with torch.no_grad():
            for uid in get_tqdm(typical_uids, desc="Behavior-dropout ablation"):
                test_pos = base_dataset.test_pos[uid]
                hist_beh_counts = {
                    beh: base_dataset.history_behaviors[uid][:test_pos].count(beh)
                    for beh in all_behaviors
                }

                user_record: dict = {
                    "uid": uid,
                    "history_behavior_counts": hist_beh_counts,
                    "target_behavior": eval_behavior,
                    "settings": [],
                }

                for setting_name, kept_behaviors in dropout_settings:
                    sample = self._build_sample(base_dataset, uid, kept_behaviors)

                    if sample is None:
                        # Empty history after filtering — skip gracefully
                        user_record["settings"].append({
                            "name": setting_name,
                            "kept_behaviors": sorted(
                                kept_behaviors,
                                key=lambda b: base_dataset.behavior_level[b],
                            ),
                            "best_rank": None,
                            "best_score": None,
                            "top5_predictions": [],
                            "target_items": [],
                            "note": "empty_history",
                        })
                        continue

                    sample_ds = _SampleDataset([sample])
                    sample_loader = DataLoader(
                        sample_ds, batch_size=1, collate_fn=collator, num_workers=0
                    )
                    inp_batch, tgt_batch = next(iter(sample_loader))
                    inp_batch = inp_batch.to(self.device)

                    output_str, scores, _ = self._run_inference(
                        inp_batch, eval_behavior, base_dataset, num_beams
                    )
                    (rank, score) = self._compute_ranks(
                        output_str, scores, tgt_batch, num_beams
                    )[0]

                    top5 = [s.replace(" ", "") for s in output_str[:5]]
                    tgt_strs = [t.replace(" ", "") for t in tgt_batch[0]]

                    user_record["settings"].append({
                        "name": setting_name,
                        "kept_behaviors": sorted(
                            kept_behaviors,
                            key=lambda b: base_dataset.behavior_level[b],
                        ),
                        "best_rank": rank if rank <= num_beams else None,
                        "best_score": score,
                        "top5_predictions": top5,
                        "target_items": tgt_strs,
                    })

                user_records.append(user_record)

                # Quick per-user summary in log
                full_setting = f"keep_{'_'.join(behaviors_by_level)}"
                full_res = next(
                    (s for s in user_record["settings"] if s["name"] == full_setting),
                    None,
                )
                full_rank = full_res["best_rank"] if full_res else "?"
                logger.info(f"  uid={uid}  full_rank={full_rank}  hist={hist_beh_counts}")

        # ---- Step 5: Aggregate & save ----
        results = {
            "backbone": backbone,
            "dataset": dataset,
            "test_task": test_task,
            "behaviors": all_behaviors,
            "behavior_levels": base_dataset.behavior_level,
            "target_behavior": eval_behavior,
            "num_typical_users": len(typical_uids),
            "dropout_settings": [s[0] for s in dropout_settings],
            "users": user_records,
        }

        # Per-setting average ranks
        summary: list[dict] = []
        for sname, _ in dropout_settings:
            ranks = [
                s["best_rank"]
                for u in user_records
                for s in u["settings"]
                if s["name"] == sname and s["best_rank"] is not None
            ]
            hr1 = sum(r == 1 for r in ranks) / len(ranks) if ranks else 0.0
            summary.append({
                "setting": sname,
                "n_valid": len(ranks),
                "avg_rank": float(np.mean(ranks)) if ranks else None,
                "hr@1": hr1,
            })
        results["summary"] = summary

        ensure_dir(os.path.dirname(results_file))
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self._print_summary(summary, dataset)
        logger.success(f"Full results saved to {results_file}.")
        self.finish(False)

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------

    @staticmethod
    def _print_summary(summary: list[dict], dataset_name: str):
        logger.success("=" * 65)
        logger.success(f"Behavior Dropout Analysis — {dataset_name}")
        logger.success(f"{'Setting':<40} {'N':>4}  {'AvgRank':>8}  {'HR@1':>6}")
        logger.success("-" * 65)
        for row in summary:
            avg = f"{row['avg_rank']:.2f}" if row["avg_rank"] is not None else "  N/A"
            logger.success(
                f"{row['setting']:<40} {row['n_valid']:>4}  {avg:>8}  {row['hr@1']:>6.3f}"
            )
        logger.success("=" * 65)
