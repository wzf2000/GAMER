from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

try:
    from SeqRec.datasets.session_behavior.ranking import SMBRankingDatasetForDecoder
except ModuleNotFoundError as exc:
    SMBRankingDatasetForDecoder = None
    DATASET_IMPORT_ERROR = exc
else:
    DATASET_IMPORT_ERROR = None

try:
    import torch
    from transformers import BatchEncoding

    from SeqRec.datasets.collators.generative import DecoderOnlyRankingCollator
    from SeqRec.evaluation.ranking import binary_auc, get_metrics_results, get_ranked_item_hits, rank_items_by_scores
    from SeqRec.models.generative.qwen3.temporal_hierarchical import resolve_relation_action_indices
except ModuleNotFoundError as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.unk_token_id = 0
        self.model_max_length = 128
        self.padding_side = "right"
        self.truncation_side = "right"
        self._vocab = {"<pad>": 0}

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"<[^>]+>", text)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for token in self._tokens(text):
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
            ids.append(self._vocab[token])
        return ids

    def _pad(self, rows: list[list[int]], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        padded = []
        masks = []
        for row in rows:
            if len(row) > max_length:
                row = row[-max_length:] if self.truncation_side == "left" else row[:max_length]
            padding = [self.pad_token_id] * (max_length - len(row))
            padded.append(row + padding)
            masks.append([1] * len(row) + [0] * len(padding))
        return torch.tensor(padded, dtype=torch.long), torch.tensor(masks, dtype=torch.long)

    def __call__(
        self,
        text,
        text_target=None,
        return_tensors=None,
        padding=None,
        max_length=None,
        truncation=False,
        return_attention_mask=True,
    ):
        texts = text if isinstance(text, list) else [text]
        rows = [self.encode(one, add_special_tokens=False) for one in texts]
        input_max = min(max(len(row) for row in rows), max_length or self.model_max_length)
        input_ids, attention_mask = self._pad(rows, input_max)
        data = {"input_ids": input_ids, "attention_mask": attention_mask}
        if text_target is not None:
            target_texts = text_target if isinstance(text_target, list) else [text_target]
            target_rows = [self.encode(one, add_special_tokens=False) for one in target_texts]
            target_max = min(max(len(row) for row in target_rows), max_length or self.model_max_length)
            data["labels"], _target_mask = self._pad(target_rows, target_max)
        return BatchEncoding(data)


@unittest.skipIf(DATASET_IMPORT_ERROR is not None, f"dataset dependencies unavailable: {DATASET_IMPORT_ERROR}")
class SMBRankingDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmp.name)
        dataset_dir = self.data_root / "Tiny"
        dataset_dir.mkdir()
        prefix = dataset_dir / "Tiny"

        item_ids = list(range(1, 10))
        index = {
            str(item_id): [f"<i{item_id}a>", f"<i{item_id}b>"]
            for item_id in item_ids
        }
        files = {
            f"{prefix}.SMB.inter.json": {
                "u1": [1, 2, 3, 4, 5, 6, 7],
                "u2": [8, 9],
            },
            f"{prefix}.SMB.behavior.json": {
                "u1": ["view", "click", "cart", "buy", "view", "buy", "click"],
                "u2": ["view", "buy"],
            },
            f"{prefix}.SMB.session.json": {
                "u1": [10, 10, 20, 30, 30, 40, 40],
                "u2": [1, 2],
            },
            f"{prefix}.SMB.time.json": {
                "u1": [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:30:00",
                    "2026-01-02 00:00:00",
                    "2026-01-03 00:00:00",
                    "2026-01-03 00:30:00",
                    "2026-01-04 00:00:00",
                    "2026-01-04 00:30:00",
                ],
                "u2": ["2026-01-01 00:00:00", "2026-01-02 00:00:00"],
            },
            f"{prefix}.behavior_level.json": {
                "view": 0,
                "click": 1,
                "cart": 2,
                "buy": 3,
            },
            f"{prefix}.index.json": index,
        }
        for path, content in files.items():
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(content, handle)

    def tearDown(self):
        self.tmp.cleanup()

    def _dataset(self, mode: str):
        return SMBRankingDatasetForDecoder(
            dataset="Tiny",
            data_path=str(self.data_root),
            max_his_len=20,
            index_file=".index.json",
            mode=mode,
            train_session=True,
        )

    def test_session_split_uses_minus_three_minus_two_minus_one(self):
        train = self._dataset("train")
        valid = self._dataset("valid")
        test = self._dataset("test")

        self.assertEqual(len(train), 1)
        self.assertEqual(train[0]["target_item"], "<i3a><i3b>")
        self.assertEqual(train[0]["labels"], "<behavior_cart>")
        self.assertEqual(train[0]["target_session_id"], 10)

        self.assertEqual([sample["target_item"] for sample in valid], ["<i4a><i4b>", "<i5a><i5b>"])
        self.assertEqual([sample["target_session_id"] for sample in valid], [20, 20])

        self.assertEqual(len(test), 1)
        self.assertEqual(test[0]["labels"], ["<i6a><i6b>", "<i7a><i7b>"])
        self.assertEqual(test[0]["target_session_id"], 30)

    def test_candidate_input_has_raw_item_and_relation_unknown(self):
        sample = self._dataset("train")[0]

        self.assertTrue(sample["input_ids"].endswith("<i3a><i3b>"))
        self.assertNotIn("<i3a><i3b><behavior_cart>", sample["input_ids"])
        self.assertNotIn("<behavior_cart><i3a><i3b>", sample["input_ids"])
        self.assertEqual(sample["ranking_labels"], 0.0)
        self.assertEqual(sample["relation_actions"], [1, 1, 1, 2, 2, 2, 0, 0])
        self.assertEqual(sample["actions"], sample["relation_actions"])

    def test_collator_appends_behavior_label_for_shared_lm_head(self):
        if TORCH_IMPORT_ERROR is not None:
            self.skipTest(f"torch-dependent collator test skipped: {TORCH_IMPORT_ERROR}")
        sample = self._dataset("train")[0]
        tokenizer = FakeTokenizer()
        batch = DecoderOnlyRankingCollator(tokenizer)([sample])
        input_len = len(tokenizer.encode(sample["input_ids"], add_special_tokens=False))
        full_len = input_len + len(tokenizer.encode(sample["labels"], add_special_tokens=False))

        self.assertEqual(batch["input_ids"].shape[1], full_len)
        self.assertIn("labels", batch)
        self.assertNotIn("ranking_labels", batch)
        self.assertTrue(torch.all(batch["labels"][0, :input_len] == -100))
        self.assertNotEqual(batch["labels"][0, input_len].item(), -100)
        self.assertEqual(batch["relation_actions"][0, input_len - 1].item(), 0)
        self.assertEqual(batch["relation_actions"][0, input_len].item(), 0)

    def test_relation_override_helper_preserves_old_path_when_absent(self):
        if TORCH_IMPORT_ERROR is not None:
            self.skipTest(f"torch-dependent model helper test skipped: {TORCH_IMPORT_ERROR}")
        router_actions = torch.tensor([[1, 2, 3]])
        self.assertIs(resolve_relation_action_indices(router_actions, None), router_actions)
        override = torch.tensor([[0, 0, 0]])
        self.assertTrue(torch.equal(resolve_relation_action_indices(router_actions, override), override))
        cached = resolve_relation_action_indices(torch.tensor([[0]]), torch.tensor([[4, 5, 6]]), torch.tensor([1]))
        self.assertEqual(cached.item(), 5)

    def test_ranking_helpers_sort_and_score_multi_target(self):
        if TORCH_IMPORT_ERROR is not None:
            self.skipTest(f"torch-dependent ranking helper test skipped: {TORCH_IMPORT_ERROR}")
        ranked = rank_items_by_scores(["a", "b", "c"], torch.tensor([0.2, 0.9, 0.5]))
        topk_hits = [get_ranked_item_hits(ranked, ["a", "c"], 3)]
        metrics = get_metrics_results(topk_hits, ["hit@1", "recall@2"], [["a", "c"]], list_output=False)

        self.assertEqual(ranked, ["b", "c", "a"])
        self.assertEqual(topk_hits, [[0, 1, 1]])
        self.assertEqual(metrics["hit@1"], 0.0)
        self.assertEqual(metrics["recall@2"], 0.5)
        self.assertEqual(binary_auc([1, 0, 1, 0], [0.9, 0.8, 0.4, 0.4]), 0.625)


if __name__ == "__main__":
    unittest.main()
