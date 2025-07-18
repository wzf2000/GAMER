import torch
from transformers import BatchEncoding
from transformers.tokenization_utils import PreTrainedTokenizer

from SeqRec.datasets.MB_dataset import EvaluationType


class Collator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch: list[dict]) -> BatchEncoding:
        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]

        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        labels = self.tokenizer(
            label_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        inputs["labels"] = labels["input_ids"]
        inputs["labels"][inputs["labels"] == self.tokenizer.pad_token_id] = -100
        if "behavior" in batch[0]:
            # If the batch contains target behavior, add it to the inputs
            inputs["target_behavior"] = [d["behavior"] for d in batch]

        return inputs


class TestCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch: list[dict]) -> tuple[BatchEncoding, list[str], torch.LongTensor]:
        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]
        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        label_ids = self.tokenizer(
            text=targets,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )['input_ids']
        if "behavior" in batch[0]:
            # If the batch contains target behavior, add it to the inputs
            inputs["target_behavior"] = [d["behavior"] for d in batch]

        return (inputs, targets, label_ids)
