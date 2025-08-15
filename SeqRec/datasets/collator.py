import copy
import torch
from transformers import BatchEncoding
from transformers.tokenization_utils import PreTrainedTokenizer


class EncoderDecoderCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch: list[dict]) -> BatchEncoding:
        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            text_target=label_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        inputs["labels"][inputs["labels"] == self.tokenizer.pad_token_id] = -100
        if "behavior" in batch[0]:
            # If the batch contains target behavior, add it to the inputs
            inputs["behavior"] = [d["behavior"] for d in batch]
        if "session_ids" in batch[0]:
            # If the batch contains session IDs, add it to the inputs
            session_ids = [d["session_ids"] for d in batch]
            max_length = max([len(sub) for sub in session_ids])
            session_ids = [session + [0] * (max_length - len(session)) for session in session_ids]
            inputs["session_ids"] = torch.tensor(session_ids, dtype=torch.long)
        if "time" in batch[0]:
            time = [d["time"] for d in batch]
            max_length = max([len(sub) for sub in time])
            time = [t + [-1] * (max_length - len(t)) for t in time]
            inputs["time"] = torch.tensor(time, dtype=torch.float32)

        return inputs


class DecoderOnlyCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, only_train_response: bool = False):
        self.only_train_response = only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

    def __call__(self, batch: list[dict]) -> BatchEncoding:
        input_texts = [d["input_ids"] for d in batch]
        full_texts = [d["input_ids"] + d["labels"] + self.tokenizer.eos_token for d in batch]

        inputs = self.tokenizer(
            text=full_texts,
            text_target=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        labels = copy.deepcopy(inputs["input_ids"])
        # ignore padding
        labels[labels == self.tokenizer.pad_token_id] = -100
        if self.only_train_response or ('split' in batch[0] and batch[0]['split'] == 'valid'):
            # ignore input text
            labels[torch.where(inputs["labels"] != self.tokenizer.pad_token_id)] = -100

        inputs["labels"] = labels
        if "behavior" in batch[0]:
            # If the batch contains target behavior, add it to the inputs
            inputs["target_behavior"] = [d["behavior"] for d in batch]
        if "session_ids" in batch[0]:
            # If the batch contains session IDs, add it to the inputs
            inputs["session_ids"] = torch.tensor([d["session_ids"] for d in batch], dtype=torch.long)

        return inputs


class EncoderDecoderTestCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch: list[dict]) -> tuple[BatchEncoding, list[str] | list[list[str]]]:
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

        if "behavior" in batch[0]:
            # If the batch contains target behavior, add it to the inputs
            inputs["behavior"] = [d["behavior"] for d in batch]
        if "session_ids" in batch[0]:
            # If the batch contains session IDs, add it to the inputs
            session_ids = [d["session_ids"] for d in batch]
            max_length = max([len(sub) for sub in session_ids])
            session_ids = [session + [0] * (max_length - len(session)) for session in session_ids]
            inputs["session_ids"] = torch.tensor(session_ids, dtype=torch.long)
        if "time" in batch[0]:
            time = [d["time"] for d in batch]
            max_length = max([len(sub) for sub in time])
            time = [t + [-1] * (max_length - len(t)) for t in time]
            inputs["time"] = torch.tensor(time, dtype=torch.float32)

        return (inputs, targets)


class DecoderOnlyTestCollator(object):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        # Allow batched inference
        self.tokenizer.padding_side = "left"

    def __call__(self, batch: list[dict]) -> tuple[BatchEncoding, list[str] | list[list[str]]]:
        targets = [d["labels"] for d in batch]
        if isinstance(batch[0]["labels"], str):
            full_texts = [d["input_ids"] + d["labels"] for d in batch]
        else:
            assert isinstance(batch[0]["labels"], list), "labels should be a string or a list of strings"
            full_texts = [d["input_ids"] for d in batch]
        inputs = self.tokenizer(
            text=full_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        if "behavior" in batch[0]:
            # If the batch contains target behavior, add it to the inputs
            inputs["target_behavior"] = [d["behavior"] for d in batch]
        if "session_ids" in batch[0]:
            # If the batch contains session IDs, add it to the inputs
            inputs["session_ids"] = [d["session_ids"] for d in batch]

        return (inputs, targets)
