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
    def __init__(self, tokenizer: PreTrainedTokenizer, only_train_response: bool = False, ignore_behavior_tokens: list[int] | None = None):
        self.only_train_response = only_train_response
        self.ignore_behavior_tokens = ignore_behavior_tokens if ignore_behavior_tokens is not None else []
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

    def __call__(self, batch: list[dict]) -> BatchEncoding:
        input_texts = [d["input_ids"] for d in batch]
        full_texts = [d["input_ids"] + d["labels"] for d in batch]

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
        # ignore behavior tokens
        for token in self.ignore_behavior_tokens:
            labels[labels == token] = -100
        if self.only_train_response or ('split' in batch[0] and batch[0]['split'] == 'valid'):
            # ignore input text
            labels[torch.where(inputs["labels"] != self.tokenizer.pad_token_id)] = -100

        inputs["labels"] = labels
        if "behavior" in batch[0]:
            # If the batch contains target behavior, add it to the inputs
            inputs["behavior"] = [d["behavior"] for d in batch]
        if "session_ids" in batch[0]:
            # If the batch contains session IDs, add it to the inputs
            session_ids = [d["session_ids"] for d in batch]
            max_length = max([len(sub) for sub in session_ids])
            session_ids = [session + [0] * (max_length - len(session)) for session in session_ids]
            inputs["session_ids"] = torch.tensor(session_ids, dtype=torch.long)
        if "extended_session_ids" in batch[0]:
            # If the batch contains extended session IDs, add it to the inputs
            extended_session_ids = [d["extended_session_ids"] for d in batch]
            max_length = max([len(sub) for sub in extended_session_ids])
            extended_session_ids = [session + [0] * (max_length - len(session)) for session in extended_session_ids]
            inputs["extended_session_ids"] = torch.tensor(extended_session_ids, dtype=torch.long)
        if "attention_mask" in batch[0]:
            # If the batch contains attention mask, add it to the inputs
            attention_masks = [d["attention_mask"] for d in batch]
            max_length = max([sub.shape[0] for sub in attention_masks])
            attention_mask = torch.full((len(attention_masks), 1, max_length, max_length), fill_value=torch.finfo(torch.float32).min, dtype=torch.float32)
            for i, mask in enumerate(attention_masks):
                attention_mask[i, 0, :mask.shape[0], :mask.shape[1]] = mask
            inputs["attention_mask"] = attention_mask
        if "time" in batch[0]:
            time = [d["time"] for d in batch]
            max_length = max([len(sub) for sub in time])
            time = [t + [-1] * (max_length - len(t)) for t in time]
            inputs["time"] = torch.tensor(time, dtype=torch.float32)

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
    def __init__(self, tokenizer: PreTrainedTokenizer, add_behavior_token: bool = True):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        # Allow batched inference
        self.tokenizer.padding_side = "left"
        self.add_behavior_token = add_behavior_token

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
            inputs["behavior"] = [d["behavior"] for d in batch]
        if "session_ids" in batch[0]:
            # If the batch contains session IDs, add it to the inputs
            session_ids = [d["session_ids"] for d in batch]
            max_length = max([len(sub) for sub in session_ids])
            session_ids = [[0] * (max_length - len(session)) + session for session in session_ids]
            inputs["session_ids"] = torch.tensor(session_ids, dtype=torch.long)
        if "extended_session_ids" in batch[0]:
            # If the batch contains extended session IDs, add it to the inputs
            extended_session_ids = [d["extended_session_ids"] for d in batch]
            max_extended_session_ids = [max(sub) for sub in extended_session_ids]
            max_length = max([len(sub) for sub in extended_session_ids])
            if self.add_behavior_token:
                extended_session_ids = [[0] * (max_length - len(session)) + session + [max_extended_session_id + 1] for session, max_extended_session_id in zip(extended_session_ids, max_extended_session_ids)]
            else:
                extended_session_ids = [[0] * (max_length - len(session)) + session for session in extended_session_ids]
            inputs["extended_session_ids"] = torch.tensor(extended_session_ids, dtype=torch.long)
        if "attention_mask" in batch[0]:
            # If the batch contains attention mask, add it to the inputs
            attention_masks = [d["attention_mask"] for d in batch]
            max_length = max([sub.shape[0] for sub in attention_masks])
            if self.add_behavior_token:
                attention_mask = torch.full((len(attention_masks), 1, max_length + 1, max_length + 1), fill_value=torch.finfo(torch.float32).min, dtype=torch.float32)
                for i, mask in enumerate(attention_masks):
                    new_mask = torch.full((mask.shape[0] + 1, mask.shape[1] + 1), fill_value=torch.finfo(torch.float32).min, dtype=torch.float32)
                    new_mask[:mask.shape[0], :mask.shape[1]] = mask
                    new_mask[-1, :-1] = 0  # all tokens can be seen from the first behavior token
                    attention_mask[i, 0, -new_mask.shape[0]:, -new_mask.shape[1]:] = new_mask
            else:
                attention_mask = torch.full((len(attention_masks), 1, max_length, max_length), fill_value=torch.finfo(torch.float32).min, dtype=torch.float32)
                for i, mask in enumerate(attention_masks):
                    attention_mask[i, 0, -mask.shape[0]:, -mask.shape[1]:] = mask
            inputs["attention_mask"] = attention_mask

        return (inputs, targets)
