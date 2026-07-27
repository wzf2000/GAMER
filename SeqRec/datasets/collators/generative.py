import copy
import torch
from transformers import BatchEncoding
from transformers.tokenization_utils import PreTrainedTokenizer


def _pad_sequence_field(
    batch: list[dict],
    field: str,
    pad_value: int | float,
    *,
    dtype: torch.dtype,
    padding_side: str = "right",
) -> torch.Tensor:
    values = [d[field] for d in batch]
    max_length = max(len(sub) for sub in values)
    if padding_side == "left":
        padded = [[pad_value] * (max_length - len(sub)) + sub for sub in values]
    else:
        padded = [sub + [pad_value] * (max_length - len(sub)) for sub in values]
    return torch.tensor(padded, dtype=dtype)


def _copy_optional_list_fields(inputs: BatchEncoding, batch: list[dict], fields: tuple[str, ...]):
    for field in fields:
        if field in batch[0]:
            inputs[field] = [d[field] for d in batch]


def _add_common_optional_fields(inputs: BatchEncoding, batch: list[dict], *, include_uid: bool = False):
    fields = ("behavior", "inters_item_list")
    if include_uid:
        fields = ("uid",) + fields
    _copy_optional_list_fields(inputs, batch, fields)


def _add_right_padded_sequence_fields(inputs: BatchEncoding, batch: list[dict]):
    if "session_ids" in batch[0]:
        inputs["session_ids"] = _pad_sequence_field(batch, "session_ids", 0, dtype=torch.long)
    if "extended_session_ids" in batch[0]:
        inputs["extended_session_ids"] = _pad_sequence_field(batch, "extended_session_ids", 0, dtype=torch.long)
    if "actions" in batch[0]:
        inputs["actions"] = _pad_sequence_field(batch, "actions", 100, dtype=torch.long)
    if "time" in batch[0]:
        inputs["time"] = _pad_sequence_field(batch, "time", -1, dtype=torch.float32)


def _align_sequence_to_length(
    sequence: list[int],
    length: int,
    pad_value: int,
    *,
    truncation_side: str = "left",
    padding_side: str = "right",
) -> list[int]:
    if len(sequence) > length:
        if truncation_side == "left":
            sequence = sequence[-length:]
        else:
            sequence = sequence[:length]
    if len(sequence) < length:
        padding = [pad_value] * (length - len(sequence))
        if padding_side == "left":
            sequence = padding + sequence
        else:
            sequence = sequence + padding
    return sequence


def _pad_ranking_field(
    inputs: BatchEncoding,
    batch: list[dict],
    field: str,
    *,
    pad_value: int,
    append_label_value,
    dtype: torch.dtype,
):
    if field not in batch[0]:
        return
    max_length = inputs["input_ids"].shape[1]
    values = []
    for sample in batch:
        label_len = sample.get("_ranking_label_len", 1)
        label_values = append_label_value(sample, label_len)
        values.append(
            _align_sequence_to_length(
                list(sample[field]) + label_values,
                max_length,
                pad_value,
                truncation_side="left",
                padding_side="right",
            )
        )
    inputs[field] = torch.tensor(values, dtype=dtype)


def _add_ranking_sequence_fields(inputs: BatchEncoding, batch: list[dict]):
    _pad_ranking_field(
        inputs,
        batch,
        "relation_actions",
        pad_value=0,
        append_label_value=lambda _sample, label_len: [0] * label_len,
        dtype=torch.long,
    )
    _pad_ranking_field(
        inputs,
        batch,
        "actions",
        pad_value=0,
        append_label_value=lambda _sample, label_len: [0] * label_len,
        dtype=torch.long,
    )
    _pad_ranking_field(
        inputs,
        batch,
        "session_ids",
        pad_value=0,
        append_label_value=lambda sample, label_len: [sample["session_ids"][-1] if sample["session_ids"] else 0] * label_len,
        dtype=torch.long,
    )
    _pad_ranking_field(
        inputs,
        batch,
        "extended_session_ids",
        pad_value=0,
        append_label_value=lambda sample, label_len: [
            (sample["extended_session_ids"][-1] + index + 1) if sample["extended_session_ids"] else index
            for index in range(label_len)
        ],
        dtype=torch.long,
    )


def _add_left_padded_decoder_test_fields(
    inputs: BatchEncoding,
    batch: list[dict],
    *,
    add_behavior_token: bool,
):
    if "session_ids" in batch[0]:
        session_ids = [d["session_ids"] for d in batch]
        max_length = max(len(sub) for sub in session_ids)
        if add_behavior_token:
            max_session_ids = [max(sub) for sub in session_ids]
            session_ids = [[0] * (max_length - len(session)) + session for session in session_ids]
            session_ids = [
                session_id + [max_session_id + 1]
                for session_id, max_session_id in zip(session_ids, max_session_ids)
            ]
        else:
            session_ids = [[0] * (max_length - len(session)) + session for session in session_ids]
        inputs["session_ids"] = torch.tensor(session_ids, dtype=torch.long)
    if "extended_session_ids" in batch[0]:
        extended_session_ids = [d["extended_session_ids"] for d in batch]
        max_extended_session_ids = [max(sub) for sub in extended_session_ids]
        max_length = max(len(sub) for sub in extended_session_ids)
        if add_behavior_token:
            extended_session_ids = [
                [0] * (max_length - len(session)) + session + [max_extended_session_id + 1]
                for session, max_extended_session_id in zip(extended_session_ids, max_extended_session_ids)
            ]
        else:
            extended_session_ids = [
                [0] * (max_length - len(session)) + session
                for session in extended_session_ids
            ]
        inputs["extended_session_ids"] = torch.tensor(extended_session_ids, dtype=torch.long)
    if "actions" in batch[0]:
        inputs["actions"] = _pad_sequence_field(
            batch,
            "actions",
            100,
            dtype=torch.long,
            padding_side="left",
        )


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
        inputs["split"] = batch[0].get("split", "train")
        _add_common_optional_fields(inputs, batch)
        _add_right_padded_sequence_fields(inputs, batch)

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
        inputs["split"] = batch[0].get("split", "train")
        _add_common_optional_fields(inputs, batch)
        _add_right_padded_sequence_fields(inputs, batch)

        return inputs


class DecoderOnlyRankingCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"

    def __call__(self, batch: list[dict]) -> BatchEncoding:
        batch = [dict(sample) for sample in batch]
        for sample in batch:
            sample["_ranking_label_len"] = len(
                self.tokenizer.encode(sample["labels"], add_special_tokens=False)
            )

        input_texts = [sample["input_ids"] for sample in batch]
        full_texts = [sample["input_ids"] + sample["labels"] for sample in batch]

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
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[torch.where(inputs["labels"] != self.tokenizer.pad_token_id)] = -100

        inputs["labels"] = labels
        inputs["split"] = batch[0].get("split", "train")
        _add_common_optional_fields(inputs, batch)
        _add_ranking_sequence_fields(inputs, batch)

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

        _add_common_optional_fields(inputs, batch, include_uid=True)
        _add_right_padded_sequence_fields(inputs, batch)

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
        _add_common_optional_fields(inputs, batch, include_uid=True)
        _add_left_padded_decoder_test_fields(
            inputs,
            batch,
            add_behavior_token=self.add_behavior_token,
        )

        return (inputs, targets)
