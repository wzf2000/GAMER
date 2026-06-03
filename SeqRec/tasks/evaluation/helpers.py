import torch
from typing import Any, Callable

from SeqRec.generation.trie import Trie, prefix_allowed_tokens_fn, prefix_allowed_tokens_fn_by_last_token
from SeqRec.models.generative.registry import (
    backbone_uses_actions,
    backbone_uses_sessions,
    is_decoder_only_backbone,
)


def get_generation_model(model: Any):
    from transformers.generation import GenerationMixin

    return model if isinstance(model, GenerationMixin) else model.module


def build_generation_kwargs(
    *,
    backbone: str,
    inputs: Any,
    max_new_tokens: int,
    prefix_allowed_tokens_fn: Callable,
    num_beams: int,
    device: torch.device,
    decoder_input_ids: list[list[int]] | None = None,
    decoder_start_token_id: int | None = None,
) -> dict:
    gen_kwargs = dict(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        output_scores=True,
        return_dict_in_generate=True,
        early_stopping=True,
    )
    # Only pass session / action fields the backbone declares AND the collated
    # inputs actually carry. MB datasets, for example, produce ``actions`` for
    # Qwen3Multi but never produce ``session_ids``. ``getattr`` (not ``inputs.get``)
    # is required because the test tasks sometimes reassign ``inputs.actions``
    # via attribute set, which does not propagate to BatchEncoding's dict.
    if backbone_uses_sessions(backbone):
        session_ids = getattr(inputs, "session_ids", None)
        if session_ids is not None:
            gen_kwargs["session_ids"] = session_ids
            gen_kwargs["extended_session_ids"] = getattr(inputs, "extended_session_ids")
    if backbone_uses_actions(backbone):
        actions = getattr(inputs, "actions", None)
        if actions is not None:
            gen_kwargs["actions"] = actions
    if not is_decoder_only_backbone(backbone):
        if decoder_input_ids is None:
            assert decoder_start_token_id is not None
            decoder_input_ids = [[decoder_start_token_id] for _ in range(inputs.input_ids.shape[0])]
        gen_kwargs["decoder_input_ids"] = torch.tensor(decoder_input_ids, device=device)
    return gen_kwargs


def prepare_behavior_generation_prompt(
    *,
    inputs: Any,
    tokenizer: Any,
    dataset: Any,
    behaviors: list[str],
    device: torch.device,
    is_decoder_only: bool,
    decoder_start_token_id: int,
) -> tuple[list[list[int]] | None, int]:
    behavior_texts = ["".join(dataset.get_behavior_tokens(behavior)) for behavior in behaviors]
    behavior_tokens = tokenizer.batch_encode_plus(behavior_texts, add_special_tokens=False)
    behavior_token_ids = behavior_tokens["input_ids"]
    behavior_token_lens = [len(tokens) for tokens in behavior_token_ids]
    assert len(set(behavior_token_lens)) == 1, "All behavior tokens should be of the same length."
    behavior_token_num = behavior_token_lens[0]

    if is_decoder_only:
        inputs.input_ids = torch.cat(
            [inputs.input_ids, torch.tensor(behavior_token_ids, device=device)],
            dim=1,
        )
        inputs.attention_mask = torch.cat(
            [inputs.attention_mask, torch.tensor(behavior_tokens["attention_mask"], device=device)],
            dim=1,
        )
        if "actions" in inputs:
            actions = [[dataset.behavior_level[behavior]] for behavior in behaviors]
            inputs.actions = torch.cat([inputs.actions, torch.tensor(actions, device=device)], dim=1)
        return None, behavior_token_num

    decoder_input_ids = [[decoder_start_token_id] + tokens for tokens in behavior_token_ids]
    return decoder_input_ids, behavior_token_num


def slice_decoder_only_output(backbone: str, output_ids: torch.Tensor, item_len: int) -> torch.Tensor:
    if is_decoder_only_backbone(backbone):
        return output_ids[:, -item_len:]
    return output_ids


def get_item_token_info(tokenizer: Any, items: list[str], pad_token_id: int) -> tuple[list[list[int]], int, set[int]]:
    item_tokens = tokenizer.batch_encode_plus(items, add_special_tokens=False)["input_ids"]
    item_len = len(item_tokens[0])
    last_token_set = {tokens[-1] for tokens in item_tokens}
    last_token_set.add(pad_token_id)
    return item_tokens, item_len, last_token_set


def build_candidate_prefix_fn(
    *,
    backbone: str,
    tokenizer: Any,
    config: Any,
    items: list[str],
    last_token_set: set[int] | None = None,
):
    if is_decoder_only_backbone(backbone):
        candidate_tokens = tokenizer.batch_encode_plus(items, add_special_tokens=False)["input_ids"]
        if last_token_set is None:
            last_token_set = {tokens[-1] for tokens in candidate_tokens}
            last_token_set.add(config.pad_token_id)
        return prefix_allowed_tokens_fn_by_last_token(Trie(candidate_tokens), last_token_set)
    candidate_tokens = tokenizer.batch_encode_plus(items)["input_ids"]
    candidate_tokens = [[config.decoder_start_token_id] + tokens for tokens in candidate_tokens]
    return prefix_allowed_tokens_fn(Trie(candidate_tokens))


def build_behavior_prefix_fns(
    *,
    backbone: str,
    tokenizer: Any,
    config: Any,
    dataset: Any,
    last_token_set: set[int],
) -> dict[str, Callable[[int, torch.Tensor], list[int]]]:
    prefix_fns = {}
    for behavior in dataset.behaviors:
        items = list(dataset.get_all_items(behavior))
        prefix_fns[behavior] = build_candidate_prefix_fn(
            backbone=backbone,
            tokenizer=tokenizer,
            config=config,
            items=items,
            last_token_set=last_token_set,
        )
    return prefix_fns
