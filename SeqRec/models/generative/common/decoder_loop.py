import torch
from dataclasses import dataclass
from functools import partial
from loguru import logger
from typing import Any
from transformers.cache_utils import Cache, DynamicCache

from SeqRec.models.generative.common.cache import prepare_cache_position_and_position_ids


@dataclass
class DecoderForwardState:
    inputs_embeds: torch.FloatTensor
    past_key_values: Cache | None
    cache_position: torch.LongTensor
    position_ids: torch.LongTensor
    use_cache: bool
    output_attentions: bool
    output_hidden_states: bool


@dataclass
class DecoderLayerLoopOutput:
    hidden_states: torch.FloatTensor
    all_hidden_states: tuple[torch.FloatTensor, ...] | None
    all_self_attns: tuple[torch.Tensor, ...] | None


def prepare_decoder_forward_state(
    model: torch.nn.Module,
    *,
    input_ids: torch.LongTensor | None,
    inputs_embeds: torch.FloatTensor | None,
    past_key_values: Cache | None,
    cache_position: torch.LongTensor | None,
    position_ids: torch.LongTensor | None,
    use_cache: bool | None,
    output_attentions: bool | None,
    output_hidden_states: bool | None,
) -> DecoderForwardState:
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else model.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if model.gradient_checkpointing and model.training and use_cache:
        logger.warning("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
        use_cache = False

    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

    if inputs_embeds is None:
        inputs_embeds = model.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    cache_position, position_ids = prepare_cache_position_and_position_ids(
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        position_ids=position_ids,
    )

    return DecoderForwardState(
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values,
        cache_position=cache_position,
        position_ids=position_ids,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )


def init_cross_level_cache_state(model: torch.nn.Module):
    model.cross_past_key_values = None
    model.multi_self_mask = None
    model.multi_cross_mask = None


def reset_cross_level_cache_if_needed(
    model: torch.nn.Module,
    *,
    use_cache: bool,
    past_key_values: Cache | None,
):
    if use_cache and past_key_values is not None and past_key_values.get_seq_length() == 0:
        model.cross_past_key_values = DynamicCache()


def run_multi_cross_decoder_layers(
    model: torch.nn.Module,
    *,
    hidden_states: torch.FloatTensor,
    position_indices: torch.LongTensor,
    behavior_indices: torch.LongTensor,
    action_indices: torch.LongTensor,
    multi_self_mask: torch.Tensor | None,
    multi_cross_mask: torch.Tensor | None,
    position_ids: torch.LongTensor,
    past_key_values: Cache | None,
    cross_past_key_values: Cache | None,
    output_attentions: bool,
    output_hidden_states: bool,
    use_cache: bool,
    cache_position: torch.LongTensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    flash_attn_kwargs: dict[str, Any],
) -> DecoderLayerLoopOutput:
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in model.layers[: model.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if model.gradient_checkpointing and model.training:
            layer_outputs = model._gradient_checkpointing_func(
                partial(decoder_layer.__call__, **flash_attn_kwargs),
                hidden_states,
                position_indices,
                behavior_indices,
                action_indices,
                multi_self_mask,
                multi_cross_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                cross_past_key_values,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                position_indices,
                behavior_indices,
                action_indices,
                multi_self_mask=multi_self_mask,
                multi_cross_mask=multi_cross_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                cross_past_key_value=cross_past_key_values,
                **flash_attn_kwargs,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    return DecoderLayerLoopOutput(
        hidden_states=hidden_states,
        all_hidden_states=all_hidden_states,
        all_self_attns=all_self_attns,
    )


def run_temporal_hierarchical_decoder_layers(
    model: torch.nn.Module,
    *,
    hidden_states: torch.FloatTensor,
    position_indices: torch.LongTensor,
    behavior_indices: torch.LongTensor,
    action_indices: torch.LongTensor,
    behavior_level_indices: torch.LongTensor,
    key_behavior_level_indices: torch.LongTensor,
    causal_mask: torch.Tensor | None,
    position_ids: torch.LongTensor,
    past_key_values: Cache | None,
    output_attentions: bool,
    output_hidden_states: bool,
    use_cache: bool,
    cache_position: torch.LongTensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    flash_attn_kwargs: dict[str, Any],
) -> DecoderLayerLoopOutput:
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in model.layers[: model.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if model.gradient_checkpointing and model.training:
            layer_outputs = model._gradient_checkpointing_func(
                partial(decoder_layer.__call__, **flash_attn_kwargs),
                hidden_states,
                position_indices,
                behavior_indices,
                action_indices,
                behavior_level_indices,
                key_behavior_level_indices,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                position_indices,
                behavior_indices,
                action_indices,
                behavior_level_indices,
                key_behavior_level_indices,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )
        hidden_states = layer_outputs[0]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    return DecoderLayerLoopOutput(
        hidden_states=hidden_states,
        all_hidden_states=all_hidden_states,
        all_self_attns=all_self_attns,
    )
