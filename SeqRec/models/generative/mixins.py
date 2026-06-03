import torch
from dataclasses import dataclass
from functools import partial
from loguru import logger
from typing import Any
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.loss.loss_utils import ForCausalLMLoss


def prepare_cache_position_and_position_ids(
    *,
    past_key_values,
    inputs_embeds: torch.Tensor,
    cache_position: torch.LongTensor | None,
    position_ids: torch.LongTensor | None,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    return cache_position, position_ids


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


class TemperatureMixin:
    def init_temperature(self):
        self.temperature = 1.0

    def set_hyper(self, temperature: float):
        self.temperature = temperature

    def apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "temperature"), "Model must have a temperature attribute."
        return logits / self.temperature


class TemperatureCausalLMLossMixin(TemperatureMixin):

    @property
    def loss_function(self):
        if hasattr(self, "_loss_function"):
            return self._loss_function

        def ForCausalLMLossWithTemperature(
            logits,
            labels,
            vocab_size: int,
            num_items_in_batch: int | None = None,
            ignore_index: int = -100,
            shift_labels: torch.Tensor | None = None,
            **kwargs,
        ) -> torch.Tensor:
            logits = self.apply_temperature(logits)
            return ForCausalLMLoss(
                logits,
                labels,
                vocab_size=vocab_size,
                num_items_in_batch=num_items_in_batch,
                ignore_index=ignore_index,
                shift_labels=shift_labels,
                **kwargs,
            )

        self._loss_function = ForCausalLMLossWithTemperature
        return self._loss_function


class CustomCausalLMWrapperMixin(TemperatureCausalLMLossMixin):
    def init_custom_causal_lm(self, config: Any, model_cls: type):
        self.model = model_cls(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.init_temperature()

    def prepare_custom_causal_lm_inputs(
        self,
        *,
        position_ids: torch.LongTensor | None,
        cache_position: torch.LongTensor | None,
        model_kwargs: dict[str, Any],
        extra_kwargs: dict[str, Any],
        wrapper_kwargs: dict[str, Any],
    ) -> tuple[torch.LongTensor | None, dict[str, Any]]:
        return position_ids, model_kwargs

    def forward_custom_causal_lm(
        self,
        *,
        labels: torch.LongTensor | None,
        position_ids: torch.LongTensor | None,
        output_attentions: bool | None,
        output_hidden_states: bool | None,
        logits_to_keep: int | torch.Tensor,
        model_kwargs: dict[str, Any],
        extra_kwargs: dict[str, Any],
        wrapper_kwargs: dict[str, Any] | None = None,
    ) -> CausalLMOutputWithPast:
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        position_ids, model_kwargs = self.prepare_custom_causal_lm_inputs(
            position_ids=position_ids,
            cache_position=model_kwargs.get("cache_position"),
            model_kwargs=model_kwargs,
            extra_kwargs=extra_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )
        model_call_kwargs = {**model_kwargs, **extra_kwargs}
        outputs = self.model(
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **model_call_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **extra_kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ExtendedSessionPositionMixin(CustomCausalLMWrapperMixin):
    def prepare_custom_causal_lm_inputs(
        self,
        *,
        position_ids: torch.LongTensor | None,
        cache_position: torch.LongTensor | None,
        model_kwargs: dict[str, Any],
        extra_kwargs: dict[str, Any],
        wrapper_kwargs: dict[str, Any],
    ) -> tuple[torch.LongTensor | None, dict[str, Any]]:
        extended_session_ids = wrapper_kwargs.get("extended_session_ids")
        if cache_position is not None and cache_position.min() == 0:
            if extended_session_ids is not None:
                self.max_extended_session_id = extended_session_ids.max(dim=-1)[0]
        elif cache_position is not None:
            if extended_session_ids is not None:
                assert cache_position.shape[-1] == 1
                if self.max_extended_session_id.ndim == 1:
                    self.max_extended_session_id += 1
                    extended_session_ids = self.max_extended_session_id[:, None]
                else:
                    self.max_extended_session_id += 1
                    extended_session_ids = self.max_extended_session_id[None]
        if extended_session_ids is not None:
            position_ids = extended_session_ids
        return position_ids, model_kwargs
