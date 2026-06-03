import torch
from loguru import logger
from typing import Unpack
from functools import partial
from transformers.utils import can_return_tuple
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3 import Qwen3Model, Qwen3ForCausalLM, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import KwargsForCausalLM
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from SeqRec.models.generative.common.cache import prepare_cache_position_and_position_ids
from SeqRec.models.generative.common.wrappers import ExtendedSessionPositionMixin
from SeqRec.models.generative.common.session_masks import apply_attention_padding_mask, build_mask_context, build_incremental_causal_mask, build_session_item_in_item_mask, build_session_in_item_self_mask


class Qwen3SessionModel(Qwen3Model):
    def __init__(self, config: Qwen3Config):
        assert 'num_positions' in config and isinstance(config.num_positions, int), "Config must have 'num_positions' attribute for Qwen3SessionModel."
        assert 'model_max_length' in config and isinstance(config.model_max_length, int), "Config must have 'model_max_length' attribute for Qwen3SessionModel."
        super().__init__(config)
        self.in_item_mask = build_session_item_in_item_mask(
            num_positions=config.num_positions,
            model_max_length=config.model_max_length,
        )

    def _update_session_wise_causal_mask(
        self,
        attention_mask: torch.Tensor | None = None,
        input_tensor: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        session_ids: torch.LongTensor | None = None,  # [B, S]
    ) -> torch.Tensor:
        mask_ctx = build_mask_context(input_tensor, past_key_values)
        past_seen_tokens = mask_ctx.past_seen_tokens
        batch_size = mask_ctx.batch_size
        sequence_length = mask_ctx.sequence_length
        dtype, device = mask_ctx.dtype, mask_ctx.device
        min_dtype = mask_ctx.min_dtype
        if past_seen_tokens == 0:
            assert session_ids is not None, "Session IDs must be provided to generate session-wise causal mask."
            # during training or the first time to generate, generate the complete causal mask
            target_length = sequence_length
            causal_mask = build_session_in_item_self_mask(
                in_item_mask=self.in_item_mask,
                session_ids=session_ids,
                sequence_length=sequence_length,
                batch_size=batch_size,
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
            )
        else:
            # not the first time to generate, generate the causal mask for the new tokens
            target_length = len(cache_position) + past_seen_tokens
            causal_mask = build_incremental_causal_mask(
                sequence_length=sequence_length,
                target_length=target_length,
                cache_position=cache_position,
                batch_size=batch_size,
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
            )
        return apply_attention_padding_mask(
            causal_mask,
            attention_mask,
            target_length=target_length,
            min_dtype=min_dtype,
        )

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        session_ids: torch.LongTensor | None = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        cache_position, position_ids = prepare_cache_position_and_position_ids(
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
        )

        causal_mask = self._update_session_wise_causal_mask(
            attention_mask=attention_mask,
            input_tensor=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
            session_ids=session_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
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

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen3SessionWithTemperature(ExtendedSessionPositionMixin, Qwen3ForCausalLM):
    def __init__(self, config: Qwen3Config):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.init_custom_causal_lm(config, Qwen3SessionModel)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        session_ids: torch.LongTensor | None = None,
        extended_session_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        return self.forward_custom_causal_lm(
            labels=labels,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            logits_to_keep=logits_to_keep,
            model_kwargs=dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                session_ids=session_ids,
            ),
            extra_kwargs=kwargs,
            wrapper_kwargs=dict(extended_session_ids=extended_session_ids),
        )
