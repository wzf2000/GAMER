import torch
from torch import nn
from loguru import logger
from typing import Unpack, Callable, Optional, Tuple
from functools import partial
from transformers.utils import can_return_tuple
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3 import Qwen3ForCausalLM, Qwen3PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import KwargsForCausalLM, Qwen3RMSNorm, Qwen3RotaryEmbedding, QWEN3_INPUTS_DOCSTRING
from transformers.models.qwen3_moe import Qwen3MoeConfig
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import add_start_docstrings_to_model_forward
from transformers.cache_utils import SlidingWindowCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm, apply_rotary_pos_emb, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from SeqRec.models.generative.qwen3._decoder_base import Qwen3DecoderModelBase
from SeqRec.models.generative.qwen3.moe_ffn import MyQwen3SparseMLP, PBATransformerSparseMLP, DenseMLP, RouterMoeBlock
from SeqRec.models.generative.qwen3.multi_router import Qwen3MultiDecoderRouter
from SeqRec.models.generative.common.attention import CrossBehaviorAttentionMixin, run_multi_level_cross_attention_block, run_multi_level_self_attention_block
from SeqRec.models.generative.common.cache import prepare_cache_position_and_position_ids
from SeqRec.models.generative.common.decoder_loop import init_cross_level_cache_state, prepare_decoder_forward_state, reset_cross_level_cache_if_needed, run_multi_cross_decoder_layers
from SeqRec.models.generative.common.wrappers import CustomCausalLMWrapperMixin
from SeqRec.models.generative.common.session_masks import apply_attention_padding_mask, build_action_level_cross_mask, build_flattened_in_item_mask, build_in_item_self_mask, build_incremental_causal_mask, build_mask_context, extend_cached_cross_mask


class Qwen3MultiAttention(CrossBehaviorAttentionMixin, nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, is_cross: bool):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = (self.head_dim) ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None

        self.is_cross = is_cross
        if self.is_cross:
            self.init_cross_behavior_attention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        action_index: torch.Tensor = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.is_cross:
            q_behavior_embedding, k_behavior_embedding, v_behavior_embedding = self.get_cross_behavior_embeddings(
                hidden_states,
                action_index,
            )
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape) + q_behavior_embedding).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape) + k_behavior_embedding).transpose(1, 2)
            value_states = (self.v_proj(hidden_states).view(hidden_shape) + v_behavior_embedding).transpose(1, 2)
        else:
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if self.is_cross:
            attn_output = self.apply_cross_behavior_gate(attn_output, hidden_states)
        return attn_output, attn_weights


class Qwen3MultiDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, is_sparse: bool, behavior_injection: bool, is_cross: bool):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.is_sparse = is_sparse
        self.behavior_injection = behavior_injection
        self.is_cross = is_cross

        self.self_attn = Qwen3MultiAttention(config=config, layer_idx=layer_idx, is_cross=False)

        if self.is_cross:
            self.cross_attn = Qwen3MultiAttention(config=config, layer_idx=layer_idx, is_cross=True)
            self.post_self_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if "mlp_type" not in config:
            self.mlp_type = "PBATransformer"
        else:
            self.mlp_type = config.mlp_type
        if self.mlp_type == "Qwen3":
            self.mlp = MyQwen3SparseMLP(config, is_sparse=self.is_sparse, behavior_injection=self.behavior_injection)
        elif self.mlp_type == "dense":
            self.mlp = DenseMLP(config)
        elif self.mlp_type == "RouterMoe":
            self.mlp = RouterMoeBlock(config)
        else:
            self.mlp = PBATransformerSparseMLP(config, is_sparse=self.is_sparse, behavior_injection=self.behavior_injection)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_cross_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout_rate)
        if (
            config.sliding_window and config._attn_implementation != "flash_attention_2"
        ):  # diff with Llama is this warning
            logger.warning(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_indices: torch.Tensor,
        behavior_indices: torch.Tensor = None,
        action_indices: torch.Tensor = None,
        multi_self_mask: Optional[torch.Tensor] = None,
        multi_cross_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        cross_past_key_value: Optional[Cache] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states, self_attn_weights = run_multi_level_self_attention_block(
            self,
            hidden_states=hidden_states,
            multi_self_mask=multi_self_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            kwargs=kwargs,
        )
        hidden_states = run_multi_level_cross_attention_block(
            self,
            hidden_states=hidden_states,
            action_indices=action_indices,
            multi_cross_mask=multi_cross_mask,
            position_ids=position_ids,
            cross_past_key_value=cross_past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            kwargs=kwargs,
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_cross_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, position_indices, behavior_indices)
        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen3MultiModelBase(Qwen3DecoderModelBase):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3DecoderLayer`]

    Args:
        config: Qwen3MoeConfig
    """

    decoder_layer_cls = Qwen3MultiDecoderLayer
    router_cls = Qwen3MultiDecoderRouter


class Qwen3MultiModel(Qwen3MultiModelBase):
    def __init__(self, config: Qwen3MoeConfig):
        assert 'num_positions' in config and isinstance(config.num_positions, int), "Config must have 'num_positions' attribute for Qwen3MultiModel."
        assert 'model_max_length' in config and isinstance(config.model_max_length, int), "Config must have 'model_max_length' attribute for Qwen3MultiModel."
        super().__init__(config)
        self.behavior_maps = config.behavior_maps
        self.in_item_mask = build_flattened_in_item_mask(
            num_positions=config.num_positions,
            model_max_length=config.model_max_length,
        )
        init_cross_level_cache_state(self)
        logger.info(f"Using cross_mask_type: {getattr(config, 'cross_mask_type', 'level')} for Qwen3MultiModel.")

    def _update_session_multi_cross_mask(
        self,
        attention_mask: torch.Tensor | None = None,
        input_tensor: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        session_ids: torch.LongTensor | None = None,  # [B, S]
        actions: torch.LongTensor | None = None,  # [B, S]
    ) -> torch.Tensor:
        mask_ctx = build_mask_context(input_tensor, past_key_values)
        past_seen_tokens = mask_ctx.past_seen_tokens
        batch_size = mask_ctx.batch_size
        sequence_length = mask_ctx.sequence_length
        dtype, device = mask_ctx.dtype, mask_ctx.device
        min_dtype = mask_ctx.min_dtype
        if past_seen_tokens == 0:
            # during training or the first time to generate, generate the complete causal mask
            target_length = sequence_length
            causal_mask = build_action_level_cross_mask(
                actions=actions,
                in_item_mask=self.in_item_mask,
                sequence_length=sequence_length,
                batch_size=batch_size,
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                mask_type=getattr(self.config, "cross_mask_type", "level"),
                soft_scale=float(getattr(self.config, "cross_mask_soft_scale", 1.0)),
                num_behavior=int(getattr(self.config, "num_behavior", 1)),
            )
            if past_key_values is not None:
                self.multi_cross_mask = causal_mask[:, :, -1, :]
        else:
            # not the first time to generate, generate the causal mask for the new tokens
            target_length = len(cache_position) + past_seen_tokens
            self.multi_cross_mask, causal_mask = extend_cached_cross_mask(
                self.multi_cross_mask,
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

    def _update_session_wise_causal_mask(
        self,
        attention_mask: torch.Tensor | None = None,
        input_tensor: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        session_ids: torch.LongTensor | None = None,  # [B, S]
        actions: torch.LongTensor | None = None,  # [B, S]
    ) -> torch.Tensor:
        mask_ctx = build_mask_context(input_tensor, past_key_values)
        past_seen_tokens = mask_ctx.past_seen_tokens
        batch_size = mask_ctx.batch_size
        sequence_length = mask_ctx.sequence_length
        dtype, device = mask_ctx.dtype, mask_ctx.device
        min_dtype = mask_ctx.min_dtype
        if past_seen_tokens == 0:
            # during training or the first time to generate, generate the complete causal mask
            target_length = sequence_length
            causal_mask = build_in_item_self_mask(
                in_item_mask=self.in_item_mask,
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
        actions: torch.LongTensor | None = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        state = prepare_decoder_forward_state(
            self,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        inputs_embeds = state.inputs_embeds
        past_key_values = state.past_key_values
        cache_position = state.cache_position
        position_ids = state.position_ids
        use_cache = state.use_cache
        output_attentions = state.output_attentions
        output_hidden_states = state.output_hidden_states

        reset_cross_level_cache_if_needed(self, use_cache=use_cache, past_key_values=past_key_values)

        position_indices, behavior_indices, action_indices = self.router(input_ids, cache_position=cache_position)

        multi_self_mask = self._update_session_wise_causal_mask(
            attention_mask=attention_mask,
            input_tensor=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
            session_ids=session_ids,
            actions=actions,
        )

        multi_cross_mask = self._update_session_multi_cross_mask(
            attention_mask=attention_mask,
            input_tensor=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
            session_ids=session_ids,
            actions=actions,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        loop_outputs = run_multi_cross_decoder_layers(
            self,
            hidden_states=hidden_states,
            position_indices=position_indices,
            behavior_indices=behavior_indices,
            action_indices=action_indices,
            multi_self_mask=multi_self_mask,
            multi_cross_mask=multi_cross_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cross_past_key_values=self.cross_past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            flash_attn_kwargs=flash_attn_kwargs,
        )
        hidden_states = loop_outputs.hidden_states
        all_hidden_states = loop_outputs.all_hidden_states
        all_self_attns = loop_outputs.all_self_attns

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


class Qwen3MultiWithTemperature(CustomCausalLMWrapperMixin, Qwen3ForCausalLM):
    def __init__(self, config: Qwen3MoeConfig):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.init_custom_causal_lm(config, Qwen3MultiModel)

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
        actions: torch.LongTensor | None = None,
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
                actions=actions,
            ),
            extra_kwargs=kwargs,
        )
