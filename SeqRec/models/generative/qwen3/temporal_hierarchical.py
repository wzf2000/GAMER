from typing import Callable, Optional, Tuple, Unpack

import torch
from loguru import logger
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3 import Qwen3ForCausalLM, Qwen3PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import KwargsForCausalLM, Qwen3RMSNorm, Qwen3RotaryEmbedding
from transformers.models.qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import can_return_tuple

from SeqRec.models.generative.qwen3._decoder_base import Qwen3DecoderModelBase
from SeqRec.models.generative.qwen3.moe_ffn import DenseMLP, MyQwen3SparseMLP, PBATransformerSparseMLP, RouterMoeBlock
from SeqRec.models.generative.qwen3.multi_router import Qwen3MultiDecoderRouter
from SeqRec.models.generative.common.cache import prepare_cache_position_and_position_ids
from SeqRec.models.generative.common.decoder_loop import prepare_decoder_forward_state, run_temporal_hierarchical_decoder_layers
from SeqRec.models.generative.common.wrappers import CustomCausalLMWrapperMixin
from SeqRec.models.generative.common.session_masks import apply_attention_padding_mask, build_mask_context, build_in_item_self_mask, build_incremental_causal_mask


class Qwen3TemporalHierarchicalAttention(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, is_temporal_hierarchical: bool):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.is_temporal_hierarchical = is_temporal_hierarchical
        self.th_attention_mode = getattr(config, "th_attention_mode", "relation_bias")

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
        self.q_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None

        if self.is_temporal_hierarchical:
            self.behavior_embedding_dim = config.head_dim
            self.q_behavior_embedding = nn.Embedding(
                config.num_behavior + 1,
                config.num_attention_heads * self.behavior_embedding_dim,
            )
            self.k_behavior_embedding = nn.Embedding(
                config.num_behavior + 1,
                config.num_key_value_heads * self.behavior_embedding_dim,
            )
            self.v_behavior_embedding = nn.Embedding(
                config.num_behavior + 1,
                config.num_key_value_heads * self.behavior_embedding_dim,
            )
            if self.th_attention_mode not in ("relation_bias", "multi_view"):
                raise ValueError("th_attention_mode must be 'relation_bias' or 'multi_view'.")
            self.use_relation_bias = self.th_attention_mode == "relation_bias" or bool(
                getattr(config, "th_multi_view_use_relation_bias", False)
            )
            if self.use_relation_bias:
                self.th_relation_bias_scale = float(getattr(config, "th_relation_bias_scale", 1.0))
                self.th_relation_bias_learnable_scale = bool(getattr(config, "th_relation_bias_learnable_scale", False))
                if self.th_relation_bias_learnable_scale:
                    alpha_init = float(getattr(config, "th_relation_bias_alpha_init", self.th_relation_bias_scale))
                    self.th_relation_bias_alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
                self.th_relation_bias_type = getattr(config, "th_relation_bias_type", "table")
                if self.th_relation_bias_type not in ("table", "factorized"):
                    raise ValueError("th_relation_bias_type must be 'table' or 'factorized'.")
                if self.th_relation_bias_type == "table":
                    bias = torch.zeros(config.num_behavior + 1, config.num_behavior + 1, config.num_attention_heads)
                    if bool(getattr(config, "th_relation_bias_trainable", True)):
                        self.level_pair_bias = nn.Parameter(bias)
                    else:
                        self.register_buffer("level_pair_bias", bias)
                else:
                    self.th_relation_bias_rank = int(getattr(config, "th_relation_bias_rank", 4))
                    if self.th_relation_bias_rank <= 0:
                        raise ValueError("th_relation_bias_rank must be positive.")
                    self.level_query_bias_factor = nn.Parameter(
                        torch.empty(config.num_behavior + 1, config.num_attention_heads, self.th_relation_bias_rank)
                    )
                    self.level_key_bias_factor = nn.Parameter(
                        torch.empty(config.num_behavior + 1, config.num_attention_heads, self.th_relation_bias_rank)
                    )
            self.gating = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
            self.act_fn = ACT2FN[config.hidden_act]
            if self.use_relation_bias:
                self._init_level_pair_bias()
            if self.th_attention_mode == "multi_view":
                self.th_multi_view_mode = getattr(config, "th_multi_view_mode", "hard")
                if self.th_multi_view_mode not in ("hard", "soft", "gated"):
                    raise ValueError("th_multi_view_mode must be 'hard', 'soft', or 'gated'.")
                if self.th_multi_view_mode == "hard":
                    view_ids = self._build_multi_view_head_ids(config)
                    self.register_buffer("multi_view_head_ids", view_ids, persistent=False)
                else:
                    view_ids = self._build_multi_view_type_ids(config)
                    self.register_buffer("multi_view_type_ids", view_ids, persistent=False)
                    self.th_multi_view_soft_bias_scale = float(getattr(config, "th_multi_view_soft_bias_scale", 1.0))
                    if self.th_multi_view_mode == "gated":
                        gate_logits = torch.zeros(config.num_attention_heads, len(view_ids))
                        if getattr(config, "th_multi_view_gate_init", "uniform") == "allocation":
                            hard_view_ids = self._build_multi_view_head_ids(config)
                            for head_idx, view_id in enumerate(hard_view_ids.tolist()):
                                matches = (view_ids == view_id).nonzero(as_tuple=False)
                                if matches.numel() > 0:
                                    gate_logits[head_idx, matches[0].item()] = 2.0
                        self.multi_view_gate_logits = nn.Parameter(gate_logits)

    def _init_level_pair_bias(self):
        init_type = getattr(self.config, "th_relation_bias_init", "zero")
        if getattr(self, "th_relation_bias_type", "table") == "factorized":
            self._init_factorized_level_pair_bias(init_type)
            return
        if init_type == "zero":
            return
        if init_type != "soft":
            raise ValueError("th_relation_bias_init must be 'zero' or 'soft'.")
        scale = float(getattr(self.config, "th_relation_bias_soft_scale", 0.1))
        with torch.no_grad():
            levels = torch.arange(self.config.num_behavior + 1, dtype=self.level_pair_bias.dtype)
            level_diff = levels[:, None] - levels[None, :]
            bias = level_diff.clamp(max=0.0) * scale
            self.level_pair_bias.copy_(bias[:, :, None].expand_as(self.level_pair_bias))

    def _init_factorized_level_pair_bias(self, init_type: str):
        if init_type not in ("zero", "soft"):
            raise ValueError("th_relation_bias_init must be 'zero' or 'soft'.")
        with torch.no_grad():
            if init_type == "zero":
                std = float(getattr(self.config, "th_relation_bias_factor_init_std", 0.02))
                nn.init.normal_(self.level_query_bias_factor, mean=0.0, std=std)
                self.level_key_bias_factor.zero_()
                return

            scale = float(getattr(self.config, "th_relation_bias_soft_scale", 0.1))
            levels = torch.arange(self.config.num_behavior + 1, dtype=self.level_query_bias_factor.dtype)
            target = (levels[:, None] - levels[None, :]).clamp(max=0.0) * scale
            u, s, vh = torch.linalg.svd(target.to(torch.float32), full_matrices=False)
            rank = min(self.th_relation_bias_rank, s.numel())
            query_factor = torch.zeros_like(self.level_query_bias_factor)
            key_factor = torch.zeros_like(self.level_key_bias_factor)
            sqrt_s = torch.sqrt(s[:rank]).to(query_factor.dtype)
            query_factor[:, :, :rank] = (u[:, :rank].to(query_factor.dtype) * sqrt_s).unsqueeze(1)
            key_factor[:, :, :rank] = (vh[:rank].T.to(key_factor.dtype) * sqrt_s).unsqueeze(1)
            self.level_query_bias_factor.copy_(query_factor)
            self.level_key_bias_factor.copy_(key_factor)

    def _compute_level_pair_bias(
        self,
        query_action_index: torch.Tensor,
        key_action_index: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        query_action_index = query_action_index.clamp(min=0, max=self.config.num_behavior)
        key_action_index = key_action_index.clamp(min=0, max=self.config.num_behavior)
        if getattr(self, "th_relation_bias_type", "table") == "factorized":
            query_factor = self.level_query_bias_factor[query_action_index]
            key_factor = self.level_key_bias_factor[key_action_index]
            return torch.einsum("blhr,bshr->bhls", query_factor, key_factor).to(dtype)
        pair_bias = self.level_pair_bias[query_action_index[:, :, None], key_action_index[:, None, :]]
        return pair_bias.permute(0, 3, 1, 2).to(dtype)

    @staticmethod
    def _get_multi_view_types(config: Qwen3MoeConfig) -> list[str]:
        view_types = getattr(config, "th_multi_view_types", ["temporal", "same", "up", "down"])
        valid_views = {"temporal": 0, "same": 1, "up": 2, "down": 3}
        for view in view_types:
            if view not in valid_views:
                raise ValueError(f"Unsupported th_multi_view type: {view}")
        return view_types

    @staticmethod
    def _build_multi_view_type_ids(config: Qwen3MoeConfig) -> torch.Tensor:
        valid_views = {"temporal": 0, "same": 1, "up": 2, "down": 3}
        return torch.tensor(
            [valid_views[view] for view in Qwen3TemporalHierarchicalAttention._get_multi_view_types(config)],
            dtype=torch.long,
        )

    @staticmethod
    def _build_multi_view_head_ids(config: Qwen3MoeConfig) -> torch.Tensor:
        view_types = Qwen3TemporalHierarchicalAttention._get_multi_view_types(config)
        allocation = getattr(config, "th_multi_view_head_allocation", None)
        valid_views = {"temporal": 0, "same": 1, "up": 2, "down": 3}
        if allocation is None:
            base = config.num_attention_heads // len(view_types)
            allocation = [base] * len(view_types)
            for i in range(config.num_attention_heads - sum(allocation)):
                allocation[i] += 1
        if len(allocation) != len(view_types):
            raise ValueError("th_multi_view_head_allocation must match th_multi_view_types length.")
        if sum(allocation) != config.num_attention_heads:
            raise ValueError("th_multi_view_head_allocation must sum to num_attention_heads.")
        head_ids = []
        for view, count in zip(view_types, allocation):
            head_ids.extend([valid_views[view]] * int(count))
        return torch.tensor(head_ids, dtype=torch.long)

    def _compute_multi_view_bias(
        self,
        query_action_index: torch.Tensor,
        key_action_index: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        query_level = query_action_index[:, :, None]
        key_level = key_action_index[:, None, :]
        special_pair = (query_level == 0) | (key_level == 0)
        same_block = (query_level != key_level) & ~special_pair
        up_block = (query_level <= key_level) & ~special_pair
        down_block = (query_level >= key_level) & ~special_pair
        block_by_view = torch.stack(
            [
                torch.zeros_like(same_block),
                same_block,
                up_block,
                down_block,
            ],
            dim=1,
        )
        head_block = block_by_view[:, self.multi_view_head_ids.to(query_action_index.device)]
        bias = torch.zeros(
            head_block.shape,
            dtype=dtype,
            device=query_action_index.device,
        )
        return bias.masked_fill(head_block, torch.finfo(dtype).min)

    def _compute_soft_multi_view_bias(
        self,
        query_action_index: torch.Tensor,
        key_action_index: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        query_level = query_action_index[:, :, None]
        key_level = key_action_index[:, None, :]
        special_pair = (query_level == 0) | (key_level == 0)
        same_block = (query_level != key_level) & ~special_pair
        up_block = (query_level <= key_level) & ~special_pair
        down_block = (query_level >= key_level) & ~special_pair
        block_by_view = torch.stack(
            [
                torch.zeros_like(same_block),
                same_block,
                up_block,
                down_block,
            ],
            dim=1,
        )
        view_ids = self.multi_view_type_ids.to(query_action_index.device)
        selected_blocks = block_by_view[:, view_ids].to(dtype)
        if self.th_multi_view_mode == "gated":
            view_weights = torch.softmax(self.multi_view_gate_logits.to(dtype), dim=-1)
        else:
            view_weights = torch.full(
                (self.config.num_attention_heads, len(view_ids)),
                1.0 / len(view_ids),
                dtype=dtype,
                device=query_action_index.device,
            )
        soft_block = torch.einsum("bvls,hv->bhls", selected_blocks, view_weights)
        return -self.th_multi_view_soft_bias_scale * soft_block

    def _apply_relation_bias_scale(self, relation_bias: torch.Tensor) -> torch.Tensor:
        if self.th_relation_bias_learnable_scale:
            return relation_bias * self.th_relation_bias_alpha.to(relation_bias.dtype)
        return relation_bias * self.th_relation_bias_scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        action_index: torch.Tensor | None = None,
        key_action_index: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.is_temporal_hierarchical:
            if action_index is None or key_action_index is None:
                raise ValueError("Temporal-hierarchical attention requires action_index and key_action_index.")
            behavior_embedding_shape = (*input_shape, -1, self.behavior_embedding_dim)
            q_behavior_embedding = self.q_behavior_embedding(action_index).view(behavior_embedding_shape)
            k_behavior_embedding = self.k_behavior_embedding(action_index).view(behavior_embedding_shape)
            v_behavior_embedding = self.v_behavior_embedding(action_index).view(behavior_embedding_shape)
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
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.is_temporal_hierarchical:
            extra_bias = None
            if self.th_attention_mode == "relation_bias":
                extra_bias = self._apply_relation_bias_scale(
                    self._compute_level_pair_bias(action_index, key_action_index, hidden_states.dtype)
                )
            elif self.th_attention_mode == "multi_view":
                if self.th_multi_view_mode == "hard":
                    extra_bias = self._compute_multi_view_bias(action_index, key_action_index, hidden_states.dtype)
                else:
                    extra_bias = self._compute_soft_multi_view_bias(action_index, key_action_index, hidden_states.dtype)
                if self.use_relation_bias:
                    extra_bias = extra_bias + self._apply_relation_bias_scale(
                        self._compute_level_pair_bias(
                            action_index,
                            key_action_index,
                            hidden_states.dtype,
                        )
                    )
            attention_mask = extra_bias if attention_mask is None else attention_mask + extra_bias

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                    "Falling back to eager attention."
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
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if self.is_temporal_hierarchical:
            attn_output = attn_output * self.act_fn(self.gating(hidden_states))
        return attn_output, attn_weights


class Qwen3TemporalHierarchicalDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_idx: int,
        is_sparse: bool,
        behavior_injection: bool,
        is_temporal_hierarchical: bool,
    ):
        super().__init__()
        self.is_temporal_hierarchical = is_temporal_hierarchical
        self.self_attn = Qwen3TemporalHierarchicalAttention(
            config=config,
            layer_idx=layer_idx,
            is_temporal_hierarchical=is_temporal_hierarchical,
        )
        mlp_type = getattr(config, "mlp_type", "PBATransformer")
        if mlp_type == "Qwen3":
            self.mlp = MyQwen3SparseMLP(config, is_sparse=is_sparse, behavior_injection=behavior_injection)
        elif mlp_type == "dense":
            self.mlp = DenseMLP(config)
        elif mlp_type == "RouterMoe":
            self.mlp = RouterMoeBlock(config)
        else:
            self.mlp = PBATransformerSparseMLP(config, is_sparse=is_sparse, behavior_injection=behavior_injection)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_indices: torch.Tensor,
        behavior_indices: torch.Tensor | None = None,
        action_indices: torch.Tensor | None = None,
        key_action_indices: torch.Tensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            action_index=action_indices if self.is_temporal_hierarchical else None,
            key_action_index=key_action_indices if self.is_temporal_hierarchical else None,
            **kwargs,
        )
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, position_indices, behavior_indices)
        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class Qwen3TemporalHierarchicalModel(Qwen3DecoderModelBase):
    decoder_layer_cls = Qwen3TemporalHierarchicalDecoderLayer
    router_cls = Qwen3MultiDecoderRouter

    def _pre_layer_setup(self, config):
        self.temporal_hierarchical_layers = getattr(config, "temporal_hierarchical_attention_decoder", [])

    def _layer_kwargs(self, config, layer_idx):
        kwargs = super()._layer_kwargs(config, layer_idx)
        kwargs["is_temporal_hierarchical"] = (layer_idx in self.temporal_hierarchical_layers)
        return kwargs

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        max_item_num = config.model_max_length // config.num_positions
        block_lower = torch.tril(
            torch.ones(config.num_positions * max_item_num, config.num_positions * max_item_num),
            diagonal=-1,
        )
        block_lower += torch.eye(config.num_positions * max_item_num)
        self.in_item_mask = 1 - block_lower
        self.cached_action_indices = None
        logger.info(
            "Using replacement-style temporal-hierarchical attention in layers: "
            f"{self.temporal_hierarchical_layers}."
        )

    def _update_session_wise_causal_mask(
        self,
        attention_mask: torch.Tensor | None = None,
        input_tensor: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        session_ids: torch.LongTensor | None = None,
        actions: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        mask_ctx = build_mask_context(input_tensor, past_key_values)
        past_seen_tokens = mask_ctx.past_seen_tokens
        batch_size = mask_ctx.batch_size
        sequence_length = mask_ctx.sequence_length
        dtype, device = mask_ctx.dtype, mask_ctx.device
        min_dtype = mask_ctx.min_dtype
        if past_seen_tokens == 0:
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

    def _update_key_action_indices(
        self,
        action_indices: torch.Tensor,
        cache_position: torch.LongTensor,
        past_key_values: Cache | None,
    ) -> torch.Tensor:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if past_seen_tokens == 0 or self.cached_action_indices is None or cache_position.min() == 0:
            self.cached_action_indices = action_indices
        else:
            self.cached_action_indices = torch.cat([self.cached_action_indices, action_indices], dim=1)
        return self.cached_action_indices

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

        position_indices, behavior_indices, action_indices = self.router(input_ids, cache_position=cache_position)
        key_action_indices = self._update_key_action_indices(action_indices, cache_position, past_key_values)

        causal_mask = self._update_session_wise_causal_mask(
            attention_mask=attention_mask,
            input_tensor=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
            session_ids=session_ids,
            actions=actions,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        loop_outputs = run_temporal_hierarchical_decoder_layers(
            self,
            hidden_states=hidden_states,
            position_indices=position_indices,
            behavior_indices=behavior_indices,
            action_indices=action_indices,
            key_action_indices=key_action_indices,
            causal_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
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
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen3TemporalHierarchicalWithTemperature(CustomCausalLMWrapperMixin, Qwen3ForCausalLM):
    def __init__(self, config: Qwen3MoeConfig):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.init_custom_causal_lm(config, Qwen3TemporalHierarchicalModel)

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
