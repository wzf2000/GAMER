from typing import Any

import torch
from torch import nn
from transformers.activations import ACT2FN


class CrossBehaviorAttentionMixin:
    def init_cross_behavior_attention(self, config: Any):
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
        self.gating = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def get_cross_behavior_embeddings(
        self,
        hidden_states: torch.Tensor,
        action_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        behavior_embedding_shape = (*input_shape, -1, self.behavior_embedding_dim)
        q_behavior_embedding = self.q_behavior_embedding(action_index).view(behavior_embedding_shape)
        k_behavior_embedding = self.k_behavior_embedding(action_index).view(behavior_embedding_shape)
        v_behavior_embedding = self.v_behavior_embedding(action_index).view(behavior_embedding_shape)
        return q_behavior_embedding, k_behavior_embedding, v_behavior_embedding

    def apply_cross_behavior_gate(self, attn_output: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return attn_output * self.act_fn(self.gating(hidden_states))


def run_multi_level_self_attention_block(
    decoder_layer: nn.Module,
    *,
    hidden_states: torch.Tensor,
    multi_self_mask: torch.Tensor | None,
    position_ids: torch.LongTensor | None,
    past_key_value: Any,
    output_attentions: bool,
    use_cache: bool,
    cache_position: torch.LongTensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    residual = hidden_states
    hidden_states = decoder_layer.input_layernorm(hidden_states)
    hidden_states, self_attn_weights = decoder_layer.self_attn(
        hidden_states=hidden_states,
        attention_mask=multi_self_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + decoder_layer.dropout(hidden_states)
    return hidden_states, self_attn_weights


def run_multi_level_cross_attention_block(
    decoder_layer: nn.Module,
    *,
    hidden_states: torch.Tensor,
    action_indices: torch.Tensor | None,
    multi_cross_mask: torch.Tensor | None,
    position_ids: torch.LongTensor | None,
    cross_past_key_value: Any,
    output_attentions: bool,
    use_cache: bool,
    cache_position: torch.LongTensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    if not decoder_layer.is_cross:
        return hidden_states

    residual = hidden_states
    hidden_states = decoder_layer.post_self_attention_layernorm(hidden_states)
    hidden_states, _ = decoder_layer.cross_attn(
        hidden_states=hidden_states,
        attention_mask=multi_cross_mask,
        position_ids=position_ids,
        past_key_value=cross_past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        action_index=action_indices,
        **kwargs,
    )
    return residual + decoder_layer.dropout(hidden_states)
