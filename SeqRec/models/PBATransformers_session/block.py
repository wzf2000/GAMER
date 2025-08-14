import torch
import torch.nn as nn
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersLayerSelfAttention,
    SwitchTransformersLayerCrossAttention,
)

from SeqRec.models.PBATransformers_session.configuration import PBATransformerConfigSession
from SeqRec.models.PBATransformers_session.FFN import PBATransformersLayerFFSession


class PBATransformersBlockSession(nn.Module):
    def __init__(
        self,
        config: PBATransformerConfigSession,
        has_relative_attention_bias: bool = False,
        is_sparse: bool = False,
        layer_idx: int | None = None,
        behavior_injection: bool = False,
        session_injection: bool = False,
        time_injection: bool = False,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.is_sparse = is_sparse
        self.layer = nn.ModuleList()
        self.layer.append(
            SwitchTransformersLayerSelfAttention(
                config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx
            )
        )
        if self.is_decoder:
            self.layer.append(SwitchTransformersLayerCrossAttention(config, layer_idx=layer_idx))

        self.layer.append(
            PBATransformersLayerFFSession(
                config, is_sparse=self.is_sparse, behavior_injection=behavior_injection, session_injection=session_injection,
            )
        )

        self.time_embedding = time_injection
        if self.time_embedding:
            self.freq = torch.arange(0, config.freqnum) * 2 * torch.pi / config.freqnum
            self.time_mlp = nn.Sequential(nn.Linear(config.freqnum, config.d_model), nn.ReLU())

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_indices: torch.Tensor,
        behavior_indices: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_decoder_position_bias: torch.Tensor | None = None,
        layer_head_mask: torch.Tensor | None = None,
        cross_attn_layer_head_mask: torch.Tensor | None = None,
        past_key_value: tuple[tuple[torch.FloatTensor, ...], ...] | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_router_logits: bool = True,
        return_dict: bool = True,
        cache_position: torch.Tensor | None = None,
        session_indices: torch.Tensor = None,
        times: torch.Tensor = None,
    ):
        batch_size, seq_length, _ = hidden_states.size()
        if self.time_embedding:
            freq_matrix = self.freq[None, None, :].repeat(batch_size, 1, 1).to(hidden_states.device)
            times = times[..., None]
            mask = times < 0
            times_position = torch.cos(torch.matmul(times, freq_matrix))
            times_position_embedding = self.time_mlp(times_position)
            times_position_embedding = times_position_embedding * (1 - mask.float())
            hidden_states = hidden_states + times_position_embedding
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states, past_key_value = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[
            2:
        ]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                query_length=cache_position[-1] + 1,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )
            hidden_states, past_key_value = cross_attention_outputs[:2]

            # clamp inf values to enable fp16 training
            if (
                hidden_states.dtype == torch.float16
                and torch.isinf(hidden_states).any()
            ):
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(
                    hidden_states, min=-clamp_value, max=clamp_value
                )

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](
            hidden_states, position_indices, behavior_indices, session_indices
        )

        if isinstance(hidden_states, tuple):
            hidden_states, router_tuple = hidden_states
        else:
            router_tuple = (
                torch.zeros((1,), device=hidden_states.device, dtype=torch.int64),
            )

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        if use_cache:
            outputs = (
                outputs
                + (past_key_value,)
                + attention_outputs
                + (router_tuple,)
            )
        else:
            outputs = outputs + attention_outputs + (router_tuple,)

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights), (router_tuple)
