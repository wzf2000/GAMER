import math
import torch
from torch import nn
from typing import Callable
from torch.nn import functional as F

from SeqRec.modules.layers.transformer import MultiHeadAttention, FeedForward


class LinearAttention(nn.Module):
    """
    compute linear attention using projection E and F.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        linear_size: int,
        max_len: int,
    ):
        super(LinearAttention, self).__init__()
        self.E = nn.Linear(max_len, linear_size)
        self.F = nn.Linear(max_len, linear_size)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_Q = nn.Linear(embed_dim, embed_dim)

        self.dense = nn.Linear(embed_dim, embed_dim)
        self.n_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.attn_dropout = nn.Dropout(p=dropout)
        self.out_dropout = nn.Dropout(p=dropout)
        self.LayerNorm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.n_heads, self.d_k)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        key = self.W_K(input_tensor)
        value = self.W_V(input_tensor)
        query = self.W_Q(input_tensor)

        key = self.transpose_for_scores(key)
        query = self.transpose_for_scores(query)
        value = self.transpose_for_scores(value)

        # b, num_heads, l, d/num_heads
        if mask is not None:
            mask = mask[:, 0:1, :].unsqueeze(-1)
            # b, 1, l, 1
            key = key * mask
            value = value * mask

        value = self.E(value.transpose(2, 3)).transpose(2, 3)
        key = self.F(key.transpose(2, 3)).transpose(2, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)) * math.sqrt(1.0 / float(query.size(-1)))

        p_attn = F.softmax(scores, dim=-1)

        p_attn = self.attn_dropout(p_attn)

        context_layer = torch.matmul(p_attn, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.d_k * self.n_heads,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states, p_attn


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        scales: list[int],
        max_len: int,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads
        self.scale_1 = scales[1]
        self.scale_2 = scales[2]
        self.max_len = max_len
        self.out_fc = nn.Linear(self.max_len + self.max_len // self.scale_1 + self.max_len // self.scale_2, self.max_len)

        self.attention1 = LinearAttention(embed_dim, num_heads, dropout, layer_norm_eps, scales[0], self.max_len)
        self.attention2 = MultiHeadAttention(embed_dim, num_heads, dropout, layer_norm_eps)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = input_tensor.size(0)
        seq_length = input_tensor.size(1)

        # 2) multi scale attention

        # linear attention over whole sequence
        # b, num_heads, seq_length, dim//num_heads
        x, linear_attn_weight = self.attention1(input_tensor, attention_mask)
        scale_outputs = []
        scale_outputs.append(torch.reshape(x, [batch_size, seq_length, self.num_heads * self.d_k]))
        next_input = torch.mean(
            input_tensor.reshape(batch_size, self.scale_1, seq_length // self.scale_1, self.num_heads * self.d_k),
            dim=1,
        )

        # attention over 1/scale_1 sequence
        x = self.attention2(next_input, None)
        scale_outputs.append(
            torch.reshape(
                x, [batch_size, seq_length // self.scale_1, self.num_heads * self.d_k]
            )
        )
        next_input = torch.mean(
            input_tensor.reshape(
                batch_size, self.scale_2, seq_length // self.scale_2, self.num_heads * self.d_k
            ),
            dim=1,
        )

        # attention over 1/scale_2 sequence
        x = self.attention2(next_input, None)
        scale_outputs.append(
            torch.reshape(
                x, [batch_size, seq_length // self.scale_2, self.num_heads * self.d_k]
            )
        )

        output = torch.cat(scale_outputs, dim=1)
        output = torch.transpose(output, 1, 2)
        output = self.out_fc(output)
        output = torch.transpose(output, 1, 2)

        return output


class MultiScaleTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[torch.Tensor], torch.Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        multiscale: bool = False,
        scales: list[int] = [],
        max_len: int = -1,
    ) -> None:
        super().__init__()
        if multiscale:
            assert max_len > 0, "max_len must be provided when using multiscale attention"
            self.multi_head_attention = MultiScaleAttention(
                d_model, nhead, dropout, layer_norm_eps, scales, max_len
            )
        else:
            self.multi_head_attention = MultiHeadAttention(
                d_model, nhead, dropout, layer_norm_eps
            )
        self.feed_forward = FeedForward(
            d_model, dim_feedforward, dropout, activation, layer_norm_eps
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_output = self.multi_head_attention(
            hidden_states, attention_mask
        )
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output
