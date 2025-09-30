import copy
import math
import torch
from torch import nn
from typing import Callable, TypeVar
from torch.nn import functional as F


T = TypeVar('T')


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (embed_dim, num_heads)
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.attn_dropout = nn.Dropout(dropout)

        self.dense = nn.Linear(embed_dim, embed_dim)
        self.LayerNorm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores * math.sqrt(1.0 / float(self.attention_head_size))
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        activation: str | Callable[[torch.Tensor], torch.Tensor],
        layer_norm_eps: float,
        residual: bool = True,
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(d_model, dim_feedforward)
        self.intermediate_act_fn = self.get_hidden_act(activation) if isinstance(activation, str) else activation

        self.dense_2 = nn.Linear(dim_feedforward, d_model)
        self.residual = residual
        if self.residual:
            self.LayerNorm = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.dropout = nn.Dropout(dropout)

    def get_hidden_act(self, act: str) -> Callable[[torch.Tensor], torch.Tensor]:
        ACT2FN = {
            "gelu": F.gelu,
            "relu": F.relu,
            "swish": F.silu,
            "tanh": F.tanh,
            "sigmoid": F.sigmoid,
            "elu": F.elu,
        }
        return ACT2FN[act]

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        if not self.residual:
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[torch.Tensor], torch.Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
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


class TransformerEncoder(nn.Module):
    r"""One TransformerEncoder consists of several TransformerLayers.
        - Args:
            - encoder_layer: an instance of the TransformerEncoderLayer() class (required).
            - num_layers: the number of sub-encoder-layers in the encoder (required).
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward(self, hidden_states: T, attention_mask: torch.Tensor, **kwargs) -> T:
        """
        Args:
            hidden_states: the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, **kwargs)
        return hidden_states


class DotProductPredictionHead(nn.Module):
    """share embedding parameters"""

    def __init__(
        self,
        d_model: int,
        n_items: int,
        token_embeddings: nn.Embedding,
    ):
        super().__init__()
        self.token_embeddings = token_embeddings
        self.vocab_size = n_items + 1
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.bias = nn.Parameter(torch.zeros(1, self.vocab_size))

    def forward(self, hidden_states: torch.Tensor, candidates: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        hidden_states = self.out(hidden_states)  # [B, H] or [M, H]
        if candidates is not None:  # [B, H]
            emb: torch.Tensor = self.token_embeddings(candidates)  # [B, C, H]
            logits = (hidden_states.unsqueeze(1) * emb).sum(-1)  # [B, C]
            bias = self.bias.expand(logits.size(0), -1).gather(1, candidates)  # [B, C]
            logits += bias  # [B, C]
        else:  # [M, H]
            emb = self.token_embeddings.weight[:self.vocab_size]  # [n_items + 1, H]
            logits = torch.matmul(hidden_states, emb.transpose(0, 1))  # [M, n_items + 1]
            logits += self.bias  # [M, n_items + 1]
        return logits
