import math
import torch
from torch import nn
from typing import Callable
from torch.nn import functional as F

from SeqRec.modules.layers.transformer import FeedForward


class RelativePositionBias(nn.Module):
    def __init__(
        self,
        num_buckets: int = 32,
        max_distance: int = 128,
        n_heads: int = 2
    ):
        super(RelativePositionBias, self).__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(self.num_buckets, n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position: torch.Tensor, num_buckets: int = 32, max_distance: int = 128) -> torch.Tensor:
        ret = 0
        n = -relative_position
        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)

        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qlen: int, klen: int) -> torch.Tensor:
        """ Compute binned relative position bias """
        device = self.relative_attention_bias.weight.device
        q_pos = torch.arange(qlen, dtype=torch.long, device=device)
        k_pos = torch.arange(klen, dtype=torch.long, device=device)
        relative_position = k_pos[None, :] - q_pos[:, None]
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values: torch.Tensor = self.relative_attention_bias(
            rp_bucket
        )  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, qlen, klen)
        return values


class MBSMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        n_behaviors: int,
        behavior_attention: bool,
        behavior_position_bias: bool,
        num_buckets: int = 32,
        max_distance: int = 40,
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
        self.n_behaviors = n_behaviors
        self.behavior_attention = behavior_attention
        self.behavior_position_bias = behavior_position_bias

        if self.behavior_attention and self.n_behaviors > 1:
            self.W1 = nn.Parameter(
                torch.randn(self.n_behaviors, self.num_attention_heads, self.attention_head_size, self.attention_head_size)
            )
            self.alpha1 = nn.Parameter(
                torch.randn(self.n_behaviors * self.n_behaviors + 1, self.n_behaviors, self.num_attention_heads)
            )
            self.W2 = nn.Parameter(
                torch.randn(self.n_behaviors, self.num_attention_heads, self.attention_head_size, self.attention_head_size)
            )
            self.alpha2 = nn.Parameter(
                torch.randn(self.n_behaviors * self.n_behaviors + 1, self.n_behaviors, self.num_attention_heads)
            )
            self.query = nn.Parameter(
                torch.randn(self.n_behaviors + 1, embed_dim, self.num_attention_heads, self.attention_head_size)
            )
            self.key = nn.Parameter(
                torch.randn(self.n_behaviors + 1, embed_dim, self.num_attention_heads, self.attention_head_size)
            )
            self.value = nn.Parameter(
                torch.randn(self.n_behaviors + 1, embed_dim, self.num_attention_heads, self.attention_head_size)
            )
        else:
            self.query = nn.Linear(embed_dim, self.all_head_size)
            self.key = nn.Linear(embed_dim, self.all_head_size)
            self.value = nn.Linear(embed_dim, self.all_head_size)

        self.attn_dropout = nn.Dropout(dropout)
        if self.behavior_position_bias:
            self.relative_position_bias = nn.ModuleList([
                RelativePositionBias(
                    num_buckets=num_buckets, max_distance=max_distance, n_heads=num_heads
                ) for _ in range(self.n_behaviors * self.n_behaviors + 1)
            ])

        self.LayerNorm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor, type_seq: torch.Tensor) -> torch.Tensor:
        # input_tensor: [B, L, H], type_seq: [B, L]
        bs, seq_len = input_tensor.size()[:2]
        behavior_meatrix = (type_seq[:, :, None] * type_seq[:, None, :] != 0).long()  # [B, L, L]
        behavior_meatrix *= ((type_seq[:, :, None] - 1) * self.n_behaviors + type_seq[:, None, :])  # [B, L, L]

        if self.behavior_position_bias:
            relative_position_bias = torch.stack([
                layer(seq_len, seq_len) for layer in self.relative_position_bias
            ], dim=-1).repeat(bs, 1, 1, 1, 1)  # [B, h, L, L, b * b + 1]
            relative_position_bias = relative_position_bias.gather(
                dim=4,
                index=behavior_meatrix[:, None, :, :, None].repeat(1, self.num_attention_heads, 1, 1, 1),  # [B, h, L, L, 1]
            )[..., 0]  # [B, h, L, L]
        else:
            relative_position_bias = 0

        if self.behavior_attention and self.n_behaviors > 1:
            query_layer = torch.einsum(
                "BLH, bHhd, BLb -> BhLd",
                input_tensor,  # [B, L, H]
                self.query,  # [b + 1, H, h, d]
                F.one_hot(type_seq, num_classes=self.n_behaviors + 1).float(),  # [B, L, b + 1]
            )  # [B, h, L, d]
            key_layer = torch.einsum(
                "BLH, bHhd, BLb -> BhLd",
                input_tensor,  # [B, L, H]
                self.key,  # [b + 1, H, h, d]
                F.one_hot(type_seq, num_classes=self.n_behaviors + 1).float(),  # [B, L, b + 1]
            )  # [B, h, L, d]
            value_layer = torch.einsum(
                "BLH, bHhd, BLb -> BhLd",
                input_tensor,  # [B, L, H]
                self.value,  # [b + 1, H, h, d]
                F.one_hot(type_seq, num_classes=self.n_behaviors + 1).float(),  # [B, L, b + 1]
            )  # [B, h, L, d]
        else:
            mixed_query_layer = self.query(input_tensor)
            mixed_key_layer = self.key(input_tensor)
            mixed_value_layer = self.value(input_tensor)
            query_layer = self.transpose_for_scores(mixed_query_layer)  # [B, h, L, d]
            key_layer = self.transpose_for_scores(mixed_key_layer)  # [B, h, L, d]
            value_layer = self.transpose_for_scores(mixed_value_layer)  # [B, h, L, d]
            behavior_meatrix = None

        if behavior_meatrix is None:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(
                query_layer, key_layer.transpose(-1, -2)
            )  # [B, h, L, L]
        else:
            W1_ = torch.einsum(
                "bhmn, Cbh -> Chmn",
                self.W1,  # [b, h, d, d]
                F.softmax(self.alpha1, dim=1)  # [b * b + 1, b, h]
            )  # [b * b + 1, h, d, d]
            attention_all = torch.einsum(
                "BhQm, Chmn, BhKn -> BhQKC",
                query_layer,  # [B, h, L, d]
                W1_,  # [b * b + 1, h, d, d]
                key_layer,  # [B, h, L, d]
            )  # [B, h, L, L, b * b + 1]
            attention_scores = attention_all.gather(
                dim=4,
                index=behavior_meatrix[:, None, :, :, None].repeat(1, self.num_attention_heads, 1, 1, 1),  # [B, h, L, L, 1]
            )[..., 0]  # [B, h, L, L]

        attention_scores = attention_scores * math.sqrt(1.0 / float(self.attention_head_size)) + relative_position_bias  # [B, h, L, L]
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
        if behavior_meatrix is None:
            context_layer = torch.matmul(attention_probs, value_layer)
        else:
            one_hot_behavior_matrix = F.one_hot(
                behavior_meatrix[:, None, :, :], num_classes=self.n_behaviors * self.n_behaviors + 1
            ).repeat(1, self.num_attention_heads, 1, 1, 1)  # [B, h, L, L, b * b + 1]
            W2_ = torch.einsum(
                "bhmn, Cbh -> Chmn",
                self.W2,  # [b, h, d, d]
                F.softmax(self.alpha2, dim=1)  # [b * b + 1, b, h]
            )  # [b * b + 1, h, d, d]
            context_layer = torch.einsum(
                "BhQK, BhQKC, Chnm, BhKn -> BhQm",
                attention_probs,  # [B, h, L, L]
                one_hot_behavior_matrix,  # [B, h, L, L, b * b + 1]
                W2_,  # [b * b + 1, h, d, d]
                value_layer,  # [B, h, L, d]
            )  # [B, h, L, d]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [B, L, h, d]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # [B, L, H]
        hidden_states = self.out_dropout(context_layer)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class MBSFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        activation: str | Callable[[torch.Tensor], torch.Tensor],
        layer_norm_eps: float,
        n_behaviors: int,
        behavior_moe: bool,
    ):
        super(MBSFeedForward, self).__init__()
        self.n_behaviors = n_behaviors
        self.behavior_moe = behavior_moe
        if self.behavior_moe and self.n_behaviors > 1:
            self.FFN = nn.ModuleList([
                FeedForward(
                    d_model, dim_feedforward, dropout, activation, layer_norm_eps
                ) for _ in range(self.n_behaviors)
            ])
            self.LayerNorm = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.dropout = nn.Dropout(dropout)
        else:
            self.FFN = FeedForward(
                d_model, dim_feedforward, dropout, activation, layer_norm_eps, residual=False
            )

    def forward(self, input_tensor: torch.Tensor, type_seq: torch.Tensor) -> torch.Tensor:
        if self.behavior_moe and self.n_behaviors > 1:
            outputs = [torch.zeros_like(input_tensor)]
            for i in range(self.n_behaviors):
                outputs.append(self.FFN[i](input_tensor))
            hidden_states = torch.einsum(
                "bBLH, BLb -> BLH",
                torch.stack(outputs, dim=0),  # [b + 1, B, L, H]
                F.one_hot(type_seq, num_classes=self.n_behaviors + 1).float()  # [B, L, b + 1]
            )  # [B, L, H]
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = self.FFN(input_tensor)
        return hidden_states


class MBSTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        n_behaviors: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[torch.Tensor], torch.Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        num_buckets: int = 32,
        max_distance: int = 40,
        behavior_attention: bool = True,
        behavior_moe: bool = True,
        behavior_position_bias: bool = True,
    ) -> None:
        super().__init__()
        self.multi_head_attention = MBSMultiHeadAttention(
            d_model, nhead, dropout, layer_norm_eps,
            n_behaviors, behavior_attention, behavior_position_bias,
            num_buckets, max_distance,
        )
        self.feed_forward = MBSFeedForward(
            d_model, dim_feedforward, dropout, activation, layer_norm_eps,
            n_behaviors, behavior_moe,
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, type_seq: torch.Tensor) -> torch.Tensor:
        attention_output = self.multi_head_attention(
            hidden_states, attention_mask, type_seq
        )
        feedforward_output = self.feed_forward(attention_output, type_seq)
        return feedforward_output


class CGCDotProductPredictionHead(nn.Module):
    """
    model with shared expert and behavior specific expert
    3 shared expert,
    1 specific expert per behavior.
    """

    def __init__(
        self,
        d_model: int,
        n_items: int,
        token_embeddings: nn.Embedding,
        layer_norm_eps: float,
        n_behaviors: int,
        n_shared_experts: int,
        n_specific_experts: int,
    ):
        super().__init__()
        self.n_behaviors = n_behaviors
        self.n_shared_experts = n_shared_experts
        self.n_specific_experts = n_specific_experts
        self.vocab_size = n_items + 1
        self.softmax = nn.Softmax(dim=-1)
        self.shared_experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, d_model)) for _ in range(self.n_shared_experts)]
        )
        self.specific_experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, d_model)) for _ in range(self.n_behaviors * self.n_specific_experts)]
        )
        self.w_gates = nn.Parameter(
            torch.randn(
                self.n_behaviors, d_model, self.n_shared_experts + self.n_specific_experts
            ),
            requires_grad=True,
        )
        self.token_embeddings = token_embeddings
        self.ln = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, type_seq: torch.Tensor, candidates: torch.Tensor | None = None) -> torch.Tensor:
        # hidden_states: [B, H] or [M, H], type_seq: [B] or [M]
        hidden_states = self.mmoe_process(hidden_states, type_seq)  # [B, H] or [M, H]
        if candidates is not None:  # [B, H]
            emb: torch.Tensor = self.token_embeddings(candidates)  # [B, C, H]
            logits = (hidden_states.unsqueeze(1) * emb).sum(-1)  # [B, C]
        else:  # [M, H]
            emb = self.token_embeddings.weight[:self.vocab_size]  # [n_items + 1, H]
            logits = torch.matmul(hidden_states, emb.transpose(0, 1))  # [M, n_items + 1]
        return logits

    def mmoe_process(self, hidden_states: torch.Tensor, type_seq: torch.Tensor) -> torch.Tensor:
        shared_experts_o = [e(hidden_states) for e in self.shared_experts]  # list of [B, H] or [M, H], length = n_shared_experts
        specific_experts_o = [e(hidden_states) for e in self.specific_experts]  # list of [B, H] or [M, H], length = n_behaviors * n_specific_experts
        gates_o = self.softmax(torch.einsum(
            'BH, bHE -> bBE',
            hidden_states,  # [B, H] or [M, H]
            self.w_gates,  # [b, H, n_experts]
        ))  # [b, B, n_experts] or [b, M, n_experts]
        # rearrange
        experts_o_tensor = torch.stack([
            torch.stack(
                shared_experts_o + specific_experts_o[
                    i * self.n_specific_experts: (i + 1) * self.n_specific_experts
                ]  # [n_experts, B, H] or [n_experts, M, H]
            ) for i in range(self.n_behaviors)
        ])  # [b, n_experts, B, H] or [b, n_experts, M, H]
        output = torch.einsum(
            'bEBH,bBE->bBH',
            experts_o_tensor,  # [b, n_experts, B, H] or [b, n_experts, M, H]
            gates_o,  # [b, B, n_experts] or [b, M, n_experts]
        )  # [b, B, H] or [b, M, H]
        outputs = torch.cat([
            torch.zeros_like(hidden_states).unsqueeze(0),  # [1, B, H] or [1, M, H]
            output,  # [b, B, H] or [b, M, H]
        ])  # [b + 1, B, H] or [b + 1, M, H]
        return hidden_states + self.ln(torch.einsum(
            'bBH, Bb -> BH',
            outputs,  # [b + 1, B, H] or [b + 1, M, H]
            F.one_hot(type_seq, num_classes=self.n_behaviors + 1).float(),  # [B, b + 1] or [M, b + 1]
        ))  # [B, H] or [M, H]
