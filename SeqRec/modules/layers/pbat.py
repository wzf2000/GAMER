import math
import torch
from torch import nn
from typing import Callable
from torch.nn import functional as F

from SeqRec.modules.layers.transformer import FeedForward


def SAGP(mean1: torch.Tensor, mean2: torch.Tensor, cov1: torch.Tensor, cov2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Self-Adaptive Gaussian Production"""

    cov1 = torch.clamp(cov1, min=1e-24)
    cov2 = torch.clamp(cov2, min=1e-24)
    mean = (cov1 * mean2 + cov2 * mean1) / (cov1 + cov2)
    cov = 2 * (cov1 * cov2) / (cov1 + cov2)
    return mean, cov


def TriSAGP(mean1: torch.Tensor, mean2: torch.Tensor, mean3: torch.Tensor, cov1: torch.Tensor, cov2: torch.Tensor, cov3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Tri-Self-Adaptive Gaussian Production"""

    cov1 = torch.clamp(cov1, min=1e-24)
    cov2 = torch.clamp(cov2, min=1e-24)
    cov3 = torch.clamp(cov3, min=1e-24)
    cov = 1. / (1. / (cov1) + 1. / (cov2) + 1. / (cov3))
    mean = cov * (mean1 / (cov1) + mean2 / (cov2) + mean3 / (cov3))
    return mean, cov


def wasserstein_distance_matmul(mean1: torch.Tensor, cov1: torch.Tensor, mean2: torch.Tensor, cov2: torch.Tensor) -> torch.Tensor:
    # mean1 & cov1: [*, C1, H]
    # mean2 & cov2: [*, C2, H]
    mean1_2 = torch.sum(mean1 ** 2, -1, keepdim=True)  # [*, C1, 1]
    mean2_2 = torch.sum(mean2 ** 2, -1, keepdim=True)  # [*, C2, 1]
    ret = -2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2)  # [*, C1, C2]

    cov1_2 = torch.sum(cov1, -1, keepdim=True)  # [*, C1, 1]
    cov2_2 = torch.sum(cov2, -1, keepdim=True)  # [*, C2, 1]
    cov_ret = -2 * torch.matmul(  # [*, C1, C2]
        torch.sqrt(torch.clamp(cov1, min=1e-24)),  # [*, C1, H]
        torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)  # [*, H, C2]
    ) + cov1_2 + cov2_2.transpose(-1, -2)  # [*, C1, C2]
    return (ret + cov_ret).squeeze()  # [*, C1, C2] or [*, C1] or [*, C2] or [*] (depends on input shape)


class SimpleEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, dropout: float, layer_norm_eps: float = 1e-12, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.activation = nn.ELU()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(seq)
        emb = self.LayerNorm(emb)
        emb = self.dropout(emb)
        emb = self.activation(emb)
        return emb


class FBAMultiHeadAttention(nn.Module):
    """Fused Behavior-Aware Multi-Head Attention"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        n_behaviors: int,
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

        def get_qkv_linear() -> nn.ModuleDict:
            return nn.ModuleDict({
                'q': nn.Linear(embed_dim, self.all_head_size),
                'k': nn.Linear(embed_dim, self.all_head_size),
                'v': nn.Linear(embed_dim, self.all_head_size),
            })

        self.xm = get_qkv_linear()
        self.xc = get_qkv_linear()
        self.bm = get_qkv_linear()
        self.bc = get_qkv_linear()

        self.attn_dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

        self.mean_dense = nn.Linear(embed_dim, embed_dim)
        self.cov_dense = nn.Linear(embed_dim, embed_dim)

        self.LayerNorm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(dropout)

        self.Wq1 = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.Wq2 = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.Wk1 = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.Wk2 = nn.Linear(self.attention_head_size, self.attention_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        if len(new_x_shape) == 4:
            return x.permute(0, 2, 1, 3)
        elif len(new_x_shape) == 5:
            return x.permute(0, 3, 1, 2, 4)
        else:
            raise ValueError(f"Wrong input shape for transpose_for_scores: {new_x_shape}")

    def apply_qkv(self, x: torch.Tensor, qkv_dict: nn.ModuleDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed_query_layer = qkv_dict['q'](x)
        mixed_key_layer = qkv_dict['k'](x)
        mixed_value_layer = qkv_dict['v'](x)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        return query_layer, key_layer, value_layer

    def forward(self, input_tensor: tuple[torch.Tensor, torch.Tensor], attention_mask: torch.Tensor, type_seq: torch.Tensor, type_tensor: tuple[torch.Tensor, torch.Tensor], type_relation_tensor: tuple[torch.Tensor, torch.Tensor], position_tensor: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # *_tensor: [B, *, H] * 2, type_seq: [B, L]
        # input & type & position: [L], type_relation: [b + 1, b + 1]
        bs, seq_len = input_tensor[0].size()[:2]

        query_xm, key_xm, value_xm = self.apply_qkv(input_tensor[0], self.xm)  # [B, h, L, d] * 3
        query_xc, key_xc, value_xc = self.apply_qkv(input_tensor[1], self.xc)  # [B, h, L, d] * 3
        query_bm, key_bm, value_bm = self.apply_qkv(type_tensor[0], self.bm)  # [B, h, L, d] * 3
        query_bc, key_bc, value_bc = self.apply_qkv(type_tensor[1], self.bc)  # [B, h, L, d] * 3

        query_1 = query_xm + query_bm  # [B, h, L, d]
        key_1 = key_xm + key_bm  # [B, h, L, d]
        value_1 = value_xm + value_bm  # [B, h, L, d]
        query_2 = self.activation(query_xc + query_bc) + 1  # [B, h, L, d]
        key_2 = self.activation(key_xc + key_bc) + 1  # [B, h, L, d]
        value_2 = self.activation(value_xc + value_bc) + 1  # [B, h, L, d]

        type_relation_m = self.transpose_for_scores(type_relation_tensor[0]).contiguous()  # [B, h, b + 1, b + 1, d]
        type_relation_c = self.transpose_for_scores(type_relation_tensor[1]).contiguous()  # [B, h, b + 1, b + 1, d]
        position_m = self.transpose_for_scores(position_tensor[0]).contiguous()  # [B, h, L, d]
        position_c = self.transpose_for_scores(position_tensor[1]).contiguous()  # [B, h, L, d]

        type_relation_m_batch = type_relation_m[
            torch.arange(bs)[:, None, None],
            :,
            type_seq[torch.arange(bs)][:, None],
            type_seq[torch.arange(bs)][:, :, None],
            :
        ].permute(0, 3, 2, 1, 4).contiguous()  # [B, h, L, L, d]
        type_relation_c_batch = type_relation_c[
            torch.arange(bs)[:, None, None],
            :,
            type_seq[torch.arange(bs)][:, None],
            type_seq[torch.arange(bs)][:, :, None],
            :
        ].permute(0, 3, 2, 1, 4).contiguous()  # [B, h, L, L, d]

        fusion_Q_m, fusion_Q_c = TriSAGP(
            query_1[:, :, :, None, :],  # [B, h, L, 1, d]
            self.Wq1(type_relation_m_batch),  # [B, h, L, L, d]
            self.Wq2(position_m)[:, :, :, None, :],  # [B, h, L, 1, d]
            query_2[:, :, :, None, :],  # [B, h, L, 1, d]
            type_relation_c_batch,  # [B, h, L, L, d]
            position_c[:, :, :, None, :],  # [B, h, L, 1, d]
        )  # [B, h, L, L, d] * 2
        fusion_K_m, fusion_K_c = TriSAGP(
            key_1[:, :, :, None, :],  # [B, h, L, 1, d]
            self.Wk1(type_relation_m_batch),  # [B, h, L, L, d]
            self.Wk2(position_m)[:, :, :, None, :],  # [B, h, L, 1, d]
            key_2[:, :, :, None, :],  # [B, h, L, 1, d]
            type_relation_c_batch,  # [B, h, L, L, d]
            position_c[:, :, :, None, :],  # [B, h, L, 1, d]
        )  # [B, h, L, L, d] * 2

        Wass_scores = -wasserstein_distance_matmul(
            fusion_Q_m[:, :, :, :, None, :],  # [B, h, L, L, 1, d]
            fusion_Q_c[:, :, :, :, None, :],  # [B, h, L, L, 1, d]
            fusion_K_m[:, :, :, :, None, :],  # [B, h, L, L, 1, d]
            fusion_K_c[:, :, :, :, None, :],  # [B, h, L, L, 1, d]
        )  # [B, h, L, L]

        Wass_scores = Wass_scores * math.sqrt(1.0 / float(self.attention_head_size))  # [B, h, L, L]
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        if attention_mask is not None:
            Wass_scores = Wass_scores + attention_mask

        # Normalize the attention scores to probabilities.
        Wass_probs = nn.Softmax(dim=-1)(Wass_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        Wass_probs = self.attn_dropout(Wass_probs)  # [B, h, L, L]
        mean_context_layer = torch.matmul(Wass_probs, value_1)  # [B, h, L, d]
        cov_context_layer = torch.matmul(Wass_probs, value_2)  # [B, h, L, d]

        mean_context_layer = mean_context_layer.permute(0, 2, 1, 3).contiguous()  # [B, L, h, d]
        cov_context_layer = cov_context_layer.permute(0, 2, 1, 3).contiguous()  # [B, L, h, d]

        new_context_layer_shape = mean_context_layer.size()[:-2] + (self.all_head_size,)
        mean_context_layer = mean_context_layer.view(*new_context_layer_shape)  # [B, L, H]
        cov_context_layer = cov_context_layer.view(*new_context_layer_shape)  # [B, L, H]

        hidden_states_m = self.mean_dense(mean_context_layer)  # [B, L, H]
        hidden_states_m = self.out_dropout(hidden_states_m)
        hidden_states_m = self.LayerNorm(hidden_states_m + input_tensor[0])

        hidden_states_c = self.cov_dense(cov_context_layer)  # [B, L, H]
        hidden_states_c = self.out_dropout(hidden_states_c)
        hidden_states_c = self.LayerNorm(hidden_states_c + input_tensor[1])

        return hidden_states_m, hidden_states_c, Wass_probs  # [B, L, H] * 2, [B, h, L, L]


class BehaviorSpecificFeedForward(nn.Module):
    """Behavior Specific Feed Forward Network (MoE)"""
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        activation: str | Callable[[torch.Tensor], torch.Tensor],
        layer_norm_eps: float,
        n_behaviors: int,
    ):
        super().__init__()
        self.n_behaviors = n_behaviors
        self.FFN = nn.ModuleList(
            [FeedForward(d_model, dim_feedforward, dropout, activation, layer_norm_eps) for _ in range(n_behaviors)]
        )

    def forward(self, input_tensor: torch.Tensor, type_seq: torch.Tensor) -> torch.Tensor:
        outputs = [torch.zeros_like(input_tensor)]
        for i in range(self.n_behaviors):
            outputs.append(self.FFN[i](input_tensor))
        hidden_states = torch.einsum(
            "bBLH, BLb -> BLH",
            torch.stack(outputs, dim=0),  # [b + 1, B, L, H]
            F.one_hot(type_seq, num_classes=self.n_behaviors + 1).float()  # [B, L, b + 1]
        )  # [B, L, H]
        return hidden_states


class PBATLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        n_behaviors: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[torch.Tensor], torch.Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.multi_head_attention = FBAMultiHeadAttention(
            d_model, nhead, dropout, layer_norm_eps,
            n_behaviors,
        )
        self.feed_forward = BehaviorSpecificFeedForward(
            d_model, dim_feedforward, dropout, activation, layer_norm_eps,
            n_behaviors,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.activation_func = nn.ELU()

    def forward(self, hidden_states: tuple[torch.Tensor, torch.Tensor, torch.Tensor], attention_mask: torch.Tensor, type_seq: torch.Tensor, type_tensor: tuple[torch.Tensor, torch.Tensor], type_relation_tensor: tuple[torch.Tensor, torch.Tensor], position_tensor: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states_m, hidden_states_c, W_probs = hidden_states
        hidden_states_m, hidden_states_c, W_probs = self.multi_head_attention(
            input_tensor=(hidden_states_m, hidden_states_c),
            attention_mask=attention_mask,
            type_seq=type_seq,
            type_tensor=type_tensor,
            type_relation_tensor=type_relation_tensor,
            position_tensor=position_tensor,
        )
        hidden_states_m = self.feed_forward(hidden_states_m, type_seq)
        hidden_states_c = self.activation_func(self.feed_forward(hidden_states_c, type_seq)) + 1
        return hidden_states_m, hidden_states_c, W_probs


class WassersteinPredictionHead(nn.Module):
    """share embedding parameters"""

    def __init__(
        self,
        d_model: int,
        n_items: int,
        token_embeddings_m: nn.Embedding,
        token_embeddings_c: nn.Embedding,
    ):
        super().__init__()
        self.token_embeddings_m = token_embeddings_m
        self.token_embeddings_c = token_embeddings_c
        self.vocab_size = n_items + 1
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ELU(),
        )
        self.activation = nn.ELU()

    def forward(self, hidden_states_m: torch.Tensor, hidden_states_c: torch.Tensor, candidates: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        hidden_states_m = self.out(hidden_states_m)  # [B, H] or [M, H]
        hidden_states_c = self.out(hidden_states_c)  # [B, H] or [M, H]
        if candidates is not None:  # [B, H]
            emb1: torch.Tensor = self.token_embeddings_m(candidates)  # [B, C, H]
            emb2: torch.Tensor = self.activation(self.token_embeddings_c(candidates)) + 1  # [B, C, H]
            logits = wasserstein_distance_matmul(
                hidden_states_m[:, None, :],
                hidden_states_c[:, None, :],
                emb1,
                emb2,
            )  # [B, C]
        else:  # [M, H]
            emb1 = self.token_embeddings_m.weight[:self.vocab_size]  # [n_items + 1, H]
            emb2 = self.activation(self.token_embeddings_c.weight[:self.vocab_size]) + 1  # [n_items + 1, H]
            logits = wasserstein_distance_matmul(
                hidden_states_m[:, None, :],
                hidden_states_c[:, None, :],
                emb1,
                emb2,
            )  # [M, n_items + 1]
        return logits
