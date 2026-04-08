import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.t5.modeling_t5 import T5DenseActDense
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


class MyQwen3MoeMLP(nn.Module):
    def __init__(
        self, config: Qwen3MoeConfig, behavior_injection: bool = False
    ):
        super().__init__()
        self.config = config
        if behavior_injection:
            self.hidden_size = config.moe_intermediate_size + config.behavior_embedding_dim
        else:
            self.hidden_size = config.moe_intermediate_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.moe_intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        down_proj = self.down_proj(self.dropout(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
        return down_proj


class MyQwen3SparseMLP(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        expert_class: nn.Module = MyQwen3MoeMLP,
        is_sparse: bool = False,
        behavior_injection: bool = False,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.is_sparse = is_sparse
        if self.is_sparse:
            self.experts = nn.ModuleDict()
            for idx in range(config.num_experts):
                self.experts[f"expert_{idx}"] = expert_class(config, behavior_injection)
        else:
            self.mlp: nn.Module = expert_class(config, behavior_injection)
        self.behavior_injection = behavior_injection
        if self.behavior_injection:
            self.behavior_embedding = nn.Embedding(
                config.num_behavior + 1, config.behavior_embedding_dim
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_index: torch.Tensor,
        behavior_index: torch.Tensor,
    ) -> torch.Tensor:
        next_states = torch.zeros_like(hidden_states)
        if self.behavior_injection:
            behavior_embedding = self.behavior_embedding(behavior_index)
            hidden_states = torch.cat((hidden_states, behavior_embedding), dim=-1)
        if self.is_sparse:
            for idx, expert in enumerate(self.experts.values()):
                token_indices = position_index == idx
                next_states[token_indices] = expert(hidden_states[token_indices]).to(
                    next_states.dtype
                )
        else:
            next_states = self.mlp(hidden_states)

        return next_states


class PBATransformerMlp(T5DenseActDense):
    def __init__(self, config: Qwen3MoeConfig, behavior_injection: bool = False):
        super(T5DenseActDense, self).__init__()
        if behavior_injection:
            self.wi = nn.Linear(
                (config.moe_intermediate_size + config.behavior_embedding_dim),
                config.intermediate_size,
                bias=False,
            )  # Concatenate the behavior embedding dimension to the input dimension
        else:
            self.wi = nn.Linear(config.moe_intermediate_size, config.intermediate_size, bias=False)
        self.wo = nn.Linear(config.intermediate_size, config.moe_intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.hidden_act]


class DenseMLP(nn.Module):
    """Plain Qwen3-style FFN without any MoE or routing. Ignores position/behavior indices."""

    def __init__(self, config: Qwen3MoeConfig, behavior_injection: bool = False):
        super().__init__()
        # behavior_injection is intentionally ignored for dense MLP
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_index: torch.Tensor,
        behavior_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.down_proj(
            self.dropout(self.act_fn(self.gate_proj(hidden_states))) * self.up_proj(hidden_states)
        )


class RouterMoeBlock(nn.Module):
    """Qwen3MoE-style MoE with learned top-k routing. Routing is NOT fixed by position/behavior indices."""

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.num_experts = getattr(config, "num_experts", 8)
        self.top_k = getattr(config, "num_experts_per_tok", 2)
        self.norm_topk_prob = getattr(config, "norm_topk_prob", False)

        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MyQwen3MoeMLP(config) for _ in range(self.num_experts)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_index: torch.Tensor,
        behavior_index: torch.Tensor,
    ) -> torch.Tensor:
        B, S, D = hidden_states.shape
        hidden_flat = hidden_states.view(-1, D)  # [T, D]

        router_logits = self.gate(hidden_flat)  # [T, num_experts]

        routing_weights = torch.softmax(router_logits.float(), dim=-1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        # selected_experts: [T, top_k], routing_weights: [T, top_k]
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # expert_mask: [num_experts, top_k, T]
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        final_hidden = torch.zeros_like(hidden_flat)
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            # idx: which top-k slot, top_x: which token
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue
            current_state = hidden_flat[top_x]  # [num_tokens, D]
            expert_out = expert(current_state).to(hidden_states.dtype)
            weighted = expert_out * routing_weights[top_x, idx, None]
            final_hidden.index_add_(0, top_x, weighted)

        return final_hidden.view(B, S, D)


class PBATransformerSparseMLP(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        expert_class: nn.Module = PBATransformerMlp,
        is_sparse: bool = False,
        behavior_injection: bool = False,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.is_sparse = is_sparse
        if self.is_sparse:
            self.experts = nn.ModuleDict()
            for idx in range(config.num_experts):
                self.experts[f"expert_{idx}"] = expert_class(config, behavior_injection)
        else:
            self.mlp: nn.Module = expert_class(config, behavior_injection)
        self.behavior_injection = behavior_injection
        if self.behavior_injection:
            self.behavior_embedding = nn.Embedding(
                config.num_behavior + 1, config.behavior_embedding_dim
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_index: torch.Tensor,
        behavior_index: torch.Tensor,
    ) -> torch.Tensor:
        """ """
        next_states = torch.zeros_like(hidden_states)
        if self.behavior_injection:
            behavior_embedding = self.behavior_embedding(behavior_index)
            hidden_states = torch.cat((hidden_states, behavior_embedding), dim=-1)
        if self.is_sparse:
            for idx, expert in enumerate(self.experts.values()):
                token_indices = position_index == idx
                next_states[token_indices] = expert(hidden_states[token_indices]).to(
                    next_states.dtype
                )
        else:
            next_states = self.mlp(hidden_states)

        return next_states
