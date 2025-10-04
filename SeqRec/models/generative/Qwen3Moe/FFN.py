import torch
import torch.nn as nn
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
