import torch
import torch.nn as nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from SeqRec.models.Qwen_Moe.FFN import MyQwen3MoeMLP


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
            for idx in range((config.num_experts - 1) * config.num_behavior + 1):
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
        action_index: torch.Tensor,
    ) -> torch.Tensor:
        next_states = torch.zeros_like(hidden_states)
        if self.behavior_injection:
            behavior_embedding = self.behavior_embedding(behavior_index)
            hidden_states = torch.cat((hidden_states, behavior_embedding), dim=-1)
        if self.is_sparse:
            index = (self.num_experts - 1) * (action_index - 1) + position_index
            index[index < 0] = 0
            for idx, expert in enumerate(self.experts.values()):
                token_indices = index == idx
                next_states[token_indices] = expert(hidden_states[token_indices]).to(
                    next_states.dtype
                )
        else:
            next_states = self.mlp(hidden_states)

        return next_states
