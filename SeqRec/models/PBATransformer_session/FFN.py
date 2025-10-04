import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.t5.modeling_t5 import T5DenseActDense
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersLayerNorm,
)

from SeqRec.models.PBATransformer_session.configuration import PBATransformerConfigSession


# Implementation reference: transformers.models.t5.modeling_t5.T5DenseActDense
class PBATransformerDenseActDense(T5DenseActDense):
    def __init__(self, config: PBATransformerConfigSession, behavior_injection: bool = False, session_injection: bool = False):
        super(T5DenseActDense, self).__init__()
        d_model = config.d_model
        if behavior_injection:
            d_model += config.behavior_embedding_dim
        if session_injection:
            d_model += config.session_embedding_dim

        self.wi = nn.Linear(d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]


class PBATransformerDenseActDenseHalfOutput(T5DenseActDense):
    def __init__(self, config: PBATransformerConfigSession, behavior_injection: bool = False):
        super(T5DenseActDense, self).__init__()
        if behavior_injection:
            self.wi = nn.Linear(
                (config.d_model + config.behavior_embedding_dim),
                config.d_ff,
                bias=False,
            )
        else:
            self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model // 2, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]


class PBATransformerSparseMLPSession(nn.Module):
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(
        self,
        config: PBATransformerConfigSession,
        expert_class: nn.Module = PBATransformerDenseActDense,
        is_sparse: bool = False,
        behavior_injection: bool = False,
        session_injection: bool = False
    ):
        super().__init__()
        self.is_sparse = is_sparse
        if config.shared_expert and is_sparse:
            expert_class = PBATransformerDenseActDenseHalfOutput
            self.shared_expert = PBATransformerDenseActDenseHalfOutput(
                config, behavior_injection
            )
        self.has_shared_expert = config.shared_expert
        if self.is_sparse:
            self.experts = nn.ModuleDict()
            for idx in range(config.num_experts):
                self.experts[f"expert_{idx}"] = expert_class(config, behavior_injection, session_injection)
        else:
            self.mlp: nn.Module = expert_class(config, behavior_injection, session_injection)
        self.behavior_injection = behavior_injection
        if self.behavior_injection:
            self.behavior_embedding = nn.Embedding(
                config.num_behavior + 1, config.behavior_embedding_dim
            )
        self.session_injection = session_injection
        self.num_session = config.num_session
        if self.session_injection:
            self.session_embedding = nn.Embedding(
                config.num_session + 2, config.session_embedding_dim
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_index: torch.Tensor,
        behavior_index: torch.Tensor,
        session_index: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        if not self.has_shared_expert:
            next_states = torch.zeros_like(hidden_states)
            if self.behavior_injection:
                behavior_embedding = self.behavior_embedding(behavior_index)
                hidden_states = torch.cat((hidden_states, behavior_embedding), dim=-1)
            if self.session_injection:
                session = - session_index + session_index.max(dim=1)[0].unsqueeze(1) + 2
                session[session_index == -2] = 0
                session[session_index == -1] = 1
                session[session > (self.num_session + 1)] = self.num_session + 1
                session_embedding = self.session_embedding(session)
                hidden_states = torch.cat((hidden_states, session_embedding), dim=-1)
            if self.is_sparse:
                for idx, expert in enumerate(self.experts.values()):
                    token_indices = position_index == idx
                    next_states[token_indices] = expert(
                        hidden_states[token_indices]
                    ).to(next_states.dtype)
            else:
                next_states = self.mlp(hidden_states)

            return next_states
        else:
            next_states = torch.zeros(
                (
                    hidden_states.size(0),
                    hidden_states.size(1),
                    hidden_states.size(2) // 2,
                ),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if self.behavior_injection:
                behavior_embedding = self.behavior_embedding(behavior_index)
                hidden_states = torch.cat((hidden_states, behavior_embedding), dim=-1)
            if self.is_sparse:
                for idx, expert in enumerate(self.experts.values()):
                    token_indices = position_index == idx
                    next_states[token_indices] = expert(
                        hidden_states[token_indices]
                    ).to(next_states.dtype)
                shared_next_states = self.shared_expert(hidden_states)
                next_states = torch.cat((next_states, shared_next_states), dim=-1)
                return next_states
            else:
                next_states = self.mlp(hidden_states)

            return next_states


class PBATransformerLayerFFSession(nn.Module):
    r"""
    Switch Transformers Feed Forward layer module. This is a wrapper around the Mixture of Experts module.

    Parameters:
        config : ([`PBATransformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        is_sparse (`bool`):
            Whether the MLP layer is a `Sparse` layer (contains a Mixture of Experts) or not
            If `True`, the MLP layer will use a Mixture of Experts (MoE) architecture.
            If `False`, the MLP layer will be a standard dense layer.
            Default: `False`
        behavior_injection (`bool`):
            Whether to inject the behavior embedding into the MLP layer. If `True`, the input
            hidden states will be concatenated with the behavior embedding before being passed to the MLP.
            If `False`, the input hidden states will be passed directly to the MLP.
            Default: `False`
    """

    def __init__(
        self, config: PBATransformerConfigSession, is_sparse: bool = False, behavior_injection: bool = False, session_injection: bool = False,
    ):
        super().__init__()
        self.is_sparse = is_sparse

        # Check if it is a sparse layer, if not then it is a dense layer
        self.mlp = PBATransformerSparseMLPSession(
            config, is_sparse=self.is_sparse, behavior_injection=behavior_injection, session_injection=session_injection,
        )

        self.layer_norm = SwitchTransformersLayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor, position_index: torch.Tensor, behavior_index: torch.Tensor, session_index: torch.Tensor) -> torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states, position_index, behavior_index, session_index)

        output = hidden_states + self.dropout(forwarded_states)

        return output
