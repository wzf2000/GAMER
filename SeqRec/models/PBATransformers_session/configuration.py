from transformers.models.switch_transformers.configuration_switch_transformers import (
    SwitchTransformersConfig,
)


class PBATransformerConfigSession(SwitchTransformersConfig):
    """
    Configuration class for the PBATransformer model.
    """
    def __init__(
        self,
        behavior_injection: bool = False,  # Whether to inject behavior embeddings in the FFN layers
        behavior_embedding_dim: int = 64,  # Dimension of the behavior embeddings
        num_positions: int = 4,  # Number of tokens for one items (1 behavior + 3 semantic tokens)
        num_behavior: int = 4,  # Number of behavior types
        n_positions: int = 50,  # Maximum history length of the sequence
        sparse_layers_encoder: list[int] = [],  # Sparse layers in the encoder (0-indexed)
        sparse_layers_decoder: list[int] = [],  # Sparse layers in the decoder (0-indexed)
        behavior_injection_encoder: list[int] = [],  # Layers in the encoder to inject behavior embeddings
        behavior_injection_decoder: list[int] = [],  # Layers in the decoder to inject behavior embeddings
        # Whether to use one expert for item semantic tokens and another for other tokens only
        # Suppose the input sequence is:
        #   [
        #       user_token,
        #       behavior_token, item_token_1, item_token_2, item_token_3,
        #       behavior_token, item_token_1, item_token_2, item_token_3,
        #       ...,
        #       EOS_token, PAD_token, ...
        #   ]
        # If Moe_behavior_only is True, the expert ID sequence will be:
        #   [
        #       0,
        #       0, 1, 1, 1,
        #       0, 1, 1, 1,
        #       ...,
        #       0, 0, 0, ...
        #   ]
        # If Moe_behavior_only is False, the expert ID sequence will be:
        #   [
        #       0,
        #       1, 2, 3, 4,
        #       1, 2, 3, 4,
        #       ...,
        #       0, 0, 0, ...
        #   ]
        Moe_behavior_only: bool = False,
        shared_expert: bool = False,  # Whether to use half output for shared expert
        use_behavior_token: bool = True,  # Whether to use behavior token in the input sequence
        behavior_maps: dict[int, int] = {},  # Mapping from behavior token IDs to behavior embedding IDs
        use_user_token: bool = False,  # Whether to use user token in the input sequence
        freqnum: int = 64,
        time_embedding_encoder: list[int] = [],
        time_embedding_decoder: list[int] = [],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.behavior_injection = behavior_injection
        self.behavior_embedding_dim = behavior_embedding_dim
        self.num_positions = num_positions
        if not Moe_behavior_only:
            self.num_experts = num_positions + 1  # 1 for the BOS, EOS, PAD tokens
        else:
            self.num_experts = 2  # 1 for the item semantic tokens, 1 for the other tokens
        self.num_behavior = num_behavior
        self.n_positions = n_positions
        self.sparse_layers_encoder = sparse_layers_encoder
        self.sparse_layers_decoder = sparse_layers_decoder
        self.behavior_injection_encoder = behavior_injection_encoder
        self.behavior_injection_decoder = behavior_injection_decoder
        self.Moe_behavior_only = Moe_behavior_only
        self.shared_expert = shared_expert
        self.use_behavior_token = use_behavior_token
        self.behavior_maps = behavior_maps
        self.use_user_token = use_user_token
        self.freqnum = freqnum
        self.time_embedding_encoder = time_embedding_encoder
        self.time_embedding_decoder = time_embedding_decoder
