from SeqRec.utils.config import Config


class MBSTRConfig(Config):
    n_layers: int = 2
    n_heads: int = 2
    hidden_size: int = 64
    inner_size: int = 256
    dropout_prob: float = 0.2
    hidden_act: str = "relu"
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    mask_ratio: float = 0.2
    loss_type: str = "CE"
    num_buckets: int = 32
    max_distance: int = 40
    behavior_head: bool = True
    behavior_attention: bool = True
    behavior_moe: bool = True
    behavior_position_bias: bool = True
    n_shared_experts: int = 3
    n_specific_experts: int = 1
