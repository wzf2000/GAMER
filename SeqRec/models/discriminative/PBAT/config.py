from SeqRec.utils.config import Config


class PBATConfig(Config):
    n_layers: int = 2
    n_heads: int = 2
    hidden_size: int = 64
    inner_size: int = 256
    dropout_prob: float = 0.2
    hidden_act: str = "elu"
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    mask_ratio: float = 0.2
    loss_type: str = "CE"
