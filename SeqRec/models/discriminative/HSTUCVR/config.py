from SeqRec.utils.config import Config


class HSTUCVRConfig(Config):
    n_layers: int = 16
    n_heads: int = 8
    hidden_size: int = 128
    dropout_prob: float = 0.2
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    use_behavior_embedding: bool = True
    loss_type: str = "BCE"

