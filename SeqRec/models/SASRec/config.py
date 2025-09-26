from SeqRec.utils.config import Config


class SASRecConfig(Config):
    n_layers: int = 2
    n_heads: int = 2
    hidden_size: int = 128
    inner_size: int = 256
    dropout_prob: float = 0.5
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    loss_type: str = "CE"  # or BPR
