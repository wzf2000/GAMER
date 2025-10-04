from SeqRec.utils.config import Config


class MBHTConfig(Config):
    n_layers: int = 2
    n_heads: int = 2
    hidden_size: int = 128
    inner_size: int = 256
    dropout_prob: float = 0.5
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    mask_ratio: float = 0.2
    loss_type: str = "CE"  # or BPR
    enable_hg: bool = True
    enable_ms: bool = True
    hyper_len: int = 6
    scales: list[int] = [5, 8, 40]
