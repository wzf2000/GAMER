from SeqRec.utils.config import Config


class DIENCVRConfig(Config):
    embedding_size: int = 64
    hidden_size: int = 64
    attention_hidden_size: int = 80
    mlp_hidden_sizes: list[int] = [128, 64]
    dropout: float = 0.2
    initializer_range: float = 0.02
    loss_type: str = "BCE"

