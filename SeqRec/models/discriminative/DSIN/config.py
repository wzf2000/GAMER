from SeqRec.utils.config import Config


class DSINConfig(Config):
    embedding_size: int = 64
    n_heads: int = 2
    lstm_hidden_size: int = 32
    attention_hidden_size: int = 80
    mlp_hidden_sizes: list[int] = [128, 64]
    max_sessions: int = 16
    dropout: float = 0.2
    initializer_range: float = 0.02
    loss_type: str = "BCE"
