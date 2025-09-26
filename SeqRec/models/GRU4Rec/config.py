from SeqRec.utils.config import Config


class GRU4RecConfig(Config):
    embedding_size: int = 64
    hidden_size: int = 128
    n_layers: int = 1
    dropout: float = 0.3
    loss_type: str = "CE"  # or BPR
