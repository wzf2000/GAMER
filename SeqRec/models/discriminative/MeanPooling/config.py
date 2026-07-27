from SeqRec.utils.config import Config


class MeanPoolingConfig(Config):
    embedding_size: int = 64
    initializer_range: float = 0.02
    loss_type: str = "BCE"

