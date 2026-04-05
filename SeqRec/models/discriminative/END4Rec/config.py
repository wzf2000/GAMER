from SeqRec.utils.config import Config


class END4RecConfig(Config):
    n_layers: int = 2
    hidden_size: int = 64
    dropout_prob: float = 0.2
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    loss_type: str = "CE"
    # Number of chunks for the diagonal MLP — must divide hidden_size
    n_chunks: int = 4
    # Gumbel Softmax temperature for hard noise elimination
    tau: float = 1.0
    # Weight for noise-decoupling contrastive loss
    cl_weight: float = 0.1
    # Weight for compactness regularization
    compactness_weight: float = 0.01
    # ε in the compactness regularization formula
    compactness_eps: float = 1.0
