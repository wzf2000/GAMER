import torch
import numpy as np
from loguru import logger
from torch.utils.data import Dataset


class EmbDataset(Dataset):
    def __init__(self, data_path: str, local_rank: int = 0):
        self.data_path = data_path
        self.embeddings: np.ndarray = np.load(data_path)
        self.local_rank = local_rank
        std = self.embeddings.std()
        if std < 0.2:
            if self.local_rank == 0:
                logger.warning(
                    f"Standard deviation of embeddings is too low: {std:.4f}. "
                    "This may lead to poor performance. Consider normalizing the embeddings."
                )
            self.embeddings /= std
        self.dim: int = self.embeddings.shape[-1]

    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, int]:
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb, index

    def __len__(self):
        return len(self.embeddings)
