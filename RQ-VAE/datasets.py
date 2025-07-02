import numpy as np
import torch
from torch.utils.data import Dataset


class EmbDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.embeddings: np.ndarray = np.load(data_path)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, int]:
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb, index

    def __len__(self):
        return len(self.embeddings)
