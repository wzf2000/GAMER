import torch
import random
import numpy as np


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(gpu_id: int) -> torch.device:
    """
    Set the device for PyTorch based on the provided GPU ID.
    If gpu_id is -1, it returns the CPU device.
    If gpu_id is a valid ID, it returns the corresponding CUDA device if available,
    otherwise it defaults to the CPU.
    """
    if gpu_id == -1:
        return torch.device("cpu")
    else:
        return torch.device(
            "cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu"
        )
