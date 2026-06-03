import os
import torch
import random
import numpy as np
from tqdm import tqdm
from typing import Iterable


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


def get_tqdm(iterable: Iterable | None = None, desc: str = None, total: int = None):
    """
    Get a tqdm progress bar for the given iterable. If iterable is None, total must be provided.
    If desc is provided, it will be used as the description of the progress bar.
    If total is provided, it will be used to set the total number of iterations.
    """
    assert iterable is not None or total is not None, "Either iterable or total must be provided for tqdm progress bar."
    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        return iterable
    if desc is None:
        desc = "Processing"
    return tqdm(iterable, desc=desc, total=total)
