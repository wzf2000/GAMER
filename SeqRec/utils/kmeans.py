import torch
import numpy as np


def constrained_km(data: np.ndarray, n_clusters: int = 10, init: bool = False) -> tuple[torch.Tensor, list[int]]:
    from k_means_constrained import KMeansConstrained

    x = data
    size_min = min(len(data) // (n_clusters * 2), 50 if init else 10)  # 50 for the very first time, 10 the latter
    clf = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_min * 4 if init else n_clusters * 6,  # 'size_min * 4' for the very first time, 'n_clusters * 6' for the latter
        max_iter=10,
        n_init=10,
        n_jobs=10,
        verbose=False,
    )
    clf.fit(x)
    t_centers = torch.from_numpy(clf.cluster_centers_)
    t_labels = torch.from_numpy(clf.labels_).tolist()
    return t_centers, t_labels


def center_distance_for_constraint(distances: torch.Tensor) -> torch.Tensor:
    # distances: B, K
    max_distance = distances.max()
    min_distance = distances.min()

    middle = (max_distance + min_distance) / 2
    amplitude = max_distance - middle + 1e-5
    assert amplitude > 0
    centered_distances = (distances - middle) / amplitude
    return centered_distances
