import torch
import torch.nn as nn
from typing import Sequence

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):

    def __init__(
        self,
        n_e_list: list[int],
        e_dim: int,
        sk_epsilons: list[float],
        beta: float = 1,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        sk_iters: int = 100,
    ):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.vq_layers: Sequence[VectorQuantizer] = nn.ModuleList(
            [
                VectorQuantizer(
                    n_e,
                    e_dim,
                    beta=beta,
                    kmeans_init=self.kmeans_init,
                    kmeans_iters=self.kmeans_iters,
                    sk_epsilon=sk_epsilon,
                    sk_iters=sk_iters,
                )
                for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)
            ]
        )

    def get_codebook(self) -> torch.Tensor:
        all_codebook = []
        for quantizer in self.vq_layers:
            quantizer: VectorQuantizer
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def vq_ini(self, x: torch.Tensor):
        x_q = 0
        residual = x
        for idx, quantizer in enumerate(self.vq_layers):
            quantizer: VectorQuantizer
            x_res = quantizer.vq_init(residual, use_sk=True)
            residual = residual - x_res
            x_q = x_q + x_res

    def forward(self, x: torch.Tensor, labels: dict[str, list[int]], use_sk: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x

        for idx, quantizer in enumerate(self.vq_layers):
            quantizer: VectorQuantizer
            label = labels[str(idx)]

            x_res, loss, indices = quantizer(residual, label, idx, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices
