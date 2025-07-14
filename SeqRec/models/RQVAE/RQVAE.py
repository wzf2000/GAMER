import torch
import numpy as np
from argparse import Namespace
from torch import nn
from torch.nn import functional as F

from SeqRec.models.RQVAE.layers import MLPLayers
from SeqRec.models.RQVAE.resiual_vector_quantizer import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        num_emb_list: list[int] = [256, 256, 256, 256],
        e_dim: int = 64,
        layers: list[int] = [2048, 1024, 512, 256, 128, 64],
        dropout_prob: float = 0.0,
        bn: bool = False,
        loss_type: str = "mse",
        quant_loss_weight: float = 1.0,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        sk_epsilons: list[float] = [0.0, 0.0, 0.0, 0.003],
        sk_iters: int = 50,
        alpha: float = 1.0,
        beta: float = 0.001,
        n_clusters: int = 10,
        sample_strategy: str = "all",
        cf_embedding: np.ndarray | int = 0,
    ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.cf_embedding = cf_embedding
        self.alpha = alpha
        self.beta = beta
        self.n_clusters = n_clusters
        self.sample_strategy = sample_strategy

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(
            layers=self.encode_layer_dims, dropout=self.dropout_prob, bn=self.bn
        )

        self.rq = ResidualVectorQuantizer(
            num_emb_list,
            e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
        )

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(
            layers=self.decode_layer_dims, dropout=self.dropout_prob, bn=self.bn
        )

    @property
    def args(self) -> Namespace:
        return Namespace(
            in_dim=self.in_dim,
            num_emb_list=self.num_emb_list,
            e_dim=self.e_dim,
            layers=self.layers,
            dropout_prob=self.dropout_prob,
            bn=self.bn,
            loss_type=self.loss_type,
            quant_loss_weight=self.quant_loss_weight,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
            alpha=self.alpha,
            beta=self.beta,
            n_clusters=self.n_clusters,
            sample_strategy=self.sample_strategy,
        )

    def forward(
        self,
        x: torch.Tensor,
        labels: dict[str, list[int]],
        use_sk: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x, labels, use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices, x_q

    def CF_loss(self, quantized_rep: torch.Tensor, encoded_rep: torch.Tensor) -> torch.Tensor:
        batch_size = quantized_rep.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=quantized_rep.device)
        similarities = torch.matmul(quantized_rep, encoded_rep.transpose(0, 1))
        cf_loss = F.cross_entropy(similarities, labels)
        return cf_loss

    def vq_initialization(self, x: torch.Tensor, use_sk: bool = True):
        self.rq.vq_ini(self.encoder(x))

    @torch.no_grad()
    def get_indices(self, xs: torch.Tensor, labels: dict[str, list[int]], use_sk: bool = False) -> torch.Tensor:
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, labels, use_sk=use_sk)
        return indices

    def compute_loss(
        self,
        out: torch.Tensor,
        quant_loss: torch.Tensor,
        emb_idx: torch.Tensor,
        dense_out: torch.Tensor,
        xs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.loss_type == "mse":
            loss_recon = F.mse_loss(out, xs, reduction="mean")
        elif self.loss_type == "l1":
            loss_recon = F.l1_loss(out, xs, reduction="mean")
        else:
            raise ValueError("incompatible loss type")

        rqvae_n_diversity_loss = loss_recon + self.quant_loss_weight * quant_loss

        # CF_Loss
        if self.alpha > 0:
            cf_embedding_in_batch = self.cf_embedding[emb_idx]
            cf_embedding_in_batch = torch.from_numpy(cf_embedding_in_batch).to(
                dense_out.device
            )
            cf_loss = self.CF_loss(dense_out, cf_embedding_in_batch)
        else:
            cf_loss = torch.tensor(0.0, device=dense_out.device)

        total_loss = rqvae_n_diversity_loss + self.alpha * cf_loss

        return total_loss, cf_loss, loss_recon, quant_loss
