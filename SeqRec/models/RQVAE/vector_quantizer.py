import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from SeqRec.models.RQVAE.layers import sinkhorn_algorithm
from SeqRec.utils.kmeans import constrained_km, center_distance_for_constraint


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_e: int,
        e_dim: int,
        mu: float = 0.25,
        beta: float = 1,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        sk_epsilon: float = 0.01,
        sk_iters: int = 100,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.mu = mu
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self) -> torch.Tensor:
        return self.embedding.weight

    def init_emb(self, data: torch.Tensor):
        centers, _ = constrained_km(data, 256, init=True)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    def diversity_loss(self, x_q: torch.Tensor, indices, indices_cluster, indices_list) -> torch.Tensor:
        emb = self.embedding.weight
        temp = 1

        pos_list = [indices_list[i] for i in indices_cluster]
        pos_sample = []
        for idx, pos in enumerate(pos_list):
            random_element = random.choice(pos)
            while random_element == indices[idx]:
                random_element = random.choice(pos)
            pos_sample.append(random_element)

        y_true = torch.tensor(pos_sample, device=x_q.device)

        # sim = F.cosine_similarity(x_q, emb, dim=-1)
        sim = torch.matmul(x_q, emb.t())

        # sampled_ids = torch.multinomial(best_scores, num_samples=1)
        sim_self = torch.zeros_like(sim)
        for idx, row in enumerate(sim_self):
            sim_self[idx, indices[idx]] = 1e12
        sim = sim - sim_self
        sim = sim / temp
        loss = F.cross_entropy(sim, y_true)

        return loss

    def diversity_loss_main_entry(self, x: torch.Tensor, x_q: torch.Tensor, indices: torch.Tensor, labels: list[int]) -> torch.Tensor:
        indices_cluster = [labels[idx.item()] for idx in indices]
        target_numbers = list(range(10))
        indices_list = {}
        for target_number in target_numbers:
            indices_list[target_number] = [
                index for index, num in enumerate(labels) if num == target_number
            ]

        diversity_loss = self.diversity_loss(
            x_q, indices, indices_cluster, indices_list
        )
        return diversity_loss

    def vq_init(self, x: torch.Tensor, use_sk: bool = True) -> torch.Tensor:
        latent = x.view(-1, self.e_dim)

        if not self.initted:
            self.init_emb(latent)

        _distance_flag = "distance"

        if _distance_flag == "distance":
            d = (
                torch.sum(latent**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
                - 2 * torch.matmul(latent, self.embedding.weight.t())
            )
        else:
            # Calculate Cosine Similarity
            d = latent @ self.embedding.weight.t()

        if not use_sk or self.sk_epsilon <= 0:
            if _distance_flag == "distance":
                indices = torch.argmin(d, dim=-1)
            else:
                indices = torch.argmax(d, dim=-1)
        else:
            d = center_distance_for_constraint(d)
            d = d.double()

            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                logger.warning("Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        x_q: torch.Tensor = self.embedding(indices)
        x_q = x_q.view(x.shape)

        return x_q

    def forward(self, x: torch.Tensor, label: list[int], idx: int, use_sk: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        _distance_flag = "distance"

        if _distance_flag == "distance":
            d = (
                torch.sum(latent**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
                - 2 * torch.matmul(latent, self.embedding.weight.t())
            )
        else:
            # Calculate Cosine Similarity
            d = latent @ self.embedding.weight.t()
        if not use_sk or self.sk_epsilon <= 0:
            if _distance_flag == "distance":
                if idx != -1:
                    indices = torch.argmin(d, dim=-1)
                else:
                    temp = 1.0
                    prob_dist = F.softmax(-d / temp, dim=1)
                    indices = torch.multinomial(prob_dist, 1).squeeze()
            else:
                indices = torch.argmax(d, dim=-1)
        else:
            d = center_distance_for_constraint(d)
            d = d.double()

            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                logger.warning("Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        x_q: torch.Tensor = self.embedding(indices)
        x_q = x_q.view(x.shape)

        # Diversity
        if self.beta > 0:
            diversity_loss = self.diversity_loss_main_entry(x, x_q, indices, label)
        else:
            diversity_loss = torch.tensor(0.0, device=x.device)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())

        loss = codebook_loss + self.mu * commitment_loss + self.beta * diversity_loss

        # preserve gradients
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices
