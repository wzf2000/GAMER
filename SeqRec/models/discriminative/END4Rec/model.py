"""
END4Rec: Efficient Noise-Decoupling for Multi-Behavior Sequential Recommendation.

Reference: https://dl.acm.org/doi/abs/10.1145/3589334.3645380
           WWW 2024

Architecture overview:
  1. Behavior-Aware Sequence Embedding: e = e_v + e_b + e_p  (Eq. 1)
  2. Efficient Behavior Miner (EBM): FFT-based O(N log N) encoder with
     a Chunked Diagonal MLP  (Eq. 2-3)
  3. Hard Noise Eliminator: token-level Gumbel-Softmax denoising driven by
     Poisson-based behavior preference values  (Eq. 5-6)
  4. Soft Noise Filter: frequency-domain denoising with behavior-specific
     learnable filters  (Eq. 7)
  5. Noise-Decoupling Contrastive Learning + Compactness Regularization
     (Eq. 4, 8-9)

Training uses a combined loss (all components active simultaneously) as a
simplified approximation of the staged training in the paper.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from SeqRec.models.discriminative.END4Rec.config import END4RecConfig
from SeqRec.modules.model_base.seq_model import SeqModel


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class ChunkedDiagonalMLP(nn.Module):
    """
    Chunked Diagonal MLP (Eq. 3): splits d_model into n_chunks and applies
    independent two-layer MLPs to each chunk, reducing total parameter count.
    Includes a residual connection and LayerNorm.
    """

    def __init__(self, d_model: int, n_chunks: int, dropout: float, layer_norm_eps: float):
        super().__init__()
        assert d_model % n_chunks == 0, (
            f"hidden_size ({d_model}) must be divisible by n_chunks ({n_chunks})"
        )
        self.n_chunks = n_chunks
        self.chunk_size = d_model // n_chunks
        # Chunk-wise weight matrices: [n_chunks, chunk_size, chunk_size]
        self.w1 = nn.Parameter(torch.empty(n_chunks, self.chunk_size, self.chunk_size))
        self.b1 = nn.Parameter(torch.zeros(n_chunks, self.chunk_size))
        self.w2 = nn.Parameter(torch.empty(n_chunks, self.chunk_size, self.chunk_size))
        self.b2 = nn.Parameter(torch.zeros(n_chunks, self.chunk_size))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        B, L, d = x.shape
        # Reshape into chunks: [B, L, n_chunks, chunk_size]
        xc = x.view(B, L, self.n_chunks, self.chunk_size)
        # First linear + activation
        h = torch.einsum('blnc,ncd->blnd', xc, self.w1) + self.b1
        h = F.gelu(h)
        # Second linear
        h = torch.einsum('blnc,ncd->blnd', h, self.w2) + self.b2
        out = h.reshape(B, L, d)
        return self.layer_norm(self.dropout(out) + x)


class EBMLayer(nn.Module):
    """
    Single Efficient Behavior Miner layer (Eq. 2-3):
      X = FFT(S)
      X̃ = W ⊙ X          (learnable complex frequency filter)
      S̃ = IFFT(X̃)
    followed by a Chunked Diagonal MLP with residual.
    """

    def __init__(self, d_model: int, n_chunks: int, dropout: float, layer_norm_eps: float):
        super().__init__()
        # Learnable complex frequency filter (real + imaginary parts)
        self.filter_real = nn.Parameter(torch.ones(1, 1, d_model))
        self.filter_imag = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mlp = ChunkedDiagonalMLP(d_model, n_chunks, dropout, layer_norm_eps)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        # S: [B, L, d]
        L = S.size(1)
        # FFT requires float32
        S_f32 = S.float()
        X = torch.fft.rfft(S_f32, dim=1)  # complex64: [B, L//2+1, d]

        # Element-wise complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        fr = self.filter_real.float()
        fi = self.filter_imag.float()
        X_real = X.real * fr - X.imag * fi
        X_imag = X.real * fi + X.imag * fr
        X_filt = torch.view_as_complex(
            torch.stack([X_real, X_imag], dim=-1).contiguous()
        )

        # IFFT back to sequence space
        S_filt = torch.fft.irfft(X_filt, n=L, dim=1).to(S.dtype)  # [B, L, d]

        # Residual + LayerNorm after frequency filtering
        S_out = self.layer_norm(self.dropout(S_filt) + S)
        # Chunked MLP with its own residual + LayerNorm
        S_out = self.mlp(S_out)
        return S_out


class HardNoiseEliminator(nn.Module):
    """
    Hard Noise Eliminator (Eq. 5-6):
    Computes Poisson-based preference values p_b for each behavior type and
    uses Gumbel Softmax to produce a differentiable binary token-level mask
    that separates signal tokens (S_hp) from noise tokens (S_hn).
    """

    def __init__(self, n_behaviors: int, max_seq_len: int, tau: float):
        super().__init__()
        self.tau = tau
        # Learnable λ_b per behavior (raw; softplus ensures positivity)
        self.lambda_raw = nn.Parameter(torch.ones(n_behaviors))
        # Learnable per-position threshold (raw; sigmoid maps to (0,1))
        self.threshold = nn.Parameter(torch.zeros(max_seq_len))

    def _preference(self) -> torch.Tensor:
        """
        Poisson PMF at mode k=λ_b, shifted by 1 (Eq. 5):
          p_b = exp(-λ_b) * λ_b^{λ_b} / λ_b! + 1
        Computed in log-space for numerical stability.
        """
        lambda_b = F.softplus(self.lambda_raw) + 1e-6  # [n_behaviors], > 0
        log_pmf = -lambda_b + lambda_b * torch.log(lambda_b) - torch.lgamma(lambda_b + 1)
        return torch.exp(log_pmf) + 1.0  # [n_behaviors], values in (1, 2]

    def forward(
        self,
        S: torch.Tensor,
        behavior_seq: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            S:             [B, L, d]  EBM representations
            behavior_seq:  [B, L]     behavior types, 1-indexed (0 = padding)
            padding_mask:  [B, L]     float, 1 = real token, 0 = padding
        Returns:
            S_hp:  [B, L, d]  signal (hard-denoised positive)
            S_hn:  [B, L, d]  noise (hard-denoised negative)
        """
        L = S.size(1)
        p_b = self._preference()  # [n_behaviors]

        # Map each position to its behavior preference value
        beh_idx = (behavior_seq - 1).clamp(min=0)  # [B, L], 0-indexed
        pref = p_b[beh_idx]  # [B, L]

        # Per-position threshold in (0, 1)
        t = torch.sigmoid(self.threshold[:L]).unsqueeze(0)  # [1, L]
        signal_logit = pref - t  # [B, L]; positive → likely signal

        # Gumbel Softmax: [noise_logit, signal_logit] → differentiable hard mask
        logits = torch.stack([-signal_logit, signal_logit], dim=-1)  # [B, L, 2]
        if self.training:
            soft = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
            mask = soft[..., 1]  # [B, L], 1 = signal
        else:
            mask = (signal_logit > 0).float()  # [B, L]

        # Apply padding mask so padding tokens are never treated as signal
        mask = mask * padding_mask  # [B, L]

        S_hp = S * mask.unsqueeze(-1)                                             # signal
        S_hn = S * (1.0 - mask).unsqueeze(-1) * padding_mask.unsqueeze(-1)       # noise
        return S_hp, S_hn


class SoftNoiseFilter(nn.Module):
    """
    Soft Noise Filter (Eq. 7):
    Applies behavior-proportion-weighted complex frequency filters to S_hp,
    producing S_sp (signal) and S_sn (noise = what was filtered out).
    """

    def __init__(self, d_model: int, n_behaviors: int, dropout: float, layer_norm_eps: float):
        super().__init__()
        self.n_behaviors = n_behaviors
        # Behavior-specific complex filters: [n_beh, d]
        self.filter_real = nn.Parameter(torch.ones(n_behaviors, d_model))
        self.filter_imag = nn.Parameter(torch.zeros(n_behaviors, d_model))
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        S_hp: torch.Tensor,
        behavior_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            S_hp:          [B, L, d]  hard-denoised positive sequence
            behavior_seq:  [B, L]     behavior types, 1-indexed (0 = padding)
        Returns:
            S_sp:  [B, L, d]  soft-denoised positive (residual+LN applied)
            S_sn:  [B, L, d]  soft noise component (S_hp - filtered)
        """
        B, L, d = S_hp.shape

        # Compute per-sample behavior-proportion weights
        one_hot = F.one_hot(
            behavior_seq.clamp(min=0), num_classes=self.n_behaviors + 1
        ).float()  # [B, L, n_beh+1]
        beh_counts = one_hot[:, :, 1:].sum(dim=1)  # [B, n_beh], drop padding class
        beh_w = beh_counts / (beh_counts.sum(dim=-1, keepdim=True) + 1e-8)  # [B, n_beh]

        # Behavior-weighted composite filter: [B, d]
        W_r = (beh_w.unsqueeze(-1) * self.filter_real.unsqueeze(0)).sum(dim=1)
        W_i = (beh_w.unsqueeze(-1) * self.filter_imag.unsqueeze(0)).sum(dim=1)

        # FFT on hard-denoised sequence
        X_hp = torch.fft.rfft(S_hp.float(), dim=1)  # complex64: [B, L//2+1, d]

        # Apply complex filter
        W_r = W_r[:, None, :].float()  # [B, 1, d]
        W_i = W_i[:, None, :].float()  # [B, 1, d]
        X_real = X_hp.real * W_r - X_hp.imag * W_i
        X_imag = X_hp.real * W_i + X_hp.imag * W_r
        X_filt = torch.view_as_complex(
            torch.stack([X_real, X_imag], dim=-1).contiguous()
        )

        # IFFT
        S_sp_raw = torch.fft.irfft(X_filt, n=L, dim=1).to(S_hp.dtype)  # [B, L, d]

        # Soft noise = what the frequency filter removed
        S_sn = S_hp - S_sp_raw  # [B, L, d]

        # Residual + LayerNorm for the positive signal
        S_sp = self.layer_norm(self.dropout(S_sp_raw) + S_hp)

        return S_sp, S_sn


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class END4Rec(SeqModel):
    """
    END4Rec: Efficient Noise-Decoupling for Multi-Behavior Sequential Recommendation.

    Reference: https://dl.acm.org/doi/abs/10.1145/3589334.3645380 (WWW 2024)
    """

    def __init__(
        self,
        config: END4RecConfig,
        n_items: int,
        max_his_len: int,
        n_behaviors: int,
        **kwargs,
    ):
        super().__init__(config, n_items)
        self.hidden_size = config.hidden_size
        self.n_layers = config.n_layers
        self.dropout_prob = config.dropout_prob
        self.layer_norm_eps = config.layer_norm_eps
        self.initializer_range = config.initializer_range
        self.n_chunks = config.n_chunks
        self.tau = config.tau
        self.cl_weight = config.cl_weight
        self.compactness_weight = config.compactness_weight
        self.compactness_eps = config.compactness_eps
        self.max_seq_length = max_his_len
        self.n_behaviors = n_behaviors

        assert config.loss_type == "CE", "END4Rec only supports CE loss"
        self._init(config.loss_type)

    # ------------------------------------------------------------------
    # Required overrides from SeqModel
    # ------------------------------------------------------------------

    def _define_parameters(self):
        # Embeddings
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.behavior_embedding = nn.Embedding(self.n_behaviors + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.embed_drop = nn.Dropout(self.dropout_prob)

        # Stacked EBM layers
        self.ebm_layers = nn.ModuleList([
            EBMLayer(self.hidden_size, self.n_chunks, self.dropout_prob, self.layer_norm_eps)
            for _ in range(self.n_layers)
        ])

        # Hard Noise Eliminator
        self.hard_eliminator = HardNoiseEliminator(
            n_behaviors=self.n_behaviors,
            max_seq_len=self.max_seq_length,
            tau=self.tau,
        )

        # Soft Noise Filter
        self.soft_filter = SoftNoiseFilter(
            d_model=self.hidden_size,
            n_behaviors=self.n_behaviors,
            dropout=self.dropout_prob,
            layer_norm_eps=self.layer_norm_eps,
        )

        # Output LayerNorm applied to S_sp
        self.out_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, item_seq: torch.Tensor, behavior_seq: torch.Tensor) -> torch.Tensor:
        """Behavior-aware embedding: e = e_v + e_b + e_p  (Eq. 1)."""
        B, L = item_seq.shape
        # Position indices 1..L; 0 at padding positions
        pos = torch.arange(1, L + 1, device=item_seq.device).unsqueeze(0).expand(B, -1)
        pos = pos * (item_seq != 0).long()
        S = self.item_embedding(item_seq) + self.behavior_embedding(behavior_seq) + self.position_embedding(pos)
        return self.embed_drop(self.embed_ln(S))

    def _compactness_reg(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compactness regularization (Eq. 4):
          R(X, ε) = 0.5 * log det(I + d/(L·ε²) · X Xᵀ)
        Uses the Sylvester determinant theorem to operate on the smaller of
        the two factor matrices.
        """
        B, L, d = X.shape
        scale = float(d) / (float(L) * self.compactness_eps ** 2)
        X32 = X.float()
        if d <= L:
            M = torch.eye(d, device=X.device, dtype=torch.float32).unsqueeze(0) + scale * torch.bmm(
                X32.transpose(1, 2), X32
            )  # [B, d, d]
        else:
            M = torch.eye(L, device=X.device, dtype=torch.float32).unsqueeze(0) + scale * torch.bmm(
                X32, X32.transpose(1, 2)
            )  # [B, L, L]
        _, logdet = torch.linalg.slogdet(M)
        return (0.5 * logdet.mean()).to(X.dtype)

    def _contrastive_loss(
        self,
        S_pos: torch.Tensor,
        S_orig: torch.Tensor,
        S_neg: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Noise-decoupling contrastive loss (Eq. 8-9):
          L_CL = -Σ [ log σ(Q(S_pos)−Q(S_orig)) + log σ(Q(S_orig)−Q(S_neg)) ]
        where Q(S) = dot product of the sequence representation with the
        target item embedding.
        """
        target_emb = self.item_embedding.weight[target]  # [B, d]
        q_pos  = (S_pos  * target_emb).sum(-1)  # [B]
        q_orig = (S_orig * target_emb).sum(-1)  # [B]
        q_neg  = (S_neg  * target_emb).sum(-1)  # [B]
        return -F.logsigmoid(q_pos - q_orig).mean() - F.logsigmoid(q_orig - q_neg).mean()

    def _encode(
        self,
        item_seq: torch.Tensor,
        behavior_seq: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full encoding pipeline.
        Returns (S_sp_last, S_ebm_last, S_hp_last, S_hn_last, S_sn_last, S_sp_full).
        All *_last tensors are [B, d]; S_sp_full is [B, L, d].
        """
        padding_mask = (item_seq != 0).float()  # [B, L]
        last_idx = seq_len - 1  # [B]

        # 1. Embedding
        S = self._embed(item_seq, behavior_seq)  # [B, L, d]

        # 2. EBM stack
        S_ebm = S
        for layer in self.ebm_layers:
            S_ebm = layer(S_ebm)

        # 3. Hard Noise Eliminator
        S_hp, S_hn = self.hard_eliminator(S_ebm, behavior_seq, padding_mask)

        # 4. Soft Noise Filter
        S_sp, S_sn = self.soft_filter(S_hp, behavior_seq)

        # 5. Output LayerNorm
        S_sp = self.out_ln(S_sp)

        return (
            self.gather_indexes(S_sp,  last_idx),   # [B, d]
            self.gather_indexes(S_ebm, last_idx),   # [B, d]
            self.gather_indexes(S_hp,  last_idx),   # [B, d]
            self.gather_indexes(S_hn,  last_idx),   # [B, d]
            self.gather_indexes(S_sn,  last_idx),   # [B, d]
            S_sp,                                   # [B, L, d]  for compactness reg
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        item_seq: torch.Tensor,
        behavior_seq: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the fully denoised representation at the last valid position: [B, d]."""
        return self._encode(item_seq, behavior_seq, seq_len)[0]

    def calculate_loss(self, interaction: dict) -> torch.Tensor:
        item_seq     = interaction['inputs']      # [B, L]
        behavior_seq = interaction['behaviors']   # [B, L]
        seq_len      = interaction['seq_len']     # [B]
        target       = interaction['target']      # [B]

        S_sp_last, S_ebm_last, S_hp_last, S_hn_last, S_sn_last, S_sp_full = self._encode(
            item_seq, behavior_seq, seq_len
        )

        # Main next-item CE prediction loss
        logits = torch.matmul(S_sp_last, self.item_embedding.weight.transpose(0, 1))  # [B, n+1]
        pred_loss = self.loss_fct(logits, target)

        # Noise-decoupling contrastive losses
        # Hard stage:  denoised (S_hp) > original (S_ebm) > noise (S_hn)
        cl_hard = self._contrastive_loss(S_hp_last, S_ebm_last, S_hn_last, target)
        # Soft stage:  denoised (S_sp) > hard-denoised (S_hp) > soft-noise (S_sn)
        cl_soft = self._contrastive_loss(S_sp_last, S_hp_last, S_sn_last, target)

        # Compactness regularization on the final denoised sequence
        comp_reg = self._compactness_reg(S_sp_full)

        return pred_loss + self.cl_weight * (cl_hard + cl_soft) + self.compactness_weight * comp_reg

    def predict(self, interaction: dict) -> torch.Tensor:
        item_seq     = interaction['inputs']
        behavior_seq = interaction['behaviors']
        seq_len      = interaction['seq_len']
        test_item    = interaction['target']
        S = self.forward(item_seq, behavior_seq, seq_len)  # [B, d]
        return (S * self.item_embedding(test_item)).sum(-1)  # [B]

    def sample_sort_predict(self, interaction: dict) -> torch.Tensor:
        item_seq     = interaction['inputs']
        behavior_seq = interaction['behaviors']
        seq_len      = interaction['seq_len']
        test_set     = interaction['all_item']   # [B, n_samples]
        S = self.forward(item_seq, behavior_seq, seq_len)       # [B, d]
        test_emb = self.item_embedding(test_set)                # [B, n_samples, d]
        return torch.bmm(test_emb, S.unsqueeze(-1)).squeeze(-1) # [B, n_samples]

    def full_sort_predict(self, interaction: dict) -> torch.Tensor:
        item_seq     = interaction['inputs']
        behavior_seq = interaction['behaviors']
        seq_len      = interaction['seq_len']
        S = self.forward(item_seq, behavior_seq, seq_len)                     # [B, d]
        return torch.matmul(S, self.item_embedding.weight.transpose(0, 1))   # [B, n+1]
