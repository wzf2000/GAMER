import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from SeqRec.models.discriminative.DSIN.config import DSINConfig
from SeqRec.modules.model_base.seq_model import SeqModel


class TargetAttention(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embedding_size * 4, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        interests: torch.Tensor,
        target: torch.Tensor,
        session_mask: torch.Tensor,
    ) -> torch.Tensor:
        expanded_target = target[:, None, :].expand_as(interests)
        features = torch.cat(
            [
                interests,
                expanded_target,
                interests * expanded_target,
                interests - expanded_target,
            ],
            dim=-1,
        )
        scores = self.scorer(features).squeeze(-1)
        scores = scores.masked_fill(~session_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(interests * weights[:, :, None], dim=1)


class DSIN(SeqModel):
    """Item-only Deep Session Interest Network for binary CVR ranking."""

    def __init__(self, config: DSINConfig, n_items: int, max_his_len: int, n_behaviors: int, **kwargs):
        super().__init__(config, n_items)
        if config.embedding_size % config.n_heads != 0:
            raise ValueError(
                f"embedding_size ({config.embedding_size}) must be divisible by n_heads ({config.n_heads})."
            )
        self.embedding_size = config.embedding_size
        self.n_heads = config.n_heads
        self.lstm_hidden_size = config.lstm_hidden_size
        self.attention_hidden_size = config.attention_hidden_size
        self.mlp_hidden_sizes = config.mlp_hidden_sizes
        self.max_sessions = config.max_sessions
        self.dropout_prob = config.dropout
        self.initializer_range = config.initializer_range
        self.max_seq_length = max_his_len
        self.n_behaviors = n_behaviors
        self._init(config.loss_type)

    def _define_parameters(self):
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.intra_session_position_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)
        self.intra_session_attention = nn.MultiheadAttention(
            self.embedding_size,
            self.n_heads,
            dropout=self.dropout_prob,
            batch_first=True,
        )
        self.intra_session_norm = nn.LayerNorm(self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.session_lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.evolved_projection = nn.Linear(self.lstm_hidden_size * 2, self.embedding_size)
        self.raw_attention = TargetAttention(self.embedding_size, self.attention_hidden_size)
        self.evolved_attention = TargetAttention(self.embedding_size, self.attention_hidden_size)

        layers = []
        in_size = self.embedding_size * 5
        for hidden_size in self.mlp_hidden_sizes:
            layers.extend(
                [
                    nn.Linear(in_size, hidden_size),
                    nn.PReLU(),
                    nn.Dropout(self.dropout_prob),
                ]
            )
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))
        self.mlp = nn.Sequential(*layers)

    def _define_loss(self, loss_type: str):
        if loss_type != "BCE":
            raise NotImplementedError("DSIN only supports BCE loss.")
        self.loss_type = loss_type
        self.loss_fct = nn.BCEWithLogitsLoss()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _extract_session_interests(
        self,
        item_seq: torch.Tensor,
        session_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = item_seq.ne(0)
        if session_ids is None:
            session_ids = torch.zeros_like(item_seq)

        previous_valid = torch.cat([torch.zeros_like(valid[:, :1]), valid[:, :-1]], dim=1)
        previous_session = torch.cat([session_ids[:, :1], session_ids[:, :-1]], dim=1)
        session_start = valid & (~previous_valid | session_ids.ne(previous_session))
        session_index = session_start.long().cumsum(dim=1) - 1
        session_count = session_start.sum(dim=1)

        first_kept_session = (session_count - self.max_sessions).clamp(min=0)
        compact_session_index = session_index - first_kept_session[:, None]
        active = (
            valid
            & compact_session_index.ge(0)
            & compact_session_index.lt(self.max_sessions)
        )

        positions = torch.arange(item_seq.size(1), device=item_seq.device)
        positions = positions.unsqueeze(0).expand_as(item_seq)
        start_positions = torch.where(session_start, positions, torch.zeros_like(positions))
        intra_session_positions = positions - torch.cummax(start_positions, dim=1).values

        history_emb = (
            self.item_embedding(item_seq)
            + self.intra_session_position_embedding(intra_session_positions)
        )
        history_emb = history_emb * active[:, :, None]

        different_session = session_index[:, :, None].ne(session_index[:, None, :])
        attention_mask = (
            active[:, :, None]
            & active[:, None, :]
            & different_session
        )
        attention_mask = attention_mask[:, None, :, :].expand(
            -1, self.n_heads, -1, -1
        )
        attention_mask = attention_mask.reshape(
            item_seq.size(0) * self.n_heads,
            item_seq.size(1),
            item_seq.size(1),
        )
        attended, _ = self.intra_session_attention(
            history_emb,
            history_emb,
            history_emb,
            key_padding_mask=~active,
            attn_mask=attention_mask,
            need_weights=False,
        )
        history_states = self.intra_session_norm(history_emb + self.dropout(attended))
        history_states = history_states * active[:, :, None]

        scatter_index = compact_session_index.clamp(0, self.max_sessions - 1)
        session_sums = history_states.new_zeros(
            item_seq.size(0), self.max_sessions, self.embedding_size
        )
        session_sums.scatter_add_(
            1,
            scatter_index[:, :, None].expand(-1, -1, self.embedding_size),
            history_states,
        )
        session_sizes = history_states.new_zeros(item_seq.size(0), self.max_sessions)
        session_sizes.scatter_add_(1, scatter_index, active.to(history_states.dtype))
        session_interests = session_sums / session_sizes.clamp(min=1)[:, :, None]

        kept_session_count = session_count.clamp(max=self.max_sessions)
        session_mask = (
            torch.arange(self.max_sessions, device=item_seq.device)[None, :]
            < kept_session_count[:, None]
        )
        return session_interests, session_mask

    def forward(
        self,
        item_seq: torch.Tensor,
        behavior_seq: torch.Tensor,
        candidate_item: torch.Tensor,
        session_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        session_interests, session_mask = self._extract_session_interests(
            item_seq,
            session_ids,
        )
        session_lengths = session_mask.sum(dim=1).clamp(min=1)
        packed_interests = pack_padded_sequence(
            session_interests,
            session_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_evolved, _ = self.session_lstm(packed_interests)
        evolved_interests, _ = pad_packed_sequence(
            packed_evolved,
            batch_first=True,
            total_length=self.max_sessions,
        )
        evolved_interests = self.evolved_projection(evolved_interests)

        candidate_emb = self.item_embedding(candidate_item)
        raw_interest = self.raw_attention(session_interests, candidate_emb, session_mask)
        evolved_interest = self.evolved_attention(
            evolved_interests,
            candidate_emb,
            session_mask,
        )
        features = torch.cat(
            [
                candidate_emb,
                raw_interest,
                evolved_interest,
                candidate_emb * raw_interest,
                candidate_emb * evolved_interest,
            ],
            dim=-1,
        )
        return self.mlp(features).squeeze(-1)

    def calculate_loss(self, interaction: dict) -> torch.Tensor:
        logits = self.forward(
            interaction["inputs"],
            interaction["behaviors"],
            interaction["candidate_item"],
            interaction.get("session_ids"),
        )
        return self.loss_fct(logits, interaction["label"])

    def predict(self, interaction: dict) -> torch.Tensor:
        return self.forward(
            interaction["inputs"],
            interaction["behaviors"],
            interaction["candidate_item"],
            interaction.get("session_ids"),
        )
