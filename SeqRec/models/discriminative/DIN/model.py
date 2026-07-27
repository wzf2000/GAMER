import torch
from torch import nn

from SeqRec.models.discriminative.DIN.config import DINConfig
from SeqRec.modules.model_base.seq_model import SeqModel


class DIN(SeqModel):
    def __init__(self, config: DINConfig, n_items: int, max_his_len: int, n_behaviors: int, **kwargs):
        super(DIN, self).__init__(config, n_items)
        self.embedding_size = config.embedding_size
        self.attention_hidden_size = config.attention_hidden_size
        self.mlp_hidden_sizes = config.mlp_hidden_sizes
        self.dropout = config.dropout
        self.initializer_range = config.initializer_range
        self.max_seq_length = max_his_len
        self.n_behaviors = n_behaviors
        self._init(config.loss_type)

    def _define_parameters(self):
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_size * 4, self.attention_hidden_size),
            nn.PReLU(),
            nn.Linear(self.attention_hidden_size, 1),
        )

        layers = []
        in_size = self.embedding_size * 3
        for hidden_size in self.mlp_hidden_sizes:
            layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.PReLU(),
                nn.Dropout(self.dropout),
            ])
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))
        self.mlp = nn.Sequential(*layers)

    def _define_loss(self, loss_type: str):
        if loss_type != "BCE":
            raise NotImplementedError("DIN only supports BCE loss.")
        self.loss_type = loss_type
        self.loss_fct = nn.BCEWithLogitsLoss()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq: torch.Tensor, behavior_seq: torch.Tensor, candidate_item: torch.Tensor) -> torch.Tensor:
        history_emb = self.item_embedding(item_seq)
        candidate_emb = self.item_embedding(candidate_item)
        candidate_expanded = candidate_emb[:, None, :].expand_as(history_emb)

        attention_input = torch.cat(
            [
                history_emb,
                candidate_expanded,
                history_emb * candidate_expanded,
                history_emb - candidate_expanded,
            ],
            dim=-1,
        )
        attention_score = self.attention(attention_input).squeeze(-1)
        attention_score = attention_score.masked_fill(item_seq == 0, torch.finfo(attention_score.dtype).min)
        attention_weight = torch.softmax(attention_score, dim=-1)
        user_interest = torch.sum(history_emb * attention_weight[:, :, None], dim=1)

        logits = self.mlp(torch.cat([user_interest, candidate_emb, user_interest * candidate_emb], dim=-1)).squeeze(-1)
        return logits

    def calculate_loss(self, interaction: dict) -> torch.Tensor:
        logits = self.forward(
            interaction["inputs"],
            interaction["behaviors"],
            interaction["candidate_item"],
        )
        return self.loss_fct(logits, interaction["label"])

    def predict(self, interaction: dict) -> torch.Tensor:
        return self.forward(
            interaction["inputs"],
            interaction["behaviors"],
            interaction["candidate_item"],
        )
