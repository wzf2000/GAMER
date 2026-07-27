import torch
from torch import nn

from SeqRec.models.discriminative.MeanPooling.config import MeanPoolingConfig
from SeqRec.modules.model_base.seq_model import SeqModel


class MeanPooling(SeqModel):
    def __init__(self, config: MeanPoolingConfig, n_items: int, max_his_len: int, n_behaviors: int, **kwargs):
        super(MeanPooling, self).__init__(config, n_items)
        self.embedding_size = config.embedding_size
        self.initializer_range = config.initializer_range
        self.max_seq_length = max_his_len
        self.n_behaviors = n_behaviors
        self._init(config.loss_type)

    def _define_parameters(self):
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)

    def _define_loss(self, loss_type: str):
        if loss_type != "BCE":
            raise NotImplementedError("MeanPooling only supports BCE loss.")
        self.loss_type = loss_type
        self.loss_fct = nn.BCEWithLogitsLoss()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)

    def forward(self, item_seq: torch.Tensor, behavior_seq: torch.Tensor, candidate_item: torch.Tensor) -> torch.Tensor:
        history_emb = self.item_embedding(item_seq)
        mask = item_seq.ne(0).to(history_emb.dtype)
        user_interest = (history_emb * mask[:, :, None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        candidate_emb = self.item_embedding(candidate_item)
        return torch.sum(user_interest * candidate_emb, dim=-1)

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

