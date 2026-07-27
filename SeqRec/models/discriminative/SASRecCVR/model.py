import torch
from torch import nn

from SeqRec.models.discriminative.SASRec.model import SASRec
from SeqRec.models.discriminative.SASRecCVR.config import SASRecCVRConfig


class SASRecCVR(SASRec):
    def __init__(self, config: SASRecCVRConfig, n_items: int, max_his_len: int, n_behaviors: int, **kwargs):
        super(SASRecCVR, self).__init__(config, n_items=n_items, max_his_len=max_his_len)
        self.n_behaviors = n_behaviors

    def _define_loss(self, loss_type: str):
        if loss_type != "BCE":
            raise NotImplementedError("SASRecCVR only supports BCE loss.")
        self.loss_type = loss_type
        self.loss_fct = nn.BCEWithLogitsLoss()

    def _score_candidate(self, interaction: dict) -> torch.Tensor:
        seq_output = self.forward(interaction["inputs"], interaction["seq_len"])
        candidate_emb = self.item_embedding(interaction["candidate_item"])
        return torch.sum(seq_output * candidate_emb, dim=-1)

    def calculate_loss(self, interaction: dict) -> torch.Tensor:
        return self.loss_fct(self._score_candidate(interaction), interaction["label"])

    def predict(self, interaction: dict) -> torch.Tensor:
        return self._score_candidate(interaction)

