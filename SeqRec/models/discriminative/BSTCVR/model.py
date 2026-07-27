import torch
from torch import nn

from SeqRec.models.discriminative.BSTCVR.config import BSTCVRConfig
from SeqRec.modules.layers.transformer import TransformerEncoder, TransformerEncoderLayer
from SeqRec.modules.model_base.seq_model import SeqModel


class BSTCVR(SeqModel):
    def __init__(self, config: BSTCVRConfig, n_items: int, max_his_len: int, n_behaviors: int, **kwargs):
        super(BSTCVR, self).__init__(config, n_items)
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size
        self.inner_size = config.inner_size
        self.dropout_prob = config.dropout_prob
        self.hidden_act = config.hidden_act
        self.layer_norm_eps = config.layer_norm_eps
        self.initializer_range = config.initializer_range
        self.max_seq_length = max_his_len + 1
        self.n_behaviors = n_behaviors
        self._init(config.loss_type)

    def _define_parameters(self):
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.dropout_prob,
            activation=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.trm_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.n_layers)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.output = nn.Linear(self.hidden_size, 1)

    def _define_loss(self, loss_type: str):
        if loss_type != "BCE":
            raise NotImplementedError("BSTCVR only supports BCE loss.")
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

    def forward(self, item_seq: torch.Tensor, behavior_seq: torch.Tensor, candidate_item: torch.Tensor) -> torch.Tensor:
        target_seq = torch.cat([item_seq, candidate_item[:, None]], dim=1)
        position_ids = torch.arange(target_seq.size(1), dtype=torch.long, device=target_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(target_seq)
        input_emb = self.item_embedding(target_seq) + self.position_embedding(position_ids)
        input_emb = self.dropout(self.LayerNorm(input_emb))

        trm_output = self.trm_encoder(input_emb, self.get_attention_mask(target_seq, bidirectional=True))
        return self.output(trm_output[:, -1]).squeeze(-1)

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

