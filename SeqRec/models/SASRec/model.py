import torch
from torch import nn

from SeqRec.models.SASRec.config import SASRecConfig
from SeqRec.modules.model_base.seq_model import SeqModel


# implementation reference: https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/sasrec.py
class SASRec(SeqModel):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config: SASRecConfig, n_items: int, max_his_len: int, **kwargs):
        super(SASRec, self).__init__(config, n_items)

        # load parameters info
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size  # same as embedding_size
        self.inner_size = config.inner_size  # the dimensionality in feed-forward layer
        self.dropout_prob = config.dropout_prob
        self.hidden_act = config.hidden_act
        self.layer_norm_eps = config.layer_norm_eps
        self.initializer_range = config.initializer_range
        self.max_seq_length = max_his_len

        self._init(config.loss_type)

    def _define_parameters(self):
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.dropout_prob,
            activation=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.trm_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.n_layers,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.dropout_prob)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, item_seq != 0, is_causal=True
        )
        output = self.gather_indexes(trm_output, item_seq_len - 1)
        return output  # [B H]
