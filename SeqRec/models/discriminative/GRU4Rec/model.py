import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

from SeqRec.models.discriminative.GRU4Rec.config import GRU4RecConfig
from SeqRec.modules.model_base.seq_model import SeqModel


# implementation reference: https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/gru4rec.py
class GRU4Rec(SeqModel):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article, we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config: GRU4RecConfig, n_items: int, **kwargs):
        super(GRU4Rec, self).__init__(config, n_items)

        # load parameters info
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.n_layers
        self.dropout_prob = config.dropout

        self._init(config.loss_type)

    def _define_parameters(self):
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output
