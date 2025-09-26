import torch
from torch import nn

from SeqRec.utils.config import Config
from SeqRec.modules.loss import BPRLoss


class SeqModel(nn.Module):
    r"""Base class for sequential recommendation models.

    Note:
        All the sequential recommendation models should inherit this class.
    """

    def __init__(self, config: Config, n_items: int, **kwargs):
        super(SeqModel, self).__init__()
        self.n_items = n_items
        self.config = config
        self.item_embedding: torch.nn.Embedding

    def _init(self, loss_type: str):
        # define layers and loss
        self._define_parameters()
        self._define_loss(loss_type)

        # parameters initialization
        self.apply(self._init_weights)

    def _define_loss(self, loss_type: str):
        self.loss_type = loss_type
        if loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

    def _define_parameters(self):
        raise NotImplementedError

    def _init_weights(self, module: nn.Module):
        raise NotImplementedError

    def gather_indexes(self, output: torch.Tensor, gather_index: torch.Tensor) -> torch.Tensor:
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq: torch.Tensor, bidirectional: bool = False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def calculate_loss(self, interaction: dict) -> torch.Tensor:
        item_seq = interaction['inputs']
        item_seq_len = interaction['seq_len']
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction['target']
        if self.loss_type == "BPR":
            assert 'neg_item' in interaction, "BPR loss requires neg_item field, please use proper training task with negative sampling."
            neg_items = interaction['neg_item']
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction: dict) -> torch.Tensor:
        item_seq = interaction['inputs']
        item_seq_len = interaction['seq_len']
        test_item = interaction['target']
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction: dict) -> torch.Tensor:
        item_seq = interaction['inputs']
        item_seq_len = interaction['seq_len']
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
