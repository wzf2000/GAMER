import torch
from torch import nn
from typing import overload

from SeqRec.models.BERT4Rec.config import BERT4RecConfig
from SeqRec.modules.model_base.seq_model import SeqModel
from SeqRec.modules.layers.transformer import TransformerEncoder, TransformerEncoderLayer, DotProductPredictionHead


# implementation reference: https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/bert4rec.py
class BERT4Rec(SeqModel):
    def __init__(self, config: BERT4RecConfig, n_items: int, max_his_len: int, **kwargs):
        super(BERT4Rec, self).__init__(config, n_items)

        # load parameters info
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size
        self.inner_size = config.inner_size
        self.dropout_prob = config.dropout_prob
        self.hidden_act = config.hidden_act
        self.layer_norm_eps = config.layer_norm_eps
        self.initializer_range = config.initializer_range
        self.ft_ratio = config.ft_ratio
        self.mask_ratio = config.mask_ratio

        self.max_seq_length = max_his_len
        self.mask_token = self.n_items + 1  # add mask token

        assert config.loss_type == 'CE', "Only support CE loss now"
        self._init(config.loss_type)

    def _define_parameters(self):
        self.item_embedding = nn.Embedding(
            self.n_items + 2, self.hidden_size, padding_idx=0
        )  # 0: <PAD>, n_items + 1: <MASK>
        self.position_embedding = nn.Embedding(
            self.max_seq_length, self.hidden_size
        )
        self.dropout = nn.Dropout(self.dropout_prob)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.dropout_prob,
            activation=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.trm_encoder = TransformerEncoder(encoder_layer, self.n_layers)
        self.output_ffn = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_gelu = nn.GELU()
        self.output_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.output_bias = nn.Parameter(torch.zeros(self.n_items + 1))
        self.head = DotProductPredictionHead(
            d_model=self.hidden_size,
            n_items=self.n_items,
            token_embeddings=self.item_embedding,
        )

    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def reconstruct_train_data(self, item_seq: torch.Tensor, seq_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mask item sequence for training.
        """
        # decide which sequences are for fine-tuning
        batch_mask = torch.rand(item_seq.size(0), 1, device=item_seq.device) < self.ft_ratio  # [B, 1]
        # mask the last item with batch_mask == True
        ft_mask = torch.zeros_like(item_seq, dtype=torch.bool)  # [B, L]
        ft_mask[torch.arange(item_seq.size(0)), torch.clamp_max(seq_len, self.max_seq_length - 1)] = True
        ft_mask &= batch_mask  # only mask the last item of fine-tuning sequences
        mask = torch.rand_like(item_seq, dtype=torch.float) < self.mask_ratio  # [B, L]
        mask &= item_seq != 0  # do not mask padding items
        mask &= (~batch_mask)  # do not mask the fine-tuning sequence
        mask |= ft_mask  # mask the last item of fine-tuning sequences
        labels = item_seq * mask  # [B, L], 0 is for non-masked positions
        masked_item_seq = item_seq * (~mask) + self.mask_token * mask
        return masked_item_seq, labels

    @overload
    def forward(self, item_seq: torch.Tensor, labels: torch.Tensor, candidates: None = None) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @overload
    def forward(self, item_seq: torch.Tensor, labels: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, item_seq: torch.Tensor, labels: torch.Tensor, candidates: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding: torch.Tensor = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        trm_output: torch.Tensor = self.trm_encoder(
            input_emb, extended_attention_mask
        )  # [B, L, H]
        ffn_output = self.output_ffn(trm_output)
        ffn_output = self.output_gelu(ffn_output)
        output = self.output_ln(ffn_output)  # [B, L, H]

        labels = labels.flatten()  # [B * L]
        valid = labels != 0
        valid_index = valid.nonzero()[:, 0]  # [M]
        output = output.view(-1, output.size(-1))  # [B * L, H]
        valid_output = output[valid_index]  # [M, H]
        valid_labels = labels[valid_index]  # [M]
        valid_logits = self.head(valid_output, candidates=candidates)  # [M, n_items + 1]
        if candidates is None:
            return valid_logits, valid_labels
        else:
            return valid_logits

    def calculate_loss(self, interaction: dict) -> torch.Tensor:
        item_seq = interaction["inputs"]
        seq_len = interaction["seq_len"]
        masked_item_seq, labels = self.reconstruct_train_data(item_seq, seq_len)
        logits, labels = self.forward(masked_item_seq, labels=labels, candidates=None)
        loss = self.loss_fct(logits, labels)
        return loss

    def sample_sort_predict(self, interaction: dict) -> torch.Tensor:
        item_seq = interaction["inputs"]
        seq_len = interaction["seq_len"]
        labels = torch.zeros_like(item_seq)
        labels[torch.arange(item_seq.size(0)), seq_len - 1] = 1
        labels *= item_seq
        test_set = interaction["all_item"]
        logits = self.forward(item_seq, labels=labels, candidates=test_set)
        return logits

    def full_sort_predict(self, interaction: dict) -> torch.Tensor:
        item_seq = interaction["inputs"]
        seq_len = interaction["seq_len"]
        labels = torch.zeros_like(item_seq)
        labels[torch.arange(item_seq.size(0)), seq_len - 1] = 1
        labels *= item_seq
        test_set = torch.arange(self.n_items + 1, device=item_seq.device)[None].expand(item_seq.size(0), -1)  # [B, n_items]
        logits = self.forward(item_seq, labels=labels, candidates=test_set)
        return logits
