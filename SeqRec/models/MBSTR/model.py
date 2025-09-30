import torch
from torch import nn
from typing import overload

from SeqRec.models.MBSTR.config import MBSTRConfig
from SeqRec.modules.model_base.seq_model import SeqModel
from SeqRec.modules.layers.transformer import TransformerEncoder, DotProductPredictionHead
from SeqRec.modules.layers.mbs_transformer import MBSTransformerEncoderLayer, CGCDotProductPredictionHead, MBSMultiHeadAttention


class MBSTR(SeqModel):
    def __init__(self, config: MBSTRConfig, n_items: int, max_his_len: int, n_behaviors: int, **kwargs):
        super(MBSTR, self).__init__(config, n_items)

        # load parameters info
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size
        self.inner_size = config.inner_size
        self.dropout_prob = config.dropout_prob
        self.hidden_act = config.hidden_act
        self.layer_norm_eps = config.layer_norm_eps
        self.initializer_range = config.initializer_range
        self.mask_ratio = config.mask_ratio

        self.num_buckets = config.num_buckets
        self.max_distance = config.max_distance
        self.behavior_head = config.behavior_head
        self.behavior_attention = config.behavior_attention
        self.behavior_moe = config.behavior_moe
        self.behavior_position_bias = config.behavior_position_bias

        self.n_shared_experts = config.n_shared_experts
        self.n_specific_experts = config.n_specific_experts

        self.max_seq_length = max_his_len
        self.n_behaviors = n_behaviors

        self.mask_token = self.n_items + 1  # add mask token

        assert config.loss_type == 'CE', "Only support CE loss now"
        self._init(config.loss_type)

    def _define_parameters(self):
        self.item_embedding = nn.Embedding(
            self.n_items + 2, self.hidden_size, padding_idx=0
        )  # 0: <PAD>, n_items + 1: <MASK>
        self.dropout = nn.Dropout(self.dropout_prob)
        encoder_layer = MBSTransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            n_behaviors=self.n_behaviors,
            dim_feedforward=self.inner_size,
            dropout=self.dropout_prob,
            activation=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
            behavior_attention=self.behavior_attention,
            behavior_moe=self.behavior_moe,
            behavior_position_bias=self.behavior_position_bias,
        )
        self.trm_encoder = TransformerEncoder(encoder_layer, self.n_layers)
        if self.behavior_head:
            self.head = CGCDotProductPredictionHead(
                d_model=self.hidden_size,
                n_items=self.n_items,
                token_embeddings=self.item_embedding,
                layer_norm_eps=self.layer_norm_eps,
                n_behaviors=self.n_behaviors,
                n_shared_experts=self.n_shared_experts,
                n_specific_experts=self.n_specific_experts,
            )
        else:
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
        if isinstance(module, MBSMultiHeadAttention) and not isinstance(module.query, nn.Linear):
            module.query.data.normal_(mean=0.0, std=self.initializer_range)
            module.key.data.normal_(mean=0.0, std=self.initializer_range)
            module.value.data.normal_(mean=0.0, std=self.initializer_range)

    def reconstruct_train_data(self, item_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mask item sequence for training.
        """
        mask = torch.rand_like(item_seq, dtype=torch.float) < self.mask_ratio
        mask &= item_seq != 0  # do not mask padding items
        labels = item_seq * mask  # [B, L], 0 is for non-masked positions
        masked_item_seq = item_seq * (~mask) + self.mask_token * mask
        return masked_item_seq, labels

    @overload
    def forward(self, item_seq: torch.Tensor, type_seq: torch.Tensor, labels: torch.Tensor, candidates: None = None) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @overload
    def forward(self, item_seq: torch.Tensor, type_seq: torch.Tensor, labels: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, item_seq: torch.Tensor, type_seq: torch.Tensor, labels: torch.Tensor, candidates: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        item_emb = self.dropout(self.item_embedding(item_seq))
        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        output: torch.Tensor = self.trm_encoder(
            item_emb, extended_attention_mask, type_seq=type_seq
        )  # [B, L, H]
        labels = labels.flatten()  # [B * L]
        valid = labels != 0
        valid_index = valid.nonzero()[:, 0]  # [M]
        output = output.view(-1, output.size(-1))  # [B * L, H]
        valid_output = output[valid_index]  # [M, H]
        valid_type_seq = type_seq.flatten()[valid_index]  # [M]
        valid_labels = labels[valid_index]  # [M]
        valid_logits = self.head(valid_output, type_seq=valid_type_seq, candidates=candidates)  # [M, n_items + 1]
        if candidates is None:
            return valid_logits, valid_labels
        else:
            return valid_logits

    def calculate_loss(self, interaction: dict) -> torch.Tensor:
        item_seq = interaction["inputs"]
        item_type = interaction["behaviors"]
        masked_item_seq, labels = self.reconstruct_train_data(item_seq)
        logits, labels = self.forward(masked_item_seq, item_type, labels=labels, candidates=None)
        loss = self.loss_fct(logits, labels)
        return loss

    def sample_sort_predict(self, interaction: dict) -> torch.Tensor:
        item_seq = interaction["inputs"]
        type_seq = interaction["behaviors"]
        seq_len = interaction["seq_len"]
        labels = torch.zeros_like(item_seq)
        labels[torch.arange(item_seq.size(0)), seq_len - 1] = 1
        labels *= item_seq
        test_set = interaction["all_item"]
        logits = self.forward(item_seq, type_seq, labels=labels, candidates=test_set)
        return logits

    def full_sort_predict(self, interaction: dict) -> torch.Tensor:
        item_seq = interaction["inputs"]
        type_seq = interaction["behaviors"]
        seq_len = interaction["seq_len"]
        labels = torch.zeros_like(item_seq)
        labels[torch.arange(item_seq.size(0)), seq_len - 1] = 1
        labels *= item_seq
        test_set = torch.arange(self.n_items + 1, device=item_seq.device)[None].expand(item_seq.size(0), -1)  # [B, n_items]
        logits = self.forward(item_seq, type_seq, labels=labels, candidates=test_set)
        return logits
