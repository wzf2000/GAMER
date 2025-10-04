import torch
from torch import nn
from typing import overload

from SeqRec.models.discriminative.PBAT.config import PBATConfig
from SeqRec.modules.model_base.seq_model import SeqModel
from SeqRec.modules.layers.transformer import TransformerEncoder
from SeqRec.modules.layers.pbat import SimpleEmbedding, PBATLayer, WassersteinPredictionHead, SAGP, wasserstein_distance_matmul


class PBAT(SeqModel):
    def __init__(self, config: PBATConfig, n_items: int, n_users: int, max_his_len: int, n_behaviors: int, **kwargs):
        super(PBAT, self).__init__(config, n_items)

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

        self.max_seq_length = max_his_len
        self.n_behaviors = n_behaviors
        self.n_users = n_users

        self.mask_token = self.n_items + 1  # add mask token

        assert config.loss_type == 'CE', "Only support CE loss now"
        self._init(config.loss_type)

    def _define_parameters(self):
        self.item_embedding_m = SimpleEmbedding(
            self.n_items + 2, self.hidden_size, dropout=self.dropout_prob, layer_norm_eps=self.layer_norm_eps, padding_idx=0
        )  # 0: <PAD>, n_items + 1: <MASK>
        self.item_embedding_c = SimpleEmbedding(
            self.n_items + 2, self.hidden_size, dropout=self.dropout_prob, layer_norm_eps=self.layer_norm_eps, padding_idx=0
        )  # 0: <PAD>, n_items + 1: <MASK>
        self.type_embedding_m = SimpleEmbedding(
            self.n_behaviors + 1, self.hidden_size, dropout=self.dropout_prob, layer_norm_eps=self.layer_norm_eps, padding_idx=0
        )  # 0: padding behavior
        self.type_embedding_c = SimpleEmbedding(
            self.n_behaviors + 1, self.hidden_size, dropout=self.dropout_prob, layer_norm_eps=self.layer_norm_eps, padding_idx=0
        )  # 0: padding behavior
        self.user_embedding_m = SimpleEmbedding(
            self.n_users + 1, self.hidden_size, dropout=self.dropout_prob, layer_norm_eps=self.layer_norm_eps, padding_idx=0
        )  # 0: padding user
        self.user_embedding_c = SimpleEmbedding(
            self.n_users + 1, self.hidden_size, dropout=self.dropout_prob, layer_norm_eps=self.layer_norm_eps, padding_idx=0
        )  # 0: padding user
        self.position_embedding_m = nn.Embedding(
            self.max_seq_length, self.hidden_size
        )
        self.position_embedding_c = nn.Embedding(
            self.max_seq_length, self.hidden_size
        )
        self.type_relation_embedding_m = SimpleEmbedding(
            self.n_behaviors * self.n_behaviors + 1, self.hidden_size, dropout=self.dropout_prob, layer_norm_eps=self.layer_norm_eps, padding_idx=0
        )  # 0: padding relation
        self.type_relation_embedding_c = SimpleEmbedding(
            self.n_behaviors * self.n_behaviors + 1, self.hidden_size, dropout=self.dropout_prob, layer_norm_eps=self.layer_norm_eps, padding_idx=0
        )  # 0: padding relation
        self.activation = nn.ELU()
        self.Wub = nn.Linear(self.hidden_size, self.hidden_size)
        self.WPub = nn.Linear(self.hidden_size, self.hidden_size)
        encoder_layer = PBATLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            n_behaviors=self.n_behaviors,
            dim_feedforward=self.inner_size,
            dropout=self.dropout_prob,
            activation=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.trm_encoder = TransformerEncoder(encoder_layer, self.n_layers)
        self.head = WassersteinPredictionHead(
            d_model=self.hidden_size,
            n_items=self.n_items,
            token_embeddings_m=self.item_embedding_m.embedding,
            token_embeddings_c=self.item_embedding_c.embedding,
        )

    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

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
    def forward(self, item_seq: torch.Tensor, type_seq: torch.Tensor, user_ids: torch.Tensor, labels: torch.Tensor, candidates: None = None) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @overload
    def forward(self, item_seq: torch.Tensor, type_seq: torch.Tensor, user_ids: torch.Tensor, labels: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, item_seq: torch.Tensor, type_seq: torch.Tensor, user_ids: torch.Tensor, labels: torch.Tensor, candidates: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        bs = item_seq.size(0)
        item_emb_m = self.item_embedding_m(item_seq)
        item_emb_c = self.item_embedding_c(item_seq) + 1
        type_emb_m = self.type_embedding_m(type_seq)
        type_emb_c = self.type_embedding_c(type_seq) + 1
        user_emb_m = self.user_embedding_m(user_ids)
        user_emb_c = self.user_embedding_c(user_ids) + 1
        position_emb_m = self.position_embedding_m(torch.arange(item_seq.size(1), device=item_seq.device)[None, :].expand(item_seq.size(0), -1))
        position_emb_c = self.position_embedding_c(torch.arange(item_seq.size(1), device=item_seq.device)[None, :].expand(item_seq.size(0), -1)) + 1

        behavior_emb_m = self.type_embedding_m(
            torch.arange(self.n_behaviors + 1, device=item_seq.device)[None, :].expand(item_seq.size(0), -1)
        )  # [B, b + 1, H]
        behavior_emb_c = self.type_embedding_c(
            torch.arange(self.n_behaviors + 1, device=item_seq.device)[None, :].expand(item_seq.size(0), -1)
        )  # [B, b + 1, H]
        behavior_emb_c = self.activation(behavior_emb_c) + 1

        P_user_behavior_m, P_user_behavior_c = SAGP(
            user_emb_m[:, None, :],  # [B, 1, H]
            self.Wub(behavior_emb_m),  # [B, b + 1, H]
            user_emb_c[:, None, :],  # [B, 1, H]
            behavior_emb_c,  # [B, b + 1, H]
        )  # [B, b + 1, H], [B, b + 1, H]

        weight_user_behavior = -wasserstein_distance_matmul(
            P_user_behavior_m,  # [B, b + 1, H]
            P_user_behavior_c,  # [B, b + 1, H]
            P_user_behavior_m,  # [B, b + 1, H]
            P_user_behavior_c,  # [B, b + 1, H]
        )  # [B, b + 1, b + 1]

        type_relation_m = torch.zeros(
            bs, self.n_behaviors + 1, self.n_behaviors + 1, self.hidden_size, device=item_seq.device
        )  # [B, b + 1, b + 1, H]
        type_relation_c = torch.ones(
            bs, self.n_behaviors + 1, self.n_behaviors + 1, self.hidden_size, device=item_seq.device
        )
        for i in range(self.n_behaviors):
            for j in range(self.n_behaviors):
                relation_emb_m = self.type_relation_embedding_m(
                    torch.tensor([i * self.n_behaviors + j + 1], device=item_seq.device)
                )  # [1, H]
                type_relation_m[:, i + 1, j + 1, :] = torch.matmul(
                    weight_user_behavior[:, None, i + 1, j + 1],  # [B, 1]
                    relation_emb_m,  # [1, H]
                )  # [B, H]
                relation_emb_c = self.type_relation_embedding_c(
                    torch.tensor([i * self.n_behaviors + j + 1], device=item_seq.device)
                )  # [1, H]
                type_relation_c[:, i + 1, j + 1, :] = torch.matmul(
                    weight_user_behavior[:, None, i + 1, j + 1],  # [B, 1]
                    relation_emb_c,  # [1, H]
                )  # [B, H]
        type_relation_c = self.activation(type_relation_c) + 1

        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = self.trm_encoder(
            (item_emb_m, item_emb_c, torch.empty(0)), extended_attention_mask,
            type_seq=type_seq,
            type_tensor=(type_emb_m, type_emb_c),
            type_relation_tensor=(type_relation_m, type_relation_c),
            position_tensor=(position_emb_m, position_emb_c)
        )  # [B, L, H] * 2, [B, h, L, L]
        output_m, output_c, W_probs = output
        output_m, output_c = SAGP(
            output_m,  # [B, L, H]
            self.WPub(
                P_user_behavior_m[
                    torch.arange(bs, device=item_seq.device)[:, None],  # [B, 1]
                    type_seq,  # [B, L]
                ],  # [B, L, H]
            ),
            output_c,  # [B, L, H]
            P_user_behavior_c[
                torch.arange(bs, device=item_seq.device)[:, None],  # [B, 1]
                type_seq,  # [B, L]
            ],  # [B, L, H]
        )  # [B, L, H], [B, L, H]

        labels = labels.flatten()  # [B * L]
        valid = labels != 0
        valid_index = valid.nonzero()[:, 0]  # [M]
        output_m = output_m.view(-1, output_m.size(-1))  # [B * L, H]
        output_c = output_c.view(-1, output_c.size(-1))  # [B * L, H]
        valid_output_m = output_m[valid_index]  # [M, H]
        valid_output_c = output_c[valid_index]  # [M, H]
        valid_type_seq = type_seq.flatten()[valid_index]  # [M]
        valid_labels = labels[valid_index]  # [M]
        valid_logits = self.head(valid_output_m, valid_output_c, type_seq=valid_type_seq, candidates=candidates)  # [M, n_items + 1]
        if candidates is None:
            return valid_logits, valid_labels
        else:
            return valid_logits

    def calculate_loss(self, interaction: dict) -> torch.Tensor:
        item_seq = interaction["inputs"]
        item_type = interaction["behaviors"]
        user_ids = interaction["uid"]
        masked_item_seq, labels = self.reconstruct_train_data(item_seq)
        logits, labels = self.forward(masked_item_seq, item_type, user_ids, labels=labels, candidates=None)
        loss = self.loss_fct(logits, labels)
        return loss

    def sample_sort_predict(self, interaction: dict) -> torch.Tensor:
        item_seq = interaction["inputs"]
        type_seq = interaction["behaviors"]
        user_ids = interaction["uid"]
        seq_len = interaction["seq_len"]
        labels = torch.zeros_like(item_seq)
        labels[torch.arange(item_seq.size(0)), seq_len - 1] = 1
        labels *= item_seq
        test_set = interaction["all_item"]
        logits = self.forward(item_seq, type_seq, user_ids, labels=labels, candidates=test_set)
        return logits

    def full_sort_predict(self, interaction: dict) -> torch.Tensor:
        item_seq = interaction["inputs"]
        type_seq = interaction["behaviors"]
        user_ids = interaction["uid"]
        seq_len = interaction["seq_len"]
        labels = torch.zeros_like(item_seq)
        labels[torch.arange(item_seq.size(0)), seq_len - 1] = 1
        labels *= item_seq
        test_set = torch.arange(self.n_items + 1, device=item_seq.device)[None].expand(item_seq.size(0), -1)  # [B, n_items]
        logits = self.forward(item_seq, type_seq, user_ids, labels=labels, candidates=test_set)
        return logits
