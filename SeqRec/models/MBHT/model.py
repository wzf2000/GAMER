import torch
import random
from torch import nn
from torch.nn import functional as F

from SeqRec.models.MBHT.config import MBHTConfig
from SeqRec.modules.model_base.seq_model import SeqModel
from SeqRec.modules.layers import HGNN, MultiScaleTransformerEncoderLayer, TransformerEncoder


def sim(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.matmul(z1, z2.permute(0, 2, 1))


# implementation reference: https://github.com/yuh-yang/MBHT-KDD22/blob/main/recbole/model/sequential_recommender/mbht.py
class MBHT(SeqModel):
    def __init__(self, config: MBHTConfig, n_items: int, max_his_len: int, target_behavior_id: int, n_behaviors: int, **kwargs):
        super(MBHT, self).__init__(config, n_items)

        # load parameters info
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size  # same as embedding_size
        self.inner_size = config.inner_size  # the dimensionality in feed-forward layer
        self.dropout_prob = config.dropout_prob
        self.hidden_act = config.hidden_act
        self.layer_norm_eps = config.layer_norm_eps
        self.mask_ratio = config.mask_ratio
        self.initializer_range = config.initializer_range
        self.hglen = config.hyper_len
        self.enable_hg = config.enable_hg
        self.enable_ms = config.enable_ms
        self.scales = config.scales
        self.max_seq_length = max_his_len
        self.target_type = target_behavior_id
        self.n_behaviors = n_behaviors

        # load dataset info
        self.mask_token = self.n_items + 1  # add mask token
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        assert config.loss_type == 'CE'
        self._init(config.loss_type)

    def _define_parameters(self):
        self.type_embedding = nn.Embedding(
            self.n_behaviors, self.hidden_size, padding_idx=0
        )
        self.item_embedding = nn.Embedding(
            self.n_items + 2, self.hidden_size, padding_idx=0
        )  # mask token add 1
        self.position_embedding = nn.Embedding(
            self.max_seq_length + 1, self.hidden_size
        )  # add mask_token at the last
        if self.enable_ms:
            encoder_layer = MultiScaleTransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.n_heads,
                dim_feedforward=self.inner_size,
                dropout=self.dropout_prob,
                activation=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
                multiscale=True,
                scales=self.scales,
                max_len=self.max_seq_length + 1,
            )
            self.trm_encoder = TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=self.n_layers
            )
        else:
            encoder_layer = MultiScaleTransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.n_heads,
                dim_feedforward=self.inner_size,
                dropout=self.dropout_prob,
                activation=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
                multiscale=False,
            )
            self.trm_encoder = TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=self.n_layers
            )
        self.hgnn_layer = HGNN(self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.hg_type_embedding = nn.Embedding(self.n_behaviors, self.hidden_size, padding_idx=0)
        self.metric_w1 = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.metric_w2 = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.gating_weight = nn.Parameter(
            torch.Tensor(self.hidden_size, self.hidden_size)
        )
        self.gating_bias = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.attn_weights = nn.Parameter(
            torch.Tensor(self.hidden_size, self.hidden_size)
        )
        self.attn = nn.Parameter(torch.Tensor(1, self.hidden_size))
        nn.init.normal_(self.attn, std=0.02)
        nn.init.normal_(self.attn_weights, std=0.02)
        nn.init.normal_(self.gating_weight, std=0.02)
        nn.init.normal_(self.metric_w1, std=0.02)
        nn.init.normal_(self.metric_w2, std=0.02)

        self.sw_before = 10
        self.sw_follow = 6

        self.hypergraphs = dict()

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

    def get_attention_mask(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Generate bidirectional attention mask for multi-scale attention."""
        if self.enable_ms:
            attention_mask = (item_seq > 0).long()
            extended_attention_mask = attention_mask.unsqueeze(1)
            return extended_attention_mask
        else:
            """Generate bidirectional attention mask for multi-head attention."""
            attention_mask = (item_seq > 0).long()
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
                2
            )  # torch.int64
            # bidirectional mask
            dtype = next(self.parameters()).dtype
            extended_attention_mask = extended_attention_mask.to(
                dtype=dtype
            )  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
            return extended_attention_mask

    def _padding_sequence(self, sequence: list[int], max_length: int) -> list[int]:
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def _right_padding(self, seq: torch.Tensor, max_length: int) -> torch.Tensor:
        assert seq.size(1) <= max_length
        pad_len = max_length - seq.size(1)
        padding = torch.zeros(seq.size(0), pad_len, dtype=torch.long, device=seq.device)
        seq = torch.cat((seq, padding), dim=1)
        return seq

    def reconstruct_train_data(self, item_seq: torch.Tensor, type_seq: torch.Tensor, last_target: torch.Tensor, last_type: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mask item sequence for training.
        """
        item_seq = self._right_padding(item_seq, self.max_seq_length)
        type_seq = self._right_padding(type_seq, self.max_seq_length)
        last_target = last_target.tolist()
        last_type = last_type.tolist()
        device = item_seq.device
        batch_size = item_seq.size(0)

        zero_padding = torch.zeros(
            item_seq.size(0), dtype=torch.long, device=item_seq.device
        )
        item_seq = torch.cat(
            (item_seq, zero_padding.unsqueeze(-1)), dim=-1
        )  # [B, max_len + 1]
        type_seq = torch.cat((type_seq, zero_padding.unsqueeze(-1)), dim=-1)
        n_objs = (torch.count_nonzero(item_seq, dim=1) + 1).tolist()
        for batch_id in range(batch_size):
            n_obj = n_objs[batch_id]
            item_seq[batch_id][n_obj - 1] = last_target[batch_id]
            type_seq[batch_id][n_obj - 1] = last_type[batch_id]

        sequence_instances = item_seq.cpu().numpy().tolist()
        type_instances = type_seq.cpu().numpy().tolist()

        # Masked Item Prediction
        # [B, Len]
        masked_item_sequence = []
        pos_items = []
        masked_index = []

        for instance_idx, instance in enumerate(sequence_instances):
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            pos_item = []
            index_ids = []
            for index_id, item in enumerate(instance):
                # padding is 0, the sequence is end
                if index_id == n_objs[instance_idx] - 1:
                    pos_item.append(item)
                    masked_sequence[index_id] = self.mask_token
                    type_instances[instance_idx][index_id] = 0
                    index_ids.append(index_id)
                    break
                prob = random.random()
                if prob < self.mask_ratio:
                    pos_item.append(item)
                    masked_sequence[index_id] = self.mask_token
                    type_instances[instance_idx][index_id] = 0
                    index_ids.append(index_id)

            masked_item_sequence.append(masked_sequence)
            pos_items.append(self._padding_sequence(pos_item, self.mask_item_length))
            masked_index.append(
                self._padding_sequence(index_ids, self.mask_item_length)
            )

        # [B, Len]
        masked_item_sequence = torch.tensor(
            masked_item_sequence, dtype=torch.long, device=device
        ).view(batch_size, -1)
        # [B, mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(
            batch_size, -1
        )
        # [B, mask_len]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(
            batch_size, -1
        )
        type_instances = torch.tensor(
            type_instances, dtype=torch.long, device=device
        ).view(batch_size, -1)
        return masked_item_sequence, pos_items, masked_index, type_instances

    def reconstruct_test_data(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor, item_type: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        item_seq = self._right_padding(item_seq, self.max_seq_length)
        item_type = self._right_padding(item_type, self.max_seq_length)
        padding = torch.zeros(
            item_seq.size(0), dtype=torch.long, device=item_seq.device
        )  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B, max_len + 1]
        item_type = torch.cat((item_type, padding.unsqueeze(-1)), dim=-1)
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        return item_seq, item_type

    def forward(self, item_seq: torch.Tensor, type_seq: torch.Tensor, mask_positions_nums: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding: torch.Tensor = self.position_embedding(position_ids)
        type_embedding: torch.Tensor = self.type_embedding(type_seq)
        item_emb: torch.Tensor = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding + type_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask
        )
        output = trm_output

        if self.enable_hg:
            x_raw = item_emb  # [B, l, H]
            x_raw = x_raw * torch.sigmoid(
                x_raw.matmul(self.gating_weight) + self.gating_bias
            )  # [B, l, H]
            x_m = torch.stack((self.metric_w1 * x_raw, self.metric_w2 * x_raw)).mean(0)  # [B, l, H]
            item_sim = sim(x_m, x_m)  # [B, l, l]
            item_sim[item_sim < 0] = 0.01

            Gs = self.build_Gs_unique(item_seq, item_sim, self.hglen)

            batch_size = item_seq.shape[0]
            seq_len = item_seq.shape[1]
            n_objs = torch.count_nonzero(item_seq, dim=1)
            indexed_embs = list()
            for batch_idx in range(batch_size):
                n_obj = n_objs[batch_idx]
                # l', dim
                indexed_embs.append(x_raw[batch_idx][:n_obj])
            indexed_embs = torch.cat(indexed_embs, dim=0)
            hgnn_embs = self.hgnn_layer(indexed_embs, Gs)
            hgnn_take_start = 0
            hgnn_embs_padded = []
            for batch_idx in range(batch_size):
                n_obj = n_objs[batch_idx]
                embs = hgnn_embs[hgnn_take_start : hgnn_take_start + n_obj]
                hgnn_take_start += n_obj
                # l', dim || padding emb -> l, dim
                padding = torch.zeros((seq_len - n_obj, embs.shape[-1])).to(
                    item_seq.device
                )
                embs = torch.cat((embs, padding), dim=0)
                if mask_positions_nums is not None:
                    mask_len = mask_positions_nums[1][batch_idx]
                    poss = mask_positions_nums[0][batch_idx][-mask_len:].tolist()
                    for pos in poss:
                        if pos == 0:
                            continue
                        sliding_window_start = (
                            pos - self.sw_before if pos - self.sw_before > -1 else 0
                        )
                        sliding_window_end = (
                            pos + self.sw_follow
                            if pos + self.sw_follow < n_obj
                            else n_obj - 1
                        )
                        readout = torch.mean(
                            torch.cat(
                                (
                                    embs[sliding_window_start:pos],
                                    embs[pos + 1 : sliding_window_end],
                                ),
                                dim=0,
                            ),
                            dim=0,
                        )
                        embs[pos] = readout
                else:
                    pos = (item_seq[batch_idx] == self.mask_token).nonzero(
                        as_tuple=True
                    )[0][0]
                    sliding_window_start = (
                        pos - self.sw_before if pos - self.sw_before > -1 else 0
                    )
                    embs[pos] = torch.mean(embs[sliding_window_start:pos], dim=0)
                hgnn_embs_padded.append(embs)
            # b, l, dim
            hgnn_embs = torch.stack(hgnn_embs_padded, dim=0)
            # 2, b, l, dim
            mixed_x = torch.stack((output, hgnn_embs), dim=0)
            weights = (
                torch.matmul(mixed_x, self.attn_weights.unsqueeze(0).unsqueeze(0))
                * self.attn
            ).sum(-1)
            # 2, b, l, 1
            score = F.softmax(weights, dim=0).unsqueeze(-1)
            mixed_x = (mixed_x * score).sum(0)
            # b, s, n
            assert not torch.isnan(mixed_x).any()
            return mixed_x

        return output  # [B L H]

    def multi_hot_embed(self, masked_index: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask 3 mask 5]

            masked_index: [1, 3]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(
            masked_index.size(0), max_length, device=masked_index.device
        )
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_loss(self, interaction: dict) -> torch.Tensor:
        item_seq = interaction['inputs']
        item_type = interaction["behaviors"]
        last_target = interaction["target"]
        last_type = interaction["behavior"]
        masked_item_seq, pos_items, masked_index, item_type_seq = (
            self.reconstruct_train_data(item_seq, item_type, last_target, last_type)
        )

        mask_nums = torch.count_nonzero(pos_items, dim=1)
        seq_output = self.forward(
            masked_item_seq,
            item_type_seq,
            mask_positions_nums=(masked_index, mask_nums),
        )
        pred_index_map = self.multi_hot_embed(
            masked_index, masked_item_seq.size(-1)
        )  # [B * mask_len, max_len]
        # [B, mask_len] -> [B, mask_len, max_len] multi hot
        pred_index_map = pred_index_map.view(
            masked_index.size(0), masked_index.size(1), -1
        )  # [B, mask_len, max_len]
        # [B, mask_len, max_len] * [B, max_len, H] -> [B, mask_len, H]
        # only calculate loss for masked position
        seq_output = torch.bmm(pred_index_map, seq_output)  # [B, mask_len, H]

        test_item_emb = self.item_embedding.weight  # [item_num, H]
        logits = torch.matmul(
            seq_output, test_item_emb.transpose(0, 1)
        )  # [B, mask_len, item_num]
        targets = (masked_index > 0).float().view(-1)  # [B * mask_len]

        loss = torch.sum(
            self.loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1))
            * targets
        ) / torch.sum(targets)
        return loss

    def sample_sort_predict(self, interaction: dict) -> torch.Tensor:
        item_seq = interaction["inputs"]
        type_seq = interaction["behaviors"]
        test_set = interaction['all_item']
        item_seq_len = torch.count_nonzero(item_seq, 1)
        item_seq, type_seq = self.reconstruct_test_data(
            item_seq, item_seq_len, type_seq
        )
        seq_output = self.forward(item_seq, type_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len)  # [B, H]
        test_items_emb = self.item_embedding(test_set)  # [B, sample_items, H]
        scores = torch.matmul(
            test_items_emb, seq_output[..., None]
        )[..., 0]  # [B, sample_items]
        return scores

    def full_sort_predict(self, interaction: dict) -> torch.Tensor:
        item_seq = interaction["inputs"]
        type_seq = interaction["behaviors"]
        item_seq_len = torch.count_nonzero(item_seq, 1)
        item_seq, type_seq = self.reconstruct_test_data(
            item_seq, item_seq_len, type_seq
        )
        seq_output = self.forward(item_seq, type_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len)  # [B, H]
        test_items_emb = self.item_embedding.weight[
            : self.n_items + 1
        ]  # delete masked token
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, item_num]
        return scores

    def build_Gs_unique(self, seqs: torch.Tensor, item_sim: torch.Tensor, group_len: int) -> torch.Tensor:
        # seqs: [B, l]
        # item_sim: [B, l, l]
        Gs = []
        n_objs = torch.count_nonzero(seqs, dim=1)  # [B]
        for batch_idx in range(seqs.shape[0]):
            seq = seqs[batch_idx]  # [l]
            n_obj = n_objs[batch_idx].item()
            seq = seq[:n_obj]  # [l']
            unique, counts = torch.unique(seq, return_counts=True)
            unique: torch.Tensor
            n_unique = len(unique)

            seq_item_sim = item_sim[batch_idx, :n_obj, :n_obj]  # [l', l']
            metrics, sim_items = torch.topk(seq_item_sim, min(group_len, n_obj), sorted=False)  # [l', group_len], [l', group_len]
            # map indices to item tokens
            sim_items = seq[sim_items]  # [l', group_len]
            row_idx, masked_pos = torch.nonzero(
                sim_items == self.mask_token, as_tuple=True
            )  # find the masked position
            sim_items[row_idx, masked_pos] = seq[row_idx]  # replace with itself
            metrics[row_idx, masked_pos] = 1.0  # self-similarity is 1.0

            multibeh_group = unique[(counts > 1) & (unique != self.mask_token)]
            n_multibeh = len(multibeh_group)

            n_edge = n_unique + n_multibeh
            # hyper graph: n_obj, n_edge
            H = torch.zeros((n_obj, n_edge), device=metrics.device)  # [l', n_unique + n_multibeh]

            # Item-wise Semantic Dependency Hypergraph
            normal_item_indexes = torch.nonzero(
                (seq != self.mask_token), as_tuple=True
            )[0]  # [l'']
            normal_mask = (seq != self.mask_token)  # [l']
            # [l', group_len, 1] == [1, 1, n_unique] -> [l', group_len, n_unique] argmax(dim=-1) -> [l', group_len]
            unique_idx_map = (sim_items[:, :, None] == unique[None, None, :]).long().argmax(dim=-1)  # [l', group_len]
            row_indices = normal_item_indexes.repeat_interleave(metrics.size(1))  # [l * group_len]
            col_indices = unique_idx_map[normal_mask].flatten()  # [l * group_len]
            H[row_indices, col_indices] = metrics[normal_mask].flatten()

            # Self-loop
            ego_mask = (seq[:, None] == unique[None, :])  # [l', n_unique]
            ego_idxs = torch.nonzero(ego_mask, as_tuple=True)
            H[ego_idxs] = 1.0  # self-loop

            if n_multibeh > 0:
                # Item-wise Multi-Behavior Dependency Hypergraph
                seq_multibeh_mask = (seq[:, None] == multibeh_group[None, :]).long()  # [l', n_multibeh]
                seq_multibeh_idx = seq_multibeh_mask.argmax(dim=1)  # [l']
                valid_mask = seq_multibeh_mask.any(dim=1)  # [l']
                H[valid_mask, n_unique + seq_multibeh_idx[valid_mask]] = 1.0

            DV = torch.sum(H, dim=1)  # [n_obj]
            DE = torch.sum(H, dim=0)  # [n_edge]
            invDE = torch.diag(torch.pow(DE, -1))  # [n_edge, n_edge]
            invDV = torch.diag(torch.pow(DV, -1))  # [n_obj, n_obj]
            HT = H.t()  # [n_edge, n_obj]
            G = invDV @ H @ invDE @ HT  # [n_obj, n_obj]
            assert not torch.isnan(G).any()
            Gs.append(G)

        Gs_block_diag = torch.block_diag(*Gs)  # [sum(n_obj), sum(n_obj)]
        return Gs_block_diag
