import torch
from torch import nn
from torch.nn import functional as F

from SeqRec.models.discriminative.HSTUCVR.config import HSTUCVRConfig
from SeqRec.modules.model_base.seq_model import SeqModel


class HSTULayer(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout_prob: float, layer_norm_eps: float):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.input_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.input_projection = nn.Linear(hidden_size, hidden_size * 4)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        gate, query, key, value = self.input_projection(self.input_norm(hidden_states)).chunk(4, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(hidden_states.size(0), hidden_states.size(1), -1)
        gated_context = F.silu(gate) * context
        return hidden_states + self.dropout(self.output_projection(gated_context))


class HSTUCVR(SeqModel):
    def __init__(self, config: HSTUCVRConfig, n_items: int, max_his_len: int, n_behaviors: int, **kwargs):
        super().__init__(config, n_items)
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size
        self.dropout_prob = config.dropout_prob
        self.layer_norm_eps = config.layer_norm_eps
        self.initializer_range = config.initializer_range
        self.use_behavior_embedding = config.use_behavior_embedding
        self.max_seq_length = max_his_len
        self.n_behaviors = n_behaviors
        self._init(config.loss_type)

    def _define_parameters(self):
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        if self.use_behavior_embedding:
            self.behavior_embedding = nn.Embedding(self.n_behaviors + 1, self.hidden_size, padding_idx=0)
        self.layers = nn.ModuleList(
            [
                HSTULayer(
                    hidden_size=self.hidden_size,
                    n_heads=self.n_heads,
                    dropout_prob=self.dropout_prob,
                    layer_norm_eps=self.layer_norm_eps,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.dropout_prob)

    def _define_loss(self, loss_type: str):
        if loss_type != "BCE":
            raise NotImplementedError("HSTUCVR only supports BCE loss.")
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
        seq_len = item_seq.ne(0).sum(dim=1).clamp(min=1)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        input_emb = self.item_embedding(item_seq) + self.position_embedding(position_ids)
        if self.use_behavior_embedding:
            input_emb = input_emb + self.behavior_embedding(behavior_seq)
        hidden_states = self.dropout(self.LayerNorm(input_emb))

        attention_mask = self.get_attention_mask(item_seq)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        user_state = self.gather_indexes(hidden_states, seq_len - 1)
        candidate_emb = self.item_embedding(candidate_item)
        return torch.sum(user_state * candidate_emb, dim=-1)

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
