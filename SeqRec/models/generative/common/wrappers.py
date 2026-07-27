import torch
from typing import Any
from transformers.modeling_outputs import CausalLMOutputWithPast

from SeqRec.models.generative.common.temperature import TemperatureCausalLMLossMixin


class CustomCausalLMWrapperMixin(TemperatureCausalLMLossMixin):
    def train(self, mode: bool = True):
        result = super().train(mode)
        if mode and getattr(self.config, "ranking_freeze_backbone", False):
            self.model.eval()
        return result

    def init_custom_causal_lm(self, config: Any, model_cls: type):
        self.model = model_cls(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "use_ranking_head", False):
            self.ranking_head_dropout = torch.nn.Dropout(
                getattr(config, "ranking_head_dropout", getattr(config, "dropout_rate", 0.0))
            )
            if getattr(config, "ranking_score_type", "hidden_head") == "llm_pair":
                ranking_use_user_embedding = getattr(config, "ranking_use_user_embedding", True)
                ranking_feature_size = config.hidden_size * (4 if ranking_use_user_embedding else 3)
                if ranking_use_user_embedding:
                    self.ranking_user_embedding = torch.nn.Embedding(
                        int(getattr(config, "ranking_num_users", 0)) + 1,
                        config.hidden_size,
                        padding_idx=0,
                    )
                self.ranking_head = torch.nn.Sequential(
                    torch.nn.Linear(ranking_feature_size, config.hidden_size),
                    torch.nn.PReLU(),
                    torch.nn.Linear(config.hidden_size, 1),
                )
            else:
                self.ranking_head = torch.nn.Linear(config.hidden_size, 1)
        self.post_init()
        self.init_temperature()

    def _ranking_logits_from_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not hasattr(self, "ranking_head"):
            raise ValueError("Ranking head is not initialized. Train or load a checkpoint with use_ranking_head=True.")
        if attention_mask is None:
            last_hidden_states = hidden_states[:, -1, :]
        else:
            last_indices = attention_mask.to(hidden_states.device).long().sum(dim=1).clamp(min=1) - 1
            batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            last_hidden_states = hidden_states[batch_indices, last_indices]
        return self.ranking_head(self.ranking_head_dropout(last_hidden_states)).squeeze(-1)

    def _llm_pair_ranking_logits(
        self,
        *,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor | None,
        user_id: torch.Tensor | None,
    ) -> torch.Tensor:
        ranking_use_user_embedding = getattr(self.config, "ranking_use_user_embedding", True)
        if ranking_use_user_embedding and not hasattr(self, "ranking_user_embedding"):
            raise ValueError("LLM-pair ranking head is not initialized. Train or load a llm_pair checkpoint.")

        if attention_mask is None:
            last_indices = torch.full(
                (hidden_states.shape[0],),
                hidden_states.shape[1] - 1,
                device=hidden_states.device,
                dtype=torch.long,
            )
        else:
            last_indices = attention_mask.to(hidden_states.device).long().sum(dim=1).clamp(min=1) - 1

        candidate_len = getattr(self.config, "ranking_candidate_len", None)
        if candidate_len is None:
            candidate_len = max(1, int(getattr(self.config, "num_positions", 2)) - 1)
        candidate_len = max(1, int(candidate_len))
        features = []
        zero_state = hidden_states.new_zeros(hidden_states.shape[-1])
        for batch_index, last_index_tensor in enumerate(last_indices):
            last_index = int(last_index_tensor.item())
            candidate_start = max(0, last_index - candidate_len + 1)
            candidate_state = hidden_states[batch_index, candidate_start : last_index + 1].mean(dim=0)
            history_state = (
                hidden_states[batch_index, candidate_start - 1]
                if candidate_start > 0
                else zero_state
            )
            features.append(
                torch.cat(
                    [
                        history_state,
                        candidate_state,
                        history_state * candidate_state,
                    ],
                    dim=-1,
                )
            )
        pair_features = torch.stack(features, dim=0)
        if not ranking_use_user_embedding:
            features = pair_features
        elif user_id is None:
            user_state = hidden_states.new_zeros(hidden_states.shape[0], hidden_states.shape[-1])
            features = torch.cat([user_state, pair_features], dim=-1)
        else:
            user_state = self.ranking_user_embedding(user_id.to(hidden_states.device).long())
            features = torch.cat([user_state, pair_features], dim=-1)
        return self.ranking_head(self.ranking_head_dropout(features)).squeeze(-1)

    def prepare_custom_causal_lm_inputs(
        self,
        *,
        position_ids: torch.LongTensor | None,
        cache_position: torch.LongTensor | None,
        model_kwargs: dict[str, Any],
        extra_kwargs: dict[str, Any],
        wrapper_kwargs: dict[str, Any],
    ) -> tuple[torch.LongTensor | None, dict[str, Any]]:
        return position_ids, model_kwargs

    def forward_custom_causal_lm(
        self,
        *,
        labels: torch.LongTensor | None,
        position_ids: torch.LongTensor | None,
        output_attentions: bool | None,
        output_hidden_states: bool | None,
        logits_to_keep: int | torch.Tensor,
        ranking_labels: torch.Tensor | None = None,
        model_kwargs: dict[str, Any],
        extra_kwargs: dict[str, Any],
        wrapper_kwargs: dict[str, Any] | None = None,
    ) -> CausalLMOutputWithPast:
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        position_ids, model_kwargs = self.prepare_custom_causal_lm_inputs(
            position_ids=position_ids,
            cache_position=model_kwargs.get("cache_position"),
            model_kwargs=model_kwargs,
            extra_kwargs=extra_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )
        model_call_kwargs = {**model_kwargs, **extra_kwargs}
        outputs = self.model(
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **model_call_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if wrapper_kwargs.get("use_ranking_head", False) or ranking_labels is not None:
            if getattr(self.config, "ranking_score_type", "hidden_head") == "llm_pair":
                ranking_logits = self._llm_pair_ranking_logits(
                    hidden_states=hidden_states,
                    attention_mask=model_kwargs.get("attention_mask"),
                    user_id=wrapper_kwargs.get("user_id"),
                )
            else:
                ranking_logits = self._ranking_logits_from_hidden_states(
                    hidden_states,
                    model_kwargs.get("attention_mask"),
                )
            loss = None
            if ranking_labels is not None:
                ranking_labels = ranking_labels.to(
                    device=ranking_logits.device,
                    dtype=ranking_logits.dtype,
                ).view_as(ranking_logits)
                ranking_pos_weight = getattr(self.config, "ranking_pos_weight", None)
                pos_weight = None
                if ranking_pos_weight is not None:
                    pos_weight = torch.tensor(
                        ranking_pos_weight,
                        device=ranking_logits.device,
                        dtype=ranking_logits.dtype,
                    )
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    ranking_logits,
                    ranking_labels,
                    pos_weight=pos_weight,
                )
            return CausalLMOutputWithPast(
                loss=loss,
                logits=ranking_logits[:, None],
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **extra_kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ExtendedSessionPositionMixin(CustomCausalLMWrapperMixin):
    def prepare_custom_causal_lm_inputs(
        self,
        *,
        position_ids: torch.LongTensor | None,
        cache_position: torch.LongTensor | None,
        model_kwargs: dict[str, Any],
        extra_kwargs: dict[str, Any],
        wrapper_kwargs: dict[str, Any],
    ) -> tuple[torch.LongTensor | None, dict[str, Any]]:
        extended_session_ids = wrapper_kwargs.get("extended_session_ids")
        if cache_position is not None and cache_position.min() == 0:
            if extended_session_ids is not None:
                self.max_extended_session_id = extended_session_ids.max(dim=-1)[0]
        elif cache_position is not None:
            if extended_session_ids is not None:
                assert cache_position.shape[-1] == 1
                if self.max_extended_session_id.ndim == 1:
                    self.max_extended_session_id += 1
                    extended_session_ids = self.max_extended_session_id[:, None]
                else:
                    self.max_extended_session_id += 1
                    extended_session_ids = self.max_extended_session_id[None]
        if extended_session_ids is not None:
            position_ids = extended_session_ids
        return position_ids, model_kwargs
