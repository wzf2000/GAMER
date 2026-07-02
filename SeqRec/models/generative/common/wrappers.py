import torch
from typing import Any
from transformers.modeling_outputs import CausalLMOutputWithPast

from SeqRec.models.generative.common.temperature import TemperatureCausalLMLossMixin


class CustomCausalLMWrapperMixin(TemperatureCausalLMLossMixin):
    def init_custom_causal_lm(self, config: Any, model_cls: type):
        self.model = model_cls(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if hasattr(self, "init_auxiliary_heads"):
            self.init_auxiliary_heads(config)
        self.post_init()
        self.init_temperature()

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
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **extra_kwargs)
            if hasattr(self, "compute_auxiliary_loss"):
                auxiliary_loss = self.compute_auxiliary_loss(
                    hidden_states=hidden_states,
                    labels=labels,
                    model_kwargs=model_kwargs,
                    extra_kwargs=extra_kwargs,
                    wrapper_kwargs=wrapper_kwargs,
                )
                if auxiliary_loss is not None:
                    loss = loss + auxiliary_loss

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
