import torch
from transformers.loss.loss_utils import ForCausalLMLoss


def prepare_cache_position_and_position_ids(
    *,
    past_key_values,
    inputs_embeds: torch.Tensor,
    cache_position: torch.LongTensor | None,
    position_ids: torch.LongTensor | None,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    return cache_position, position_ids


class TemperatureMixin:
    def init_temperature(self):
        self.temperature = 1.0

    def set_hyper(self, temperature: float):
        self.temperature = temperature

    def apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "temperature"), "Model must have a temperature attribute."
        return logits / self.temperature


class TemperatureCausalLMLossMixin(TemperatureMixin):

    @property
    def loss_function(self):
        if hasattr(self, "_loss_function"):
            return self._loss_function

        def ForCausalLMLossWithTemperature(
            logits,
            labels,
            vocab_size: int,
            num_items_in_batch: int | None = None,
            ignore_index: int = -100,
            shift_labels: torch.Tensor | None = None,
            **kwargs,
        ) -> torch.Tensor:
            logits = self.apply_temperature(logits)
            return ForCausalLMLoss(
                logits,
                labels,
                vocab_size=vocab_size,
                num_items_in_batch=num_items_in_batch,
                ignore_index=ignore_index,
                shift_labels=shift_labels,
                **kwargs,
            )

        self._loss_function = ForCausalLMLossWithTemperature
        return self._loss_function
