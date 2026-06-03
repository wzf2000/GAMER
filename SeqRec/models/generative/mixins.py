import torch


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
