import torch


class TemperatureMixin:
    def init_temperature(self):
        self.temperature = 1.0

    def set_hyper(self, temperature: float):
        self.temperature = temperature

    def apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "temperature"), "Model must have a temperature attribute."
        return logits / self.temperature
