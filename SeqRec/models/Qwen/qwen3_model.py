import torch
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.models.qwen3 import Qwen3ForCausalLM


class Qwen3WithTemperature(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.temperature = 1.0

    def set_hyper(self, temperature: float):
        self.temperature = temperature

    @property
    def loss_function(self):
        if hasattr(self, "_loss_function"):
            return self._loss_function

        assert hasattr(self, "temperature"), "Model must have a temperature attribute."

        def ForCausalLMLossWithTemperature(
            logits,
            labels,
            vocab_size: int,
            num_items_in_batch: int | None = None,
            ignore_index: int = -100,
            shift_labels: torch.Tensor | None = None,
            **kwargs,
        ) -> torch.Tensor:
            logits /= self.temperature
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
