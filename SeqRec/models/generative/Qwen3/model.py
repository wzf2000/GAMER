from transformers.models.qwen3 import Qwen3ForCausalLM

from SeqRec.models.generative.mixins import TemperatureCausalLMLossMixin


class Qwen3WithTemperature(TemperatureCausalLMLossMixin, Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.init_temperature()
