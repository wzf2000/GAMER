from contextlib import ExitStack
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from SeqRec.tasks.training.base import BaseGenerativeTrainTask


class SeedTask(BaseGenerativeTrainTask):
    checkpoint_dir_name = 'test'

    def init(self, seed, *args):
        torch.manual_seed(seed)

    def finish(self, *args):
        pass

    def load_train_data(self, data_args):
        if self.cold_cache:
            torch.manual_seed(42)
        return SimpleNamespace(datasets=[object()]), object()


class TrainingSeedTest(unittest.TestCase):
    def initial_weights(self, seed, cold_cache):
        task = SeedTask()
        task.cold_cache = cold_cache
        captured = []

        def create_model(**kwargs):
            captured.append(torch.rand(16))
            return SimpleNamespace(config=SimpleNamespace())

        with TemporaryDirectory() as directory, ExitStack() as stack:
            namespace = 'SeqRec.tasks.training.base.'
            replacements = {
                'load_config_and_tokenizer': Mock(return_value=(object(), object())),
                'get_backbone_train_profile': Mock(return_value='test'),
                'prepare_tokenizer_and_config': Mock(),
                'build_train_collator': Mock(),
                'prepare_generative_model_for_training': create_model,
                'finalize_generative_model': lambda model, *args: model,
                'build_training_arguments_from_script_args': Mock(return_value=SimpleNamespace()),
                'build_hf_trainer': Mock(return_value=Mock()),
            }
            for name, replacement in replacements.items():
                stack.enter_context(patch(namespace + name, replacement))
            task.invoke(seed=seed, backbone='Qwen3TemporalHierarchical', output_dir=directory, debug=True)
        return captured[0]

    def test_model_initialization_is_independent_of_cache_state(self):
        for seed in (7, 42, 43):
            self.assertTrue(torch.equal(self.initial_weights(seed, True), self.initial_weights(seed, False)))
        self.assertFalse(torch.equal(self.initial_weights(7, True), self.initial_weights(43, True)))


if __name__ == '__main__':
    unittest.main()
