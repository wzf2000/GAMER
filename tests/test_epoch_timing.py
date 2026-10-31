import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Trainer, TrainingArguments

from SeqRec.trainers.epoch_timing import EpochTimingCallback
from SeqRec.utils.args import ScriptTrainingArgs


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)

    def forward(self, input_ids, labels=None):
        logits = self.projection(input_ids.float())
        return {'loss': torch.nn.functional.cross_entropy(logits, labels), 'logits': logits}


class EpochTimingTest(unittest.TestCase):
    def test_default_disabled_and_invalid_epoch_limits(self):
        self.assertEqual(ScriptTrainingArgs().benchmark_epochs, 0)
        for epochs in (0, -1, True, 1.5):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                EpochTimingCallback(epochs, '.')

    def test_phase_timing_stops_after_evaluation_and_save_without_changing_schedule(self):
        with TemporaryDirectory() as directory:
            callback = EpochTimingCallback(1, directory)
            args = SimpleNamespace(eval_strategy='epoch', save_strategy='epoch', eval_delay=1, num_train_epochs=200,
                                   get_warmup_steps=lambda steps: 16)
            state = SimpleNamespace(epoch=0, global_step=0, max_steps=400)
            control = SimpleNamespace(should_training_stop=False)
            with patch('SeqRec.trainers.epoch_timing.synchronize_device'), patch('torch.cuda.is_available', return_value=False), \
                    patch('SeqRec.trainers.epoch_timing.time.perf_counter', side_effect=[0.0, 1.0, 5.0, 8.0, 9.0, 10.0]):
                callback.on_train_begin(args, state, control)
                callback.on_epoch_begin(args, state, control)
                state.epoch, state.global_step = 1, 2
                callback.on_epoch_end(args, state, control)
                self.assertTrue(control.should_training_stop)
                callback.on_evaluate(args, state, control, metrics={'eval_runtime': 2.5, 'eval_loss': 0.7})
                callback.on_save(args, state, control)
                callback.on_train_end(args, state, control)
            report = json.loads((Path(directory) / 'epoch-timing.json').read_text())
            self.assertEqual(report['planned_num_train_epochs'], 200)
            self.assertEqual(report['epochs_max_across_ranks'], [{'epoch': 1.0, 'train_seconds': 4.0, 'validation_seconds': 3.0,
                                                                 'checkpoint_seconds': 1.0, 'epoch_seconds': 8.0}])

    def test_real_cpu_trainer_keeps_200_epoch_scheduler_and_completes_one_epoch(self):
        with TemporaryDirectory() as directory:
            args = TrainingArguments(output_dir=directory, use_cpu=True, num_train_epochs=200,
                                     per_device_train_batch_size=2, per_device_eval_batch_size=2,
                                     learning_rate=0.001, warmup_ratio=0.04, eval_strategy='epoch', save_strategy='epoch', eval_delay=1,
                                     load_best_model_at_end=True, save_total_limit=1, report_to=[], disable_tqdm=True)
            dataset = [{'input_ids': torch.tensor([float(index), 1.0]), 'labels': index % 2} for index in range(4)]
            trainer = Trainer(model=_TinyModel(), args=args, train_dataset=dataset, eval_dataset=dataset,
                              callbacks=[EpochTimingCallback(1, directory)])
            trainer.train()
            self.assertEqual(trainer.state.epoch, 1.0)
            self.assertEqual(trainer.state.global_step, 2)
            self.assertEqual(trainer.state.max_steps, 400)
            self.assertEqual(args.num_train_epochs, 200)
            self.assertEqual(args.get_warmup_steps(trainer.state.max_steps), 16)
            self.assertTrue((Path(directory) / 'checkpoint-2/trainer_state.json').is_file())
            report = json.loads((Path(directory) / 'epoch-timing.json').read_text())
            self.assertEqual(len(report['epochs_max_across_ranks']), 1)
            for field in ('train_seconds', 'validation_seconds', 'checkpoint_seconds'):
                self.assertGreaterEqual(report['epochs_max_across_ranks'][0][field], 0)

    def test_first_checkpoint_identity_is_synchronized_before_loading_best(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / 'checkpoint-2').mkdir()
            callback = EpochTimingCallback(1, directory)
            callback.current = {}
            callback.epoch_start, callback.evaluation_end = 1.0, 2.0
            args = SimpleNamespace(load_best_model_at_end=True)
            state = SimpleNamespace(best_global_step=2, best_model_checkpoint=None)
            with patch('SeqRec.trainers.epoch_timing.synchronize_device'), patch('torch.cuda.is_available', return_value=False), \
                    patch('SeqRec.trainers.epoch_timing.dist.is_initialized', return_value=True), \
                    patch('SeqRec.trainers.epoch_timing.dist.barrier') as barrier:
                callback.on_save(args, state, None)
            barrier.assert_called_once()
            self.assertEqual(state.best_model_checkpoint, str(Path(directory) / 'checkpoint-2'))


if __name__ == '__main__':
    unittest.main()
