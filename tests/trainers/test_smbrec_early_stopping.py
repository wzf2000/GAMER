import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

from SeqRec.trainers.SMBRec import Trainer


class SMBRecEarlyStoppingTest(unittest.TestCase):
    def test_initial_model_is_available_when_early_stopping_finds_no_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer.__new__(Trainer)
            trainer.model = torch.nn.Linear(1, 1)
            trainer.output_dir = tmp
            trainer.metrics = ["auc"]
            trainer.main_metric = "auc"
            trainer.main_metric_higher_is_better = True
            trainer.patience = 2
            trainer.epochs = 3
            trainer.evaluate = Mock(side_effect=[
                {"auc": 0.8},
                {"auc": 0.7},
                {"auc": 0.6},
            ])
            trainer.fit = Mock(return_value=0.1)
            trainer._save_epoch_checkpoint = Mock()

            trainer.train()

            self.assertEqual(trainer.fit.call_count, 2)
            self.assertEqual(trainer.evaluate.call_count, 3)
            self.assertTrue((Path(tmp) / "best_model.pth").exists())


if __name__ == "__main__":
    unittest.main()
