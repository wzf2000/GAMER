from types import SimpleNamespace
import unittest

import torch

from SeqRec.models.generative.qwen3.temporal_hierarchical import Qwen3TemporalHierarchicalWithTemperature
from SeqRec.utils.parameter_audit import audit_parameters, synthetic_probe_batch
from tests.models.generative.qwen3.test_rebuttal_ablation import tiny_config


class ProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.active = torch.nn.Linear(2, 2)
        self.zero_gradient = torch.nn.Linear(2, 2)
        self.unused = torch.nn.Linear(2, 2)
        self.empty = torch.nn.Linear(2, 2)
        self.register_buffer('prior', torch.ones(2))

    def forward(self, input_ids, labels, use_cache):
        states = input_ids.float()
        loss = self.active(states).sum() + self.zero_gradient(states).sum() * 0 + self.empty(states[:0]).sum()
        return SimpleNamespace(loss=loss)


class ParameterAuditTest(unittest.TestCase):
    def test_audit_distinguishes_registered_called_connected_and_zero_gradient(self):
        model = ProbeModel().train()
        model.unused.eval()
        original = {name: value.clone() for name, value in model.state_dict().items()}
        report = audit_parameters(model, [{'input_ids': torch.ones((2, 2)), 'labels': torch.ones((2, 2), dtype=torch.long)}])
        self.assertEqual(report['total'], 24)
        self.assertEqual(report['buffers'], 2)
        self.assertEqual(report['active_participating'], 12)
        self.assertEqual(report['nonzero_gradient_tensors'], 6)
        self.assertIn('unused.weight', report['observed_no_loss_path'])
        self.assertIn('empty.weight', report['observed_not_executed'])
        self.assertIn('zero_gradient.weight', report['observed_zero_gradient'])
        self.assertTrue(model.training)
        self.assertFalse(model.unused.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]))

    def test_component_probe_includes_final_vocabulary_and_excludes_frozen_prior(self):
        for updates in ({}, {'th_use_behavior_qkv': False}, {'th_use_output_gate': False}, {'sparse_layers_decoder': []}):
            with self.subTest(updates=updates):
                model = Qwen3TemporalHierarchicalWithTemperature(tiny_config(**updates))
                report = audit_parameters(model, [synthetic_probe_batch(model.config)])
                self.assertEqual(report['total'], sum(p.numel() for p in model.parameters()))
                self.assertGreater(report['buffers'], 0)
                self.assertLessEqual(report['active_participating'], report['trainable'])
                self.assertGreater(report['batches']['0']['supervised_positions'], 0)


if __name__ == '__main__':
    unittest.main()
