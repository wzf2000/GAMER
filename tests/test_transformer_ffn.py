import unittest

import torch
import torch.nn.functional as F

from SeqRec.models.discriminative.SASRec import SASRec, SASRecConfig
from SeqRec.modules.layers.mbs_transformer import MBSFeedForward
from SeqRec.modules.layers.pbat import BehaviorSpecificFeedForward
from SeqRec.modules.layers.transformer import FeedForward


class FeedForwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(42)

    def test_zero_projection_preserves_normalized_residual(self):
        module = FeedForward(8, 16, 0.5, 'gelu', 1e-12).eval()
        calls = []
        module.dropout.register_forward_hook(lambda *args: calls.append('dropout'))
        module.LayerNorm.register_forward_hook(lambda *args: calls.append('norm'))
        with torch.no_grad():
            for linear in (module.dense_1, module.dense_2):
                linear.weight.zero_()
                linear.bias.zero_()
        inputs = torch.randn(2, 3, 8)
        outputs = module(inputs)
        expected = F.layer_norm(inputs, (8,), module.LayerNorm.weight, module.LayerNorm.bias, module.LayerNorm.eps)
        torch.testing.assert_close(outputs, expected)
        self.assertGreater(outputs.abs().max().item(), 1)
        self.assertEqual(calls, ['dropout', 'norm'])

    def test_disabled_residual_returns_projection_without_missing_attributes(self):
        module = FeedForward(8, 16, 0.5, 'gelu', 1e-12, residual=False)
        inputs = torch.randn(2, 3, 8)
        expected = module.dense_2(F.gelu(module.dense_1(inputs)))
        torch.testing.assert_close(module(inputs), expected)

    def test_sasrec_normalization_receives_gradients_and_updates(self):
        for n_items in (20, 80):
            with self.subTest(n_items=n_items):
                config = SASRecConfig(n_layers=2, n_heads=2, hidden_size=8, inner_size=16, dropout_prob=0.0)
                model = SASRec(config, n_items=n_items, max_his_len=5).train()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                normalizations = [layer.feed_forward.LayerNorm for layer in model.trm_encoder.layer]
                before = [module.weight.detach().clone() for module in normalizations]
                batch = {'inputs': torch.tensor([[1, 2, 3], [2, 4, 0]]), 'seq_len': torch.tensor([3, 2]), 'target': torch.tensor([4, 5])}
                loss = model.calculate_loss(batch)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for module in normalizations:
                    for parameter in module.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(parameter.grad.abs().sum().item(), 0)
                optimizer.step()
                for old, module in zip(before, normalizations):
                    self.assertFalse(torch.equal(old, module.weight))
                model.eval()
                scores = model.full_sort_predict(batch)
                self.assertEqual(scores.shape, (2, n_items + 1))
                self.assertTrue(torch.isfinite(scores).all())

    def test_behavior_specific_ffn_callers_support_both_residual_modes(self):
        inputs = torch.randn(2, 4, 8, requires_grad=True)
        types = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        modules = [BehaviorSpecificFeedForward(8, 16, 0.0, 'gelu', 1e-12, 3)]
        modules.extend(MBSFeedForward(8, 16, 0.0, 'gelu', 1e-12, 3, behavior_moe=enabled) for enabled in (False, True))
        for module in modules:
            with self.subTest(module=type(module).__name__, behavior_moe=getattr(module, 'behavior_moe', None)):
                outputs = module(inputs, types)
                self.assertEqual(outputs.shape, inputs.shape)
                self.assertTrue(torch.isfinite(outputs).all())
                (outputs * torch.randn_like(outputs)).sum().backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())


if __name__ == '__main__':
    unittest.main()
