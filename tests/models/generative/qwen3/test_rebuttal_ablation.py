from tempfile import TemporaryDirectory
import unittest

import torch
import torch.nn.functional as F
from transformers.models.qwen3_moe import Qwen3MoeConfig

from SeqRec.models.generative.qwen3.temporal_hierarchical import Qwen3TemporalHierarchicalWithTemperature


def tiny_config(**updates):
    values = dict(vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=8, attention_bias=False,
                  attention_dropout=0.0, dropout_rate=0.0, hidden_act='silu', max_position_embeddings=64,
                  pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False, mlp_type='Qwen3',
                  moe_intermediate_size=16, behavior_embedding_dim=4, n_positions=8, num_positions=3,
                  model_max_length=24, num_experts=4, num_behavior=2, behavior_maps={10: 0, 11: 1},
                  use_behavior_token=True, use_user_token=False, Moe_behavior_only=False,
                  sparse_layers_decoder=[0], behavior_injection_decoder=[0], temporal_hierarchical_attention_decoder=[0],
                  th_relation_bias_type='table', th_relation_bias_trainable=False, th_relation_bias_init='soft',
                  th_relation_bias_soft_scale=0.05)
    values.update(updates)
    return Qwen3MoeConfig(**values)


class RebuttalAblationTest(unittest.TestCase):
    def test_missing_options_preserve_default_weights_and_logits(self):
        torch.manual_seed(7)
        old_defaults = Qwen3TemporalHierarchicalWithTemperature(tiny_config()).eval()
        torch.manual_seed(7)
        explicit_defaults = Qwen3TemporalHierarchicalWithTemperature(tiny_config(th_use_behavior_qkv=True, th_use_output_gate=True)).eval()
        self.assertEqual(set(old_defaults.state_dict()), set(explicit_defaults.state_dict()))
        for name, tensor in old_defaults.state_dict().items():
            self.assertTrue(torch.equal(tensor, explicit_defaults.state_dict()[name]), name)
        ids = torch.tensor([[10, 12, 13, 11, 14, 15]])
        with torch.no_grad():
            self.assertTrue(torch.equal(old_defaults(input_ids=ids).logits, explicit_defaults(input_ids=ids).logits))

    def test_components_keep_causal_shifted_supervision_and_gradients(self):
        for updates in ({}, {'th_use_behavior_qkv': False}, {'th_use_output_gate': False}, {'sparse_layers_decoder': []}, {'th_relation_bias_soft_scale': 0.0}):
            with self.subTest(updates=updates):
                model = Qwen3TemporalHierarchicalWithTemperature(tiny_config(**updates))
                ids = torch.tensor([[10, 12, 13, 11, 14, 15, 0]])
                labels = ids.clone()
                labels[:, [0, 3, 6]] = -100
                output = model(input_ids=ids, attention_mask=(ids != 0).long(), labels=labels, use_cache=False)
                reference = F.cross_entropy(output.logits[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1))
                self.assertTrue(torch.allclose(output.loss, reference))
                output.loss.backward()
                attention = model.model.layers[0].self_attn
                self.assertTrue(torch.isfinite(attention.q_proj.weight.grad).all())
                self.assertIsNone(attention.level_pair_bias.grad)
                if updates.get('th_use_behavior_qkv', True):
                    self.assertGreater(attention.q_behavior_embedding.weight.grad.abs().sum().item(), 0)
                else:
                    self.assertFalse(hasattr(attention, 'q_behavior_embedding'))
                if updates.get('th_use_output_gate', True):
                    self.assertGreater(attention.gating.weight.grad.abs().sum().item(), 0)
                else:
                    self.assertFalse(hasattr(attention, 'gating'))

    def test_cached_generation_matches_without_cache_for_each_ablation(self):
        for updates in ({}, {'th_use_behavior_qkv': False}, {'th_use_output_gate': False}, {'sparse_layers_decoder': []}, {'temporal_hierarchical_attention_decoder': [], 'sparse_layers_decoder': [], 'behavior_injection_decoder': []}):
            with self.subTest(updates=updates):
                model = Qwen3TemporalHierarchicalWithTemperature(tiny_config(**updates)).eval()
                ids = torch.tensor([[10, 12, 13, 11]])
                with torch.no_grad():
                    cached = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=2, use_cache=True)
                    uncached = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=2, use_cache=False)
                self.assertTrue(torch.equal(cached, uncached))

    def test_weak_ranks_change_only_behavior_prior_and_preserve_special_id(self):
        torch.manual_seed(9)
        full = Qwen3TemporalHierarchicalWithTemperature(tiny_config()).eval()
        torch.manual_seed(9)
        tied = Qwen3TemporalHierarchicalWithTemperature(tiny_config(th_prior_behavior_ranks=[0, 0])).eval()
        for name, value in full.state_dict().items():
            if not name.endswith('level_pair_bias'):
                self.assertTrue(torch.equal(value, tied.state_dict()[name]), name)
        original = full.model.layers[0].self_attn.level_pair_bias
        changed = tied.model.layers[0].self_attn.level_pair_bias
        self.assertTrue(torch.equal(original[0], changed[0]))
        self.assertTrue(torch.equal(original[:, 0], changed[:, 0]))
        self.assertEqual(changed[1:, 1:].abs().sum().item(), 0.0)
        self.assertGreater(original[1:, 1:].abs().sum().item(), 0.0)
        ids = torch.tensor([[11, 12, 13, 10, 14, 15]])
        with torch.no_grad():
            self.assertFalse(torch.equal(full(input_ids=ids).logits, tied(input_ids=ids).logits))

    def test_invalid_rank_maps_are_rejected(self):
        for ranks in ([0], [0, float('nan')], [0, True], [0, -1]):
            with self.subTest(ranks=ranks), self.assertRaises(ValueError):
                Qwen3TemporalHierarchicalWithTemperature(tiny_config(th_prior_behavior_ranks=ranks))

    def test_lambda_grid_and_disabled_components_survive_checkpoint_reload(self):
        for scale in (0.0, 0.01, 0.05, 0.1):
            with self.subTest(scale=scale), TemporaryDirectory() as directory:
                model = Qwen3TemporalHierarchicalWithTemperature(tiny_config(
                    th_relation_bias_soft_scale=scale, th_use_behavior_qkv=False,
                    th_use_output_gate=False, th_prior_behavior_ranks=[0, 0],
                )).eval()
                model.save_pretrained(directory)
                loaded = Qwen3TemporalHierarchicalWithTemperature.from_pretrained(directory).eval()
                self.assertFalse(hasattr(loaded.model.layers[0].self_attn, 'gating'))
                self.assertFalse(hasattr(loaded.model.layers[0].self_attn, 'q_behavior_embedding'))
                self.assertNotIn('level_pair_bias', dict(loaded.model.layers[0].self_attn.named_parameters()))
                self.assertAlmostEqual(loaded.model.layers[0].self_attn.level_pair_bias[0, 1, 0].item(), -scale)
                ids = torch.tensor([[10, 12, 13, 11]])
                with torch.no_grad():
                    self.assertTrue(torch.equal(model(input_ids=ids).logits, loaded(input_ids=ids).logits))


if __name__ == '__main__':
    unittest.main()
