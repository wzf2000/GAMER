import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from SeqRec.models.generative.qwen3.temporal_hierarchical import Qwen3TemporalHierarchicalWithTemperature
from SeqRec.tasks.training.helpers import _configure_behavior_tokens
from SeqRec.utils.behavior_hierarchy import load_behavior_schema
from SeqRec.utils.rebuttal import dataset_fingerprints, effective_config
from tests.models.generative.qwen3.test_rebuttal_ablation import tiny_config


LEVELS = dict(pv=0, click=1, cart=2, collect=2, buy=3)
RANDOM_RANKS = [2, 1, 0, 0, 3]


def v3_config(**updates):
    values = dict(num_behavior=5, behavior_maps={10 + i: i for i in range(5)},
                  separate_behavior_identity_and_level=True, num_behavior_levels=4,
                  behavior_level_maps={10 + i: level for i, level in enumerate(LEVELS.values())})
    values.update(updates)
    return tiny_config(**values)


class RebuttalV3HierarchyTest(unittest.TestCase):
    def test_legacy_identity_levels_are_numerically_unchanged(self):
        torch.manual_seed(42)
        legacy = Qwen3TemporalHierarchicalWithTemperature(tiny_config()).eval()
        torch.manual_seed(42)
        separated = Qwen3TemporalHierarchicalWithTemperature(tiny_config(
            separate_behavior_identity_and_level=True, num_behavior_levels=2,
            behavior_level_maps={10: 0, 11: 1},
        )).eval()
        for name, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, separated.state_dict()[name]), name)
        ids = torch.tensor([[10, 20, 21, 11, 22, 23]])
        with torch.no_grad():
            self.assertTrue(torch.equal(legacy(input_ids=ids).logits, separated(input_ids=ids).logits))

    def test_v3_changes_prior_lookup_and_preserves_all_weights_and_special_id(self):
        models = []
        for updates in ({}, {'th_prior_behavior_ranks': RANDOM_RANKS}):
            torch.manual_seed(42)
            models.append(Qwen3TemporalHierarchicalWithTemperature(v3_config(**updates)))
        natural, random = models
        for name, value in natural.state_dict().items():
            self.assertTrue(torch.equal(value, random.state_dict()[name]), name)
        attn = random.model.layers[0].self_attn
        self.assertEqual(attn.q_behavior_embedding.num_embeddings, 6)
        self.assertEqual(tuple(attn.level_pair_bias.shape), (5, 5, 2))
        self.assertEqual(attn.behavior_level_indices.tolist(), [0, 3, 2, 1, 1, 4])
        actions = torch.arange(6)[None]
        indices = attn.behavior_level_indices[actions]
        actual = attn._compute_level_pair_bias(indices, indices, torch.float32)
        expected = (indices[:, :, None] - indices[:, None, :]).clamp(max=0) * 0.05
        self.assertTrue(torch.equal(actual, expected[:, None].expand_as(actual)))
        for model in models:
            model.eval()
        ids = torch.tensor([[10, 20, 21, 12, 22, 23, 11, 24, 25, 14, 26, 27]])
        with torch.no_grad():
            self.assertFalse(torch.equal(natural(input_ids=ids).logits, random(input_ids=ids).logits))

    def test_shifted_supervision_gradients_cached_beam_and_checkpoint_reload(self):
        model = Qwen3TemporalHierarchicalWithTemperature(v3_config(th_prior_behavior_ranks=RANDOM_RANKS))
        ids = torch.tensor([[10, 20, 21, 12, 22, 23, 13, 24, 25, 14, 26, 27, 0]])
        labels = ids.clone()
        for token in (0, 10, 11, 12, 13, 14):
            labels[labels == token] = -100
        output = model(input_ids=ids, attention_mask=(ids != 0).long(), labels=labels, use_cache=False)
        reference = F.cross_entropy(output.logits[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1))
        self.assertTrue(torch.allclose(output.loss, reference))
        output.loss.backward()
        attn = model.model.layers[0].self_attn
        self.assertTrue(torch.isfinite(attn.q_proj.weight.grad).all())
        self.assertGreater(attn.q_behavior_embedding.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(attn.level_pair_bias.grad)
        model.eval()
        with TemporaryDirectory() as directory, torch.no_grad():
            model.save_pretrained(directory)
            loaded = Qwen3TemporalHierarchicalWithTemperature.from_pretrained(directory).eval()
            self.assertTrue(torch.equal(model(input_ids=ids).logits, loaded(input_ids=ids).logits))
            prompt = ids[:, :7]
            cached = loaded.generate(input_ids=prompt, max_new_tokens=2, num_beams=2, use_cache=True)
            uncached = loaded.generate(input_ids=prompt, max_new_tokens=2, num_beams=2, use_cache=False)
            self.assertTrue(torch.equal(cached, uncached))

    def test_planner_and_training_resolve_the_same_v3_hierarchy(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'TinyV3'
            data.mkdir()
            for suffix, value in [('behavior_level', LEVELS), ('behavior_schema', {'schema_version': 3, 'separate_behavior_identity_and_level': True}),
                                  ('index', {'0': ['<a_0>', '<b_0>']})]:
                (data / f'TinyV3.{suffix}.json').write_text(json.dumps(value))
            for suffix in ('inter', 'behavior', 'session', 'time'):
                (data / f'TinyV3.SMB.{suffix}.json').write_text('{}')
            values, _ = effective_config('config/s2s-models/Qwen3TemporalHierarchicalFixedSoft', root, 'TinyV3', '.index.json', 100, 1024)
            config = SimpleNamespace()
            dataset = SimpleNamespace(behaviors=list(LEVELS), behavior_level=LEVELS, separate_behavior_identity_and_level=True,
                                      target_behavior='buy', get_behavior_tokens=lambda behavior: [f'<behavior_{behavior}>'])
            _configure_behavior_tokens(config, dataset, None, list(values['behavior_maps']))
            for name in ('behavior_maps', 'behavior_level_maps', 'num_behavior', 'num_behavior_levels', 'separate_behavior_identity_and_level'):
                self.assertEqual(getattr(config, name), values[name])
            self.assertIn('TinyV3.behavior_schema.json', dataset_fingerprints(root, 'TinyV3', '.index.json'))

    def test_schema_and_invalid_v3_rank_indices_are_rejected(self):
        for ranks in ([0], [0, 1, 2, 2, 4], [0, 1, 2, 2, -1], [0, 1, 2, 2, True], [0, 1, 2, 2, 3.0]):
            with self.subTest(ranks=ranks), self.assertRaises(ValueError):
                Qwen3TemporalHierarchicalWithTemperature(v3_config(th_prior_behavior_ranks=ranks))
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'Tiny.behavior_schema.json'
            path.write_text('{"separate_behavior_identity_and_level":true}')
            for levels in ({'a': 0, 'b': 2}, {'a': True}, {'a': -1}):
                with self.subTest(levels=levels), self.assertRaises(ValueError):
                    load_behavior_schema(directory, 'Tiny', levels)


if __name__ == '__main__':
    unittest.main()
