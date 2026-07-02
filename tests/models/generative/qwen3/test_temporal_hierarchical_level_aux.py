import unittest

import torch
from transformers.models.qwen3_moe import Qwen3MoeConfig

from SeqRec.models.generative.qwen3.temporal_hierarchical import (
    Qwen3TemporalHierarchicalAttention,
    Qwen3TemporalHierarchicalWithTemperature,
)


class TemporalHierarchicalLevelAuxTest(unittest.TestCase):
    def _build_config(self, loss_weight: float = 0.0) -> Qwen3MoeConfig:
        config = Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            attention_bias=False,
            attention_dropout=0.0,
            hidden_act="silu",
            max_position_embeddings=64,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            use_cache=False,
        )
        config.mlp_type = "dense"
        config.dropout_rate = 0.0
        config.n_positions = 16
        config.num_positions = 3
        config.model_max_length = 24
        config.num_behavior = 2
        config.behavior_maps = {10: 0, 11: 1}
        config.use_behavior_token = True
        config.use_user_token = False
        config.Moe_behavior_only = False
        config.sparse_layers_decoder = []
        config.behavior_injection_decoder = []
        config.temporal_hierarchical_attention_decoder = []
        config.th_level_auxiliary_loss_weight = loss_weight
        config.th_relation_regularization_weight = 0.0
        return config

    def test_level_head_is_config_gated(self):
        disabled_model = Qwen3TemporalHierarchicalWithTemperature(
            self._build_config(loss_weight=0.0)
        )
        enabled_model = Qwen3TemporalHierarchicalWithTemperature(
            self._build_config(loss_weight=0.05)
        )

        self.assertFalse(hasattr(disabled_model, "level_head"))
        self.assertTrue(hasattr(enabled_model, "level_head"))
        self.assertEqual(enabled_model.level_head.out_features, 3)

    def test_next_behavior_level_labels_are_built_from_behavior_tokens(self):
        model = Qwen3TemporalHierarchicalWithTemperature(
            self._build_config(loss_weight=0.05)
        )
        input_ids = torch.tensor([
            [10, 12, 13, 11, 14, 15, 2, 0],
            [7, 10, 16, 17, 11, 18, 19, 0],
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 0],
        ])

        level_labels = model._build_next_behavior_level_labels(
            input_ids,
            attention_mask,
        )

        expected = torch.full((2, 7), -100, dtype=torch.long)
        expected[0, 2] = 2
        expected[1, 0] = 1
        expected[1, 3] = 2
        self.assertTrue(torch.equal(level_labels.cpu(), expected))

    def test_auxiliary_loss_is_scalar_when_behavior_targets_exist(self):
        model = Qwen3TemporalHierarchicalWithTemperature(
            self._build_config(loss_weight=0.05)
        )
        input_ids = torch.tensor([[10, 12, 13, 11, 14, 15]])
        attention_mask = torch.ones_like(input_ids)
        hidden_states = torch.randn(1, 6, 16)

        loss = model.compute_auxiliary_loss(
            hidden_states=hidden_states,
            labels=input_ids.clone(),
            model_kwargs={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            extra_kwargs={},
            wrapper_kwargs={},
        )

        self.assertIsNotNone(loss)
        self.assertEqual(loss.ndim, 0)
        self.assertGreater(loss.item(), 0.0)

    def test_relation_regularization_skips_frozen_table_bias(self):
        config = self._build_config()
        config.th_attention_mode = "relation_bias"
        config.th_relation_bias_type = "table"
        config.th_relation_bias_trainable = False
        config.th_relation_bias_init = "soft"
        config.th_relation_bias_soft_scale = 0.05
        attention = Qwen3TemporalHierarchicalAttention(
            config,
            layer_idx=0,
            is_temporal_hierarchical=True,
        )

        self.assertIsNone(attention.compute_relation_regularization_loss())

    def test_relation_regularization_supports_trainable_factorized_bias(self):
        config = self._build_config()
        config.th_attention_mode = "relation_bias"
        config.th_relation_bias_type = "factorized"
        config.th_relation_bias_rank = 2
        config.th_relation_bias_init = "soft"
        config.th_relation_bias_soft_scale = 0.05
        config.th_relation_regularization_target = "soft"
        config.th_relation_regularization_soft_scale = 0.05
        config.th_relation_regularization_include_special_level = False
        attention = Qwen3TemporalHierarchicalAttention(
            config,
            layer_idx=0,
            is_temporal_hierarchical=True,
        )

        loss = attention.compute_relation_regularization_loss()

        self.assertIsNotNone(loss)
        self.assertEqual(loss.ndim, 0)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_model_relation_regularization_is_config_gated(self):
        disabled_config = self._build_config()
        disabled_config.temporal_hierarchical_attention_decoder = [0]
        disabled_config.th_attention_mode = "relation_bias"
        disabled_config.th_relation_bias_type = "factorized"
        disabled_config.th_relation_bias_rank = 2
        disabled_model = Qwen3TemporalHierarchicalWithTemperature(
            disabled_config
        )

        enabled_config = self._build_config()
        enabled_config.temporal_hierarchical_attention_decoder = [0]
        enabled_config.th_attention_mode = "relation_bias"
        enabled_config.th_relation_bias_type = "factorized"
        enabled_config.th_relation_bias_rank = 2
        enabled_config.th_relation_regularization_weight = 0.01
        enabled_config.th_relation_regularization_target = "soft"
        enabled_config.th_relation_regularization_soft_scale = 0.05
        enabled_model = Qwen3TemporalHierarchicalWithTemperature(
            enabled_config
        )

        self.assertIsNone(
            disabled_model.model.compute_relation_regularization_loss()
        )
        self.assertIsNotNone(
            enabled_model.model.compute_relation_regularization_loss()
        )


if __name__ == "__main__":
    unittest.main()
