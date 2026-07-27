import unittest
from types import SimpleNamespace

import torch

from SeqRec.models.generative.qwen3.multi_router import Qwen3MultiDecoderRouter
from SeqRec.tasks.training.helpers import _configure_behavior_tokens


class _Dataset:
    behaviors = ["click", "collect", "cart", "buy"]
    behavior_level = {
        "click": 0,
        "collect": 1,
        "cart": 1,
        "buy": 2,
    }
    target_behavior = "buy"

    def __init__(self, separate_behavior_identity_and_level: bool):
        self.separate_behavior_identity_and_level = (
            separate_behavior_identity_and_level
        )

    @staticmethod
    def get_behavior_tokens(behavior: str) -> list[str]:
        return [f"<behavior_{behavior}>"]


class BehaviorHierarchySeparationTest(unittest.TestCase):
    def test_training_config_separates_identity_and_level_for_v3(self):
        config = SimpleNamespace()
        dataset = _Dataset(separate_behavior_identity_and_level=True)

        _configure_behavior_tokens(
            config,
            dataset,
            tokenizer=None,
            behavior_token_ids=[10, 11, 12, 13],
        )

        self.assertEqual(config.behavior_maps, {10: 0, 11: 1, 12: 2, 13: 3})
        self.assertEqual(
            config.behavior_level_maps,
            {10: 0, 11: 1, 12: 1, 13: 2},
        )
        self.assertEqual(config.num_behavior, 4)
        self.assertEqual(config.num_behavior_levels, 3)
        self.assertTrue(config.separate_behavior_identity_and_level)

    def test_legacy_dataset_keeps_identity_as_relation_level(self):
        config = SimpleNamespace()
        dataset = _Dataset(separate_behavior_identity_and_level=False)

        _configure_behavior_tokens(
            config,
            dataset,
            tokenizer=None,
            behavior_token_ids=[10, 11, 12, 13],
        )

        self.assertEqual(
            config.behavior_level_maps,
            config.behavior_maps,
        )
        self.assertEqual(config.num_behavior_levels, 4)
        self.assertFalse(config.separate_behavior_identity_and_level)

    def test_router_keeps_identity_but_ties_same_level_behaviors(self):
        config = SimpleNamespace(
            num_experts=4,
            num_positions=3,
            num_behavior=4,
            eos_token_id=2,
            pad_token_id=0,
            bos_token_id=1,
            behavior_maps={10: 0, 11: 1, 12: 2, 13: 3},
            behavior_level_maps={10: 0, 11: 1, 12: 1, 13: 2},
            use_user_token=False,
            use_behavior_token=True,
            Moe_behavior_only=False,
        )
        router = Qwen3MultiDecoderRouter(num_items=2, config=config)
        input_ids = torch.tensor([[11, 20, 21, 12, 22, 23, 2]])

        (
            _,
            behavior_indices,
            behavior_identity_indices,
            behavior_level_indices,
        ) = router(input_ids)

        self.assertTrue(torch.equal(
            behavior_indices,
            torch.tensor([[0, 2, 2, 0, 3, 3, 0]]),
        ))
        self.assertTrue(torch.equal(
            behavior_identity_indices,
            torch.tensor([[2, 2, 2, 3, 3, 3, 0]]),
        ))
        self.assertTrue(torch.equal(
            behavior_level_indices,
            torch.tensor([[2, 2, 2, 2, 2, 2, 0]]),
        ))


if __name__ == "__main__":
    unittest.main()
