import unittest

import numpy as np

from SeqRec.datasets.session_behavior.augmentation_policies import (
    AugmentationContext,
    BehaviorSequence,
    DatasetProportionPolicy,
    MultiViewAugmentationPolicy,
    SessionAwareDropoutPolicy,
    TargetConditionedPolicy,
    TimeDecayDropoutPolicy,
    UserAdaptiveRatioPolicy,
)


class SequenceAugmentationPolicyTest(unittest.TestCase):
    def setUp(self):
        self.context = AugmentationContext(
            uid="user",
            target_behavior="conversion",
            target_level=2,
            target_time=100.0,
            behavior_level={
                "pxs": 0,
                "click": 1,
                "conversion": 2,
            },
            max_behavior_level=2,
        )

    def test_behavior_sequence_rejects_misaligned_fields(self):
        with self.assertRaises(ValueError):
            BehaviorSequence(
                items=["a"],
                behaviors=[],
                session_ids=[0],
                times=[0.0],
            )

    def test_time_decay_preserves_recent_and_target_level_items(self):
        sequence = BehaviorSequence(
            items=["old", "target-level", "recent"],
            behaviors=["pxs", "conversion", "pxs"],
            session_ids=[0, 1, 2],
            times=[0.0, 10.0, 99.0],
        )
        policy = TimeDecayDropoutPolicy(
            tau=10.0,
            severity=1.0,
            max_drop_probability=1.0,
            min_recent_items=1,
            preserve_target_level=True,
        )

        view = policy.generate_view(
            sequence,
            self.context,
            np.random.default_rng(2),
        )

        self.assertNotIn(0, view.keep_indices)
        self.assertIn(1, view.keep_indices)
        self.assertIn(2, view.keep_indices)

    def test_time_decay_is_reproducible(self):
        sequence = BehaviorSequence(
            items=[str(index) for index in range(20)],
            behaviors=["pxs"] * 20,
            session_ids=list(range(20)),
            times=[float(index) for index in range(20)],
        )
        policy = TimeDecayDropoutPolicy()

        first = policy.generate_view(
            sequence,
            self.context,
            np.random.default_rng(42),
        )
        second = policy.generate_view(
            sequence,
            self.context,
            np.random.default_rng(42),
        )

        self.assertEqual(first.keep_indices, second.keep_indices)

    def test_session_dropout_preserves_sessions_atomically(self):
        sequence = BehaviorSequence(
            items=["a", "b", "c", "d", "e"],
            behaviors=["pxs", "click", "pxs", "click", "pxs"],
            session_ids=[0, 0, 1, 1, 2],
            times=[0.0, 1.0, 10.0, 11.0, 20.0],
        )
        policy = SessionAwareDropoutPolicy(
            recent_session_count=1,
            base_keep_probability=0.0,
            high_level_bonus=0.0,
            preserve_target_level=False,
            min_history_items=3,
        )

        view = policy.generate_view(
            sequence,
            self.context,
            np.random.default_rng(0),
        )

        kept_by_session = {
            session_id: [
                index in view.keep_indices
                for index, current_session_id in enumerate(sequence.session_ids)
                if current_session_id == session_id
            ]
            for session_id in set(sequence.session_ids)
        }
        self.assertTrue(all(
            all(values) or not any(values)
            for values in kept_by_session.values()
        ))
        self.assertIn(4, view.keep_indices)
        self.assertGreaterEqual(len(view.keep_indices), 3)

    def test_dataset_proportion_applies_soft_cap(self):
        sequence = BehaviorSequence(
            items=[str(index) for index in range(10)],
            behaviors=["pxs"] * 8 + ["conversion"] * 2,
            session_ids=list(range(10)),
            times=[float(index) for index in range(10)],
        )
        policy = DatasetProportionPolicy(
            target_proportions=(0.5, 0.0, 0.5),
            tolerance=1.0,
        )

        view = policy.generate_view(
            sequence,
            self.context,
            np.random.default_rng(0),
        )
        kept_behaviors = [
            sequence.behaviors[index]
            for index in view.keep_indices
        ]

        self.assertEqual(kept_behaviors.count("pxs"), 5)
        self.assertEqual(kept_behaviors.count("conversion"), 2)

    def test_user_adaptive_ratio_handles_zero_target_history(self):
        sequence = BehaviorSequence(
            items=[str(index) for index in range(10)],
            behaviors=["pxs"] * 10,
            session_ids=list(range(10)),
            times=[float(index) for index in range(10)],
        )
        policy = UserAdaptiveRatioPolicy(
            global_proportions=(0.2, 0.3, 0.5),
            tolerance=1.0,
        )

        view = policy.generate_view(
            sequence,
            self.context,
            np.random.default_rng(0),
        )

        self.assertEqual(len(view.keep_indices), 2)
        self.assertEqual(view.metadata["target_count"], 0)

    def test_target_conditioned_restores_relevant_levels(self):
        sequence = BehaviorSequence(
            items=["shallow", "precursor", "same-level"],
            behaviors=["pxs", "click", "conversion"],
            session_ids=[0, 1, 2],
            times=[0.0, 0.0, 0.0],
        )
        base_policy = TimeDecayDropoutPolicy(
            tau=1.0,
            severity=1.0,
            max_drop_probability=1.0,
            min_recent_items=0,
            min_history_items=0,
            preserve_target_level=False,
        )
        policy = TargetConditionedPolicy(
            base_policy=base_policy,
            same_level_restore_probability=1.0,
            precursor_restore_probability=1.0,
            min_history_items=0,
        )

        view = policy.generate_view(
            sequence,
            self.context,
            np.random.default_rng(0),
        )

        self.assertNotIn(0, view.keep_indices)
        self.assertIn(1, view.keep_indices)
        self.assertIn(2, view.keep_indices)

    def test_multi_view_generates_named_semantic_views(self):
        sequence = BehaviorSequence(
            items=["a", "b", "c", "d", "e"],
            behaviors=["pxs", "click", "conversion", "pxs", "click"],
            session_ids=[0, 0, 1, 2, 2],
            times=[0.0, 1.0, 2.0, 90.0, 99.0],
        )
        policy = MultiViewAugmentationPolicy(
            recent_policy=TimeDecayDropoutPolicy(
                tau=10.0,
                severity=1.0,
            ),
            session_policy=SessionAwareDropoutPolicy(
                recent_session_count=1,
                base_keep_probability=0.0,
                high_level_bonus=0.0,
            ),
        )

        views = policy.generate_views(
            sequence,
            self.context,
            np.random.default_rng(0),
        )
        views_by_name = {
            view.name: view
            for view in views
        }

        self.assertEqual(
            set(views_by_name),
            {
                "multi_view_recent",
                "multi_view_hierarchy",
                "multi_view_session",
            },
        )
        hierarchy_behaviors = {
            sequence.behaviors[index]
            for index in views_by_name[
                "multi_view_hierarchy"
            ].keep_indices
        }
        self.assertNotIn("pxs", hierarchy_behaviors)


if __name__ == "__main__":
    unittest.main()
