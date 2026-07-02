import json
import tempfile
import unittest
from pathlib import Path

from SeqRec.datasets.loaders.session_behavior import load_SMB_datasets
from SeqRec.datasets.session_behavior import (
    SMBExplicitDataset,
    SMBPolicyAugmentedDatasetForDecoder,
)
from SeqRec.datasets.session_behavior.augmentation_policies import (
    AugmentedView,
    BehaviorSequence,
)


class DropTailPolicy:
    name = "drop_tail"

    def generate_views(self, sequence, context, rng):
        return [
            AugmentedView(
                name=self.name,
                keep_indices=list(range(len(sequence.items) - 1)),
                metadata={},
            )
        ]

    def cache_config(self):
        return {"name": self.name}


class DropTailPolicyDataset(SMBPolicyAugmentedDatasetForDecoder):
    def _build_policy(self):
        return DropTailPolicy()


class PolicyAugmentedDatasetTest(unittest.TestCase):
    def _write_json(self, path: Path, value):
        with path.open("w") as file:
            json.dump(value, file)

    def _build_dataset_files(self, root: Path):
        dataset_dir = root / "TinySMB"
        dataset_dir.mkdir()
        prefix = dataset_dir / "TinySMB"
        self._write_json(
            Path(f"{prefix}.SMB.inter.json"),
            {"u1": list(range(8))},
        )
        self._write_json(
            Path(f"{prefix}.SMB.behavior.json"),
            {
                "u1": [
                    "pxs",
                    "pxs",
                    "click",
                    "conversion",
                    "pxs",
                    "click",
                    "conversion",
                    "conversion",
                ],
            },
        )
        self._write_json(
            Path(f"{prefix}.SMB.session.json"),
            {"u1": [10, 10, 11, 11, 12, 12, 13, 13]},
        )
        self._write_json(
            Path(f"{prefix}.SMB.time.json"),
            {
                "u1": [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:30:00",
                    "2026-01-03 00:00:00",
                    "2026-01-03 00:30:00",
                    "2026-01-04 00:00:00",
                    "2026-01-04 00:30:00",
                    "2026-01-05 00:00:00",
                    "2026-01-05 00:30:00",
                ],
            },
        )
        self._write_json(
            Path(f"{prefix}.behavior_level.json"),
            {"pxs": 0, "click": 1, "conversion": 2},
        )
        self._write_json(
            Path(f"{prefix}.index.json"),
            {
                str(index): [f"<item_{index}>"]
                for index in range(8)
            },
        )

    def test_loader_builds_augmented_train_and_original_valid_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._build_dataset_files(root)

            train_data, valid_data = load_SMB_datasets(
                dataset="TinySMB",
                data_path=str(root),
                max_his_len=100,
                index_file=".index.json",
                tasks="smb_policy_decoder",
                sequence_augmentation_config={
                    "sequence_augmentation": "time_decay",
                    "augmentation_views": 2,
                    "augmentation_seed": 7,
                    "augmentation_drop_original": False,
                    "augmentation_config": {
                        "time_decay_tau": 1.0,
                        "time_decay_severity": 1.0,
                        "time_decay_max_drop": 1.0,
                        "time_decay_min_recent_items": 1,
                    },
                },
            )

            train_dataset = train_data.datasets[0]
            self.assertIsInstance(
                train_dataset,
                SMBPolicyAugmentedDatasetForDecoder,
            )
            self.assertIsInstance(valid_data, SMBExplicitDataset)
            self.assertNotIsInstance(
                valid_data,
                SMBPolicyAugmentedDatasetForDecoder,
            )
            self.assertGreaterEqual(len(train_dataset), 1)
            self.assertEqual(
                set(train_dataset[0]),
                {
                    "input_ids",
                    "labels",
                    "session_ids",
                    "extended_session_ids",
                    "actions",
                    "time",
                    "behavior",
                    "inters_item_list",
                    "split",
                },
            )
            self.assertIn("time_decay", train_dataset.cached_file_name)

    def test_loader_requires_policy_configuration(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._build_dataset_files(root)

            with self.assertRaises(ValueError):
                load_SMB_datasets(
                    dataset="TinySMB",
                    data_path=str(root),
                    max_his_len=100,
                    index_file=".index.json",
                    tasks="smb_policy_decoder",
                )

    def test_loader_builds_each_static_policy(self):
        policy_configs = {
            "time_decay": {},
            "session": {},
            "dataset_proportion": {
                "dataset_proportion_preset": "balanced",
            },
            "user_adaptive_ratio": {},
            "target_conditioned": {},
            "multi_view": {},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._build_dataset_files(root)

            for policy_name, policy_config in policy_configs.items():
                with self.subTest(policy=policy_name):
                    train_data, _ = load_SMB_datasets(
                        dataset="TinySMB",
                        data_path=str(root),
                        max_his_len=100,
                        index_file=".index.json",
                        tasks="smb_policy_decoder",
                        sequence_augmentation_config={
                            "sequence_augmentation": policy_name,
                            "augmentation_views": 1,
                            "augmentation_seed": 7,
                            "augmentation_drop_original": False,
                            "augmentation_config": policy_config,
                        },
                    )
                    train_dataset = train_data.datasets[0]
                    self.assertEqual(
                        train_dataset.augmentation_policy.name,
                        policy_name,
                    )
                    self.assertGreaterEqual(len(train_dataset), 1)

    def test_training_statistics_exclude_validation_and_test_history(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._build_dataset_files(root)

            train_data, _ = load_SMB_datasets(
                dataset="TinySMB",
                data_path=str(root),
                max_his_len=100,
                index_file=".index.json",
                tasks="smb_policy_decoder",
                sequence_augmentation_config={
                    "sequence_augmentation": "user_adaptive_ratio",
                    "augmentation_views": 1,
                    "augmentation_seed": 7,
                    "augmentation_drop_original": False,
                    "augmentation_config": {},
                },
            )
            statistics = train_data.datasets[0].training_statistics

            self.assertEqual(statistics.level_counts, (2, 1, 1))
            self.assertEqual(statistics.total_interactions, 4)

    def test_multi_view_can_include_random_ratio_views(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._build_dataset_files(root)

            common_config = {
                "sequence_augmentation": "multi_view",
                "augmentation_views": 1,
                "augmentation_seed": 7,
                "augmentation_drop_original": False,
            }
            pure_train_data, _ = load_SMB_datasets(
                dataset="TinySMB",
                data_path=str(root),
                max_his_len=100,
                index_file=".index.json",
                tasks="smb_policy_decoder",
                sequence_augmentation_config={
                    **common_config,
                    "augmentation_config": {},
                },
            )
            hybrid_train_data, _ = load_SMB_datasets(
                dataset="TinySMB",
                data_path=str(root),
                max_his_len=100,
                index_file=".index.json",
                tasks="smb_policy_decoder",
                sequence_augmentation_config={
                    **common_config,
                    "augmentation_config": {
                        "multi_view_random_ratio_views": 2,
                    },
                },
            )

            pure_dataset = pure_train_data.datasets[0]
            hybrid_dataset = hybrid_train_data.datasets[0]
            self.assertEqual(
                hybrid_dataset._multi_view_random_ratio_views(),
                2,
            )
            self.assertEqual(
                len(pure_dataset._generate_random_ratio_views(
                    BehaviorSequence(
                        items=["a", "b", "c", "d"],
                        behaviors=["pxs", "pxs", "click", "conversion"],
                        session_ids=[0, 0, 1, 1],
                        times=[0.0, 1.0, 2.0, 3.0],
                    ),
                    pure_dataset._view_rng("u1", 1_000_000),
                )),
                0,
            )
            random_ratio_views = hybrid_dataset._generate_random_ratio_views(
                BehaviorSequence(
                    items=["a", "b", "c", "d"],
                    behaviors=["pxs", "pxs", "click", "conversion"],
                    session_ids=[0, 0, 1, 1],
                    times=[0.0, 1.0, 2.0, 3.0],
                ),
                hybrid_dataset._view_rng("u1", 1_000_000),
            )
            self.assertEqual(
                [name for name, _ in random_ratio_views],
                [
                    "multi_view_random_ratio_1_of_2",
                    "multi_view_random_ratio_2_of_2",
                ],
            )
            self.assertTrue(
                all(indices for _, indices in random_ratio_views),
            )

    def test_policy_views_are_built_from_full_sequences(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._build_dataset_files(root)

            dataset = DropTailPolicyDataset(
                dataset="TinySMB",
                data_path=str(root),
                max_his_len=100,
                index_file=".index.json",
                mode="train",
                sequence_augmentation="drop_tail",
                augmentation_views=1,
                augmentation_seed=7,
                augmentation_drop_original=False,
            )

            self.assertEqual(len(dataset.inter_data), 2)
            original_sample = dataset.inter_data[0]
            augmented_sample = dataset.inter_data[1]
            self.assertEqual(original_sample["behavior"], "conversion")
            self.assertEqual(augmented_sample["behavior"], "click")
            self.assertEqual(
                augmented_sample["item"],
                "<behavior_click><item_2>",
            )
            self.assertEqual(
                augmented_sample["inters"],
                "<behavior_pxs><item_0><behavior_pxs><item_1>",
            )

    def test_single_interaction_training_sequence_is_valid(self):
        sequence = BehaviorSequence(
            items=["<item_0>"],
            behaviors=["pxs"],
            session_ids=[0],
            times=[0.0],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._build_dataset_files(root)
            dataset = DropTailPolicyDataset(
                dataset="TinySMB",
                data_path=str(root),
                max_his_len=100,
                index_file=".index.json",
                mode="train",
                sequence_augmentation="drop_tail",
                augmentation_views=1,
                augmentation_seed=7,
                augmentation_drop_original=True,
            )

            sample = dataset._build_sample(sequence)

        self.assertEqual(sample["inters"], "")
        self.assertEqual(sample["item"], "<behavior_pxs><item_0>")
        self.assertEqual(sample["behavior"], "pxs")


if __name__ == "__main__":
    unittest.main()
