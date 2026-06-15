import json
import tempfile
import unittest
from pathlib import Path

from SeqRec.datasets.loaders.session_behavior import load_SMB_datasets
from SeqRec.datasets.session_behavior import (
    SMBExplicitDataset,
    SMBPolicyAugmentedDatasetForDecoder,
)


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


if __name__ == "__main__":
    unittest.main()
