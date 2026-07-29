from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from SeqRec.datasets.discriminative.session_behavior import SMBDINDataset
from SeqRec.datasets.loaders.session_behavior_discriminative import (
    load_SMBDis_datasets,
    load_SMBDis_test_dataset,
)


class SMBDiscriminativeCTRDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmp.name)
        dataset_dir = self.data_root / "Tiny"
        dataset_dir.mkdir()
        prefix = dataset_dir / "Tiny"
        files = {
            f"{prefix}.SMB.inter.json": {
                "0": [0, 1, 2, 3, 4, 5],
            },
            f"{prefix}.SMB.behavior.json": {
                "0": ["p3s", "p3s", "click", "cvr", "click", "cvr"],
            },
            f"{prefix}.SMB.session.json": {
                "0": [10, 20, 20, 20, 30, 40],
            },
            f"{prefix}.SMB.time.json": {
                "0": [
                    "2026-01-01 00:00:00",
                    "2026-01-02 00:00:00",
                    "2026-01-02 00:30:00",
                    "2026-01-02 01:00:00",
                    "2026-01-03 00:00:00",
                    "2026-01-04 00:00:00",
                ],
            },
            f"{prefix}.behavior_level.json": {
                "p3s": 0,
                "click": 1,
                "cvr": 2,
            },
        }
        for path, content in files.items():
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(content, handle)

    def tearDown(self):
        self.tmp.cleanup()

    def test_ctr_labels_and_cache_are_distinct_from_cvr(self):
        common = {
            "dataset": "Tiny",
            "data_path": str(self.data_root),
            "max_his_len": 100,
            "mode": "train",
            "diff": False,
            "add_uid": True,
        }
        cvr = SMBDINDataset(**common)
        ctr = SMBDINDataset(**common, positive_behavior="click")

        self.assertEqual([sample["label"] for sample in cvr], [0.0, 0.0, 1.0])
        self.assertEqual([sample["label"] for sample in ctr], [0.0, 1.0, 1.0])
        self.assertEqual(ctr.target_behavior, "click")
        self.assertEqual(ctr.ranking_task_name, "CTR")
        self.assertNotEqual(cvr.cached_file_name, ctr.cached_file_name)

    def test_ctr_loader_tasks_cover_din_and_dsin(self):
        train, valid = load_SMBDis_datasets(
            dataset="Tiny",
            data_path=str(self.data_root),
            max_his_len=100,
            tasks="smb_ctr_din",
            add_uid=True,
        )
        test = load_SMBDis_test_dataset(
            dataset="Tiny",
            data_path=str(self.data_root),
            max_his_len=100,
            test_task="smb_ctr_din",
            add_uid=True,
        )
        dsin_train, dsin_valid = load_SMBDis_datasets(
            dataset="Tiny",
            data_path=str(self.data_root),
            max_his_len=100,
            tasks="smb_ctr_dsin",
            add_uid=True,
        )

        self.assertEqual(train.datasets[0].target_behavior, "click")
        self.assertEqual(valid.target_behavior, "click")
        self.assertEqual(test.target_behavior, "click")
        self.assertEqual(dsin_train.datasets[0].target_behavior, "click")
        self.assertEqual(dsin_valid.target_behavior, "click")


if __name__ == "__main__":
    unittest.main()
