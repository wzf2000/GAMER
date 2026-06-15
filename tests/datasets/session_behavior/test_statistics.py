import unittest

from SeqRec.datasets.session_behavior.statistics import (
    compute_training_level_statistics,
)


class BehaviorLevelStatisticsTest(unittest.TestCase):
    def test_statistics_use_only_training_prefixes(self):
        statistics = compute_training_level_statistics(
            histories={
                "u1": ["pxs", "click", "conversion", "conversion"],
                "u2": ["pxs", "pxs", "click"],
            },
            valid_positions={"u1": 2, "u2": 1},
            behavior_level={
                "pxs": 0,
                "click": 1,
                "conversion": 2,
            },
            max_behavior_level=2,
        )

        self.assertEqual(statistics.level_counts, (2, 1, 0))
        self.assertEqual(statistics.total_interactions, 3)
        self.assertEqual(statistics.user_count, 2)


if __name__ == "__main__":
    unittest.main()
