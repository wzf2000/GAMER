import unittest

from SeqRec.evaluation.ranking import binary_eval_results, binary_gauc


class BinaryGAUCTest(unittest.TestCase):
    def test_weighting_is_explicit_and_skips_single_class_users(self):
        labels = [1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1]
        scores = [1, 0, 0, 0, 1, 1, 3, 2, 1, 2, 1]
        user_ids = [1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4]

        self.assertAlmostEqual(binary_gauc(labels, scores, user_ids), 1 / 3)
        self.assertAlmostEqual(binary_gauc(labels, scores, user_ids, "macro"), 1 / 2)
        self.assertAlmostEqual(binary_gauc(labels, scores, user_ids, "pair"), 1 / 5)
        with self.assertRaises(ValueError):
            binary_gauc(labels, scores, user_ids, "unknown")

        result = binary_eval_results(labels, scores, user_ids, ["gauc"])
        self.assertEqual(result["gauc_total_users"], 4)
        self.assertEqual(result["gauc_valid_users"], 2)
        self.assertEqual(result["gauc_no_positive_users"], 1)
        self.assertEqual(result["gauc_no_negative_users"], 1)
        self.assertAlmostEqual(result["gauc_valid_user_ratio"], 1 / 2)
        self.assertAlmostEqual(result["gauc_valid_example_ratio"], 6 / 11)


if __name__ == "__main__":
    unittest.main()
