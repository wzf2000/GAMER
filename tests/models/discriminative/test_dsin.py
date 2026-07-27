import unittest

try:
    import torch

    from SeqRec.models.discriminative.DSIN import DSIN, DSINConfig
except ModuleNotFoundError as exc:
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"torch-dependent DSIN tests skipped: {IMPORT_ERROR}")
class DSINTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = DSIN(
            DSINConfig(
                embedding_size=4,
                n_heads=2,
                lstm_hidden_size=2,
                attention_hidden_size=4,
                mlp_hidden_sizes=[4],
                max_sessions=3,
                dropout=0.0,
            ),
            n_items=8,
            max_his_len=5,
            n_behaviors=4,
        )
        self.model.eval()

    def test_extracts_real_sessions_and_predicts_one_logit_per_candidate(self):
        item_seq = torch.tensor([[1, 2, 3, 4, 0], [3, 4, 5, 0, 0]])
        session_ids = torch.tensor([[0, 0, 1, 2, 0], [4, 4, 5, 0, 0]])
        behaviors = torch.tensor([[1, 2, 1, 3, 0], [1, 1, 2, 0, 0]])
        candidate_item = torch.tensor([5, 2])

        interests, mask = self.model._extract_session_interests(item_seq, session_ids)

        self.assertEqual(interests.shape, torch.Size([2, 3, 4]))
        torch.testing.assert_close(
            mask,
            torch.tensor([[True, True, True], [True, True, False]]),
        )
        self.assertEqual(
            self.model(item_seq, behaviors, candidate_item, session_ids).shape,
            torch.Size([2]),
        )

    def test_behavior_sequence_is_not_used(self):
        item_seq = torch.tensor([[1, 2, 3, 4, 0]])
        session_ids = torch.tensor([[0, 0, 1, 2, 0]])
        candidate_item = torch.tensor([5])
        behavior_a = torch.tensor([[1, 1, 1, 1, 0]])
        behavior_b = torch.tensor([[4, 3, 2, 1, 0]])

        torch.testing.assert_close(
            self.model(item_seq, behavior_a, candidate_item, session_ids),
            self.model(item_seq, behavior_b, candidate_item, session_ids),
        )


if __name__ == "__main__":
    unittest.main()
