import unittest

try:
    import torch

    from SeqRec.models.discriminative.DIN.config import DINConfig
    from SeqRec.models.discriminative.DIN.model import DIN
except ModuleNotFoundError as exc:
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"torch-dependent DIN tests skipped: {IMPORT_ERROR}")
class DINItemOnlyTest(unittest.TestCase):
    def test_behavior_sequence_does_not_change_logits(self):
        torch.manual_seed(0)
        model = DIN(DINConfig(), n_items=8, max_his_len=3, n_behaviors=4)
        item_seq = torch.tensor([[1, 2, 3], [3, 4, 0]], dtype=torch.long)
        candidate_item = torch.tensor([4, 2], dtype=torch.long)
        behavior_a = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)
        behavior_b = torch.tensor([[4, 3, 2], [2, 4, 0]], dtype=torch.long)

        self.assertFalse(hasattr(model, "behavior_embedding"))
        torch.testing.assert_close(
            model(item_seq, behavior_a, candidate_item),
            model(item_seq, behavior_b, candidate_item),
        )


if __name__ == "__main__":
    unittest.main()
