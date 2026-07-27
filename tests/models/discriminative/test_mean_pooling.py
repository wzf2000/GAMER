import unittest

try:
    import torch

    from SeqRec.models.discriminative.MeanPooling.config import MeanPoolingConfig
    from SeqRec.models.discriminative.MeanPooling.model import MeanPooling
except ModuleNotFoundError as exc:
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"torch-dependent MeanPooling tests skipped: {IMPORT_ERROR}")
class MeanPoolingTest(unittest.TestCase):
    def test_forward_uses_item_mean_and_candidate_dot(self):
        model = MeanPooling(
            MeanPoolingConfig(embedding_size=2),
            n_items=4,
            max_his_len=3,
            n_behaviors=4,
        )
        with torch.no_grad():
            model.item_embedding.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [2.0, 2.0],
                        [1.0, 1.0],
                    ]
                )
            )

        item_seq = torch.tensor([[1, 2, 0]], dtype=torch.long)
        candidate_item = torch.tensor([3], dtype=torch.long)
        behavior_a = torch.tensor([[1, 2, 0]], dtype=torch.long)
        behavior_b = torch.tensor([[3, 4, 0]], dtype=torch.long)

        expected = torch.tensor([2.0])
        torch.testing.assert_close(model(item_seq, behavior_a, candidate_item), expected)
        torch.testing.assert_close(
            model(item_seq, behavior_a, candidate_item),
            model(item_seq, behavior_b, candidate_item),
        )


if __name__ == "__main__":
    unittest.main()

