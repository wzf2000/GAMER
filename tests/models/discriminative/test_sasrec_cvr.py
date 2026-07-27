import unittest

try:
    import torch

    from SeqRec.models.discriminative.SASRecCVR.config import SASRecCVRConfig
    from SeqRec.models.discriminative.SASRecCVR.model import SASRecCVR
except ModuleNotFoundError as exc:
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"torch-dependent SASRecCVR tests skipped: {IMPORT_ERROR}")
class SASRecCVRTest(unittest.TestCase):
    def test_predict_scores_candidate_from_sasrec_output(self):
        class Model(SASRecCVR):
            def forward(self, item_seq, item_seq_len):
                return torch.tensor([[1.0, 2.0]], device=item_seq.device)

        model = Model(
            SASRecCVRConfig(hidden_size=2, inner_size=4, n_heads=1),
            n_items=3,
            max_his_len=2,
            n_behaviors=4,
        )
        with torch.no_grad():
            model.item_embedding.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [3.0, 4.0],
                    ]
                )
            )

        interaction = {
            "inputs": torch.tensor([[1, 2]], dtype=torch.long),
            "behaviors": torch.tensor([[1, 2]], dtype=torch.long),
            "seq_len": torch.tensor([2], dtype=torch.long),
            "candidate_item": torch.tensor([3], dtype=torch.long),
            "label": torch.tensor([1.0]),
        }

        torch.testing.assert_close(model.predict(interaction), torch.tensor([11.0]))


if __name__ == "__main__":
    unittest.main()

