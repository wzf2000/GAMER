import unittest

try:
    import torch

    from SeqRec.models.discriminative.HSTUCVR.config import HSTUCVRConfig
    from SeqRec.models.discriminative.HSTUCVR.model import HSTUCVR
except ModuleNotFoundError as exc:
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"torch-dependent HSTUCVR tests skipped: {IMPORT_ERROR}")
class HSTUCVRTest(unittest.TestCase):
    def test_base_config_uses_requested_depth_and_heads(self):
        config = HSTUCVRConfig.from_pretrained("config/dis-models/HSTUCVR")

        self.assertEqual(config.n_layers, 16)
        self.assertEqual(config.n_heads, 8)

    def test_forward_and_loss_accept_smb_din_batch(self):
        model = HSTUCVR(
            HSTUCVRConfig(n_layers=1, n_heads=2, hidden_size=4, dropout_prob=0.0),
            n_items=5,
            max_his_len=3,
            n_behaviors=4,
        )
        batch = {
            "inputs": torch.tensor([[1, 2, 0], [2, 3, 4]], dtype=torch.long),
            "behaviors": torch.tensor([[1, 2, 0], [2, 3, 4]], dtype=torch.long),
            "candidate_item": torch.tensor([3, 5], dtype=torch.long),
            "label": torch.tensor([1.0, 0.0]),
        }

        logits = model.predict(batch)
        loss = model.calculate_loss(batch)

        self.assertEqual(logits.shape, torch.Size([2]))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()

