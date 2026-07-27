import unittest

try:
    import torch

    from SeqRec.models.discriminative.BSTCVR.config import BSTCVRConfig
    from SeqRec.models.discriminative.BSTCVR.model import BSTCVR
except ModuleNotFoundError as exc:
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"torch-dependent BSTCVR tests skipped: {IMPORT_ERROR}")
class BSTCVRTest(unittest.TestCase):
    def test_candidate_is_appended_as_target_token(self):
        class Recorder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.last_hidden_states = None

            def forward(self, hidden_states, attention_mask):
                self.last_hidden_states = hidden_states.detach().clone()
                return hidden_states

        model = BSTCVR(
            BSTCVRConfig(n_layers=1, n_heads=1, hidden_size=2, inner_size=4, dropout_prob=0.0),
            n_items=4,
            max_his_len=2,
            n_behaviors=4,
        )
        recorder = Recorder()
        model.trm_encoder = recorder
        with torch.no_grad():
            model.item_embedding.weight.zero_()
            model.item_embedding.weight[3] = torch.tensor([5.0, 7.0])
            model.position_embedding.weight.zero_()
            model.LayerNorm.weight.fill_(1.0)
            model.LayerNorm.bias.zero_()
            model.output.weight.copy_(torch.tensor([[1.0, 1.0]]))
            model.output.bias.zero_()

        logits = model(
            torch.tensor([[1, 2]], dtype=torch.long),
            torch.tensor([[1, 1]], dtype=torch.long),
            torch.tensor([3], dtype=torch.long),
        )

        self.assertEqual(logits.shape, torch.Size([1]))
        torch.testing.assert_close(recorder.last_hidden_states[:, -1], model.LayerNorm(model.item_embedding(torch.tensor([3]))))


if __name__ == "__main__":
    unittest.main()

