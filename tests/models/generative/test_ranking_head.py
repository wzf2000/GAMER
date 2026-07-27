import unittest
from types import SimpleNamespace

try:
    import torch

    from SeqRec.models.generative.common.wrappers import CustomCausalLMWrapperMixin

    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only in lightweight envs
    torch = None
    CustomCausalLMWrapperMixin = None
    IMPORT_ERROR = exc


@unittest.skipIf(torch is None, f"torch-dependent ranking head tests skipped: {IMPORT_ERROR}")
class RankingHeadFeatureTest(unittest.TestCase):
    def test_llm_pair_head_uses_candidate_hidden_states(self):
        class Model(CustomCausalLMWrapperMixin):
            pass

        model = Model()
        model.config = SimpleNamespace(
            ranking_score_type="llm_pair",
            ranking_candidate_len=2,
            num_positions=3,
            ranking_use_user_embedding=False,
        )
        model.ranking_head_dropout = torch.nn.Dropout(0.0)
        model.ranking_head = torch.nn.Linear(12, 1, bias=False)
        with torch.no_grad():
            model.ranking_head.weight.zero_()
            model.ranking_head.weight[0, 4] = 1.0

        hidden_states = torch.arange(2 * 4 * 4, dtype=torch.float32).view(2, 4, 4)
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        logits = model._llm_pair_ranking_logits(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            user_id=None,
        )

        torch.testing.assert_close(logits, torch.tensor([10.0, 18.0]))


if __name__ == "__main__":
    unittest.main()
