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
    def test_llm_pair_head_scores_history_item_and_user_features(self):
        class Model(CustomCausalLMWrapperMixin):
            def get_input_embeddings(self):
                return self.item_embedding

        model = Model()
        model.config = SimpleNamespace(
            ranking_score_type="llm_pair",
            ranking_candidate_len=2,
            num_positions=3,
        )
        model.item_embedding = torch.nn.Embedding(20, 4)
        model.ranking_user_embedding = torch.nn.Embedding(4, 4, padding_idx=0)
        model.ranking_head_dropout = torch.nn.Dropout(0.0)
        model.ranking_head = torch.nn.Linear(16, 1)

        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
        hidden_states = torch.arange(2 * 4 * 4, dtype=torch.float32).view(2, 4, 4)
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        logits = model._llm_pair_ranking_logits(
            input_ids=input_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            user_id=torch.tensor([1, 2]),
        )

        self.assertEqual(logits.shape, (2,))


if __name__ == "__main__":
    unittest.main()
