import unittest
from unittest.mock import patch
from types import SimpleNamespace

try:
    import torch

    from SeqRec.models.generative.common.wrappers import CustomCausalLMWrapperMixin
    from SeqRec.tasks.training.train_SMB_ranking_decoder import (
        TrainSMBRankingDecoder,
        _freeze_for_ranking_probe,
    )

    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only in lightweight envs
    torch = None
    CustomCausalLMWrapperMixin = None
    IMPORT_ERROR = exc


@unittest.skipIf(torch is None, f"torch-dependent ranking head tests skipped: {IMPORT_ERROR}")
class RankingHeadFeatureTest(unittest.TestCase):
    def test_hidden_head_uses_last_unpadded_candidate_state(self):
        class Model(CustomCausalLMWrapperMixin):
            pass

        model = Model()
        model.ranking_head_dropout = torch.nn.Dropout(0.0)
        model.ranking_head = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.ranking_head.weight.copy_(torch.tensor([[1.0, 0.0]]))

        hidden_states = torch.tensor([
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]],
        ])
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

        logits = model._ranking_logits_from_hidden_states(hidden_states, attention_mask)

        torch.testing.assert_close(logits, torch.tensor([3.0, 5.0]))

    def test_frozen_probe_keeps_only_linear_head_trainable(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(2, 2)
                self.lm_head = torch.nn.Linear(2, 3)
                self.ranking_head = torch.nn.Linear(2, 1)
                self.config = SimpleNamespace()

        model = Model()
        _freeze_for_ranking_probe(model)

        self.assertFalse(any(parameter.requires_grad for parameter in model.model.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.lm_head.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.ranking_head.parameters()))
        self.assertTrue(model.config.ranking_freeze_backbone)

    def test_full_finetune_keeps_pretrained_decoder_trainable(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(2, 2)
                self.lm_head = torch.nn.Linear(2, 3)
                self.ranking_head = torch.nn.Linear(2, 1)
                self.config = SimpleNamespace(ranking_freeze_backbone=True)

        model = Model()
        for parameter in model.parameters():
            parameter.requires_grad = False

        with patch.dict("os.environ", {"SMB_RANKING_FREEZE_BACKBONE": "0"}):
            TrainSMBRankingDecoder().configure_model(
                model,
                SimpleNamespace(pretrained_model="test-checkpoint"),
                SimpleNamespace(),
            )

        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(model.config.ranking_freeze_backbone)

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
