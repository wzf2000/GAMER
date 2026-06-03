"""
Shared scaffolding for SMB analysis tasks.

Subclasses provide their own argparse, ``invoke`` orchestration, and result
aggregation; this base removes the model-load / trie-build / beam-search /
rank-extraction boilerplate that ``sparse_behavior`` and ``behavior_dropout``
otherwise both copy.
"""

import copy
import torch
from typing import TYPE_CHECKING

from SeqRec.tasks.multi_gpu import MultiGPUTask
from SeqRec.datasets.session_behavior import BaseSMBDataset
from SeqRec.datasets.collators.generative import DecoderOnlyTestCollator, EncoderDecoderTestCollator
from SeqRec.models.generative.registry import (
    is_decoder_only_backbone,
    load_model_and_tokenizer,
)
from SeqRec.tasks.evaluation.helpers import (
    build_behavior_prefix_fns,
    build_generation_kwargs,
    get_generation_model,
    get_item_token_info,
    prepare_behavior_generation_prompt,
    slice_decoder_only_output,
)

if TYPE_CHECKING:
    from transformers.generation.utils import GenerateBeamOutput
    from transformers import BatchEncoding


class _BaseAnalysisTask(MultiGPUTask):
    """Shared base for tasks that load an SMB generative model and run beam-search
    inference for analysis purposes. Subclasses must call ``_setup_backbone`` before
    ``_build_tries`` / ``_run_inference``; analysis tasks that swap between two
    models (e.g. sparse_behavior's ours-vs-baseline comparison) call it again
    after each ``_load_model_and_tokenizer``.
    """

    def _setup_backbone(self, backbone: str):
        self.backbone = backbone
        self._is_decoder_only = is_decoder_only_backbone(backbone)

    def _load_model_and_tokenizer(self, backbone: str, ckpt_path: str):
        self.model, self.tokenizer = load_model_and_tokenizer(backbone, ckpt_path)
        self.model = self.model.to(self.device)
        self.config = self.model.config

    def _make_test_collator(self):
        return (
            DecoderOnlyTestCollator(self.tokenizer)
            if self._is_decoder_only
            else EncoderDecoderTestCollator(self.tokenizer)
        )

    def _build_tries(self, base_dataset: BaseSMBDataset):
        """Pre-build per-behavior candidate tries for prefix-constrained decoding."""
        all_behavior_items = base_dataset.get_all_items("all")
        item_reps = list(all_behavior_items)
        _, self.item_len, last_token_set = get_item_token_info(
            self.tokenizer,
            item_reps,
            self.config.pad_token_id,
        )
        self.sole_item_len = len(
            self.tokenizer.encode(
                next(iter(base_dataset.get_all_items())), add_special_tokens=False
            )
        )

        self.prefix_fn_by_behavior = build_behavior_prefix_fns(
            backbone=self.backbone,
            tokenizer=self.tokenizer,
            config=self.config,
            dataset=base_dataset,
            last_token_set=last_token_set,
        )

    def _run_inference(
        self,
        inputs: "BatchEncoding",
        behavior: str,
        dataset: BaseSMBDataset,
        num_beams: int,
    ) -> tuple[list[str], torch.Tensor, int]:
        """Append the behavior token, run beam search, return (output_str, scores, beh_token_num).

        ``beh_token_num`` is only consumed by ``behavior_dropout``; ``sparse_behavior``
        unpacks with ``_`` to ignore it.
        """
        batch_size = inputs.input_ids.shape[0]
        inp = copy.copy(inputs)

        gen_model = get_generation_model(self.model)
        decoder_input_ids, beh_token_num = prepare_behavior_generation_prompt(
            inputs=inp,
            tokenizer=self.tokenizer,
            dataset=dataset,
            behaviors=[behavior] * batch_size,
            device=self.device,
            is_decoder_only=self._is_decoder_only,
            decoder_start_token_id=self.config.decoder_start_token_id,
        )
        gen_kwargs = build_generation_kwargs(
            backbone=self.backbone,
            inputs=inp,
            max_new_tokens=self.sole_item_len,
            prefix_allowed_tokens_fn=self.prefix_fn_by_behavior[behavior],
            num_beams=num_beams,
            device=self.device,
            decoder_input_ids=decoder_input_ids,
        )

        output: "GenerateBeamOutput" = gen_model.generate(**gen_kwargs)
        output_ids = output.sequences
        scores = output.sequences_scores
        output_ids = slice_decoder_only_output(self.backbone, output_ids, self.item_len)
        output_str = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return output_str, scores, beh_token_num

    @staticmethod
    def _compute_ranks(
        output_str: list[str],
        scores: torch.Tensor,
        targets: list[list[str]],
        num_beams: int,
    ) -> list[tuple[int, float | None]]:
        """Return per-sample (best_rank, best_score) over beam candidates.

        Rank is 1-indexed; ``num_beams + 1`` (with score ``None``) means no
        target item appears among the candidates.
        """
        results = []
        for i, target_list in enumerate(targets):
            cands = [
                s.replace(" ", "")
                for s in output_str[i * num_beams: (i + 1) * num_beams]
            ]
            cand_scores = scores[i * num_beams: (i + 1) * num_beams]
            target_set = {t.replace(" ", "") for t in target_list}
            best_rank, best_score = num_beams + 1, None
            for rank, (cand, sc) in enumerate(zip(cands, cand_scores), start=1):
                if cand in target_set:
                    best_rank = rank
                    best_score = float(sc)
                    break
            results.append((best_rank, best_score))
        return results
