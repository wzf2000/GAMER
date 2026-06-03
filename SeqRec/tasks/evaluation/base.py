"""
Shared scaffolding for generative decoder test tasks.

Subclasses still own:
  - argparse (different flags per family)
  - dataset / collator construction
  - prefix-allowed-tokens trie construction
  - check_collision_items (different per-sample target shape)
  - test_single_* core inference loop (numerically sensitive)
  - test() orchestration

This base removes the model-load if/elif chain, DDP setup, repeated gather
boilerplate, validation loop, user-level metric save, and final
results/log save that the three test_*_decoder tasks otherwise all copy.
"""

import os
import json
import torch
import numpy as np
import torch.distributed as dist
from loguru import logger
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from SeqRec.tasks.multi_gpu import MultiGPUTask
from SeqRec.evaluation.ranking import get_topk_results, get_metrics_results
from SeqRec.models.generative.registry import load_model_and_tokenizer
from SeqRec.utils.fs import ensure_dir
from SeqRec.utils.runtime import get_tqdm


class _BaseDecoderTestTask(MultiGPUTask):
    """Shared base for ``test_decoder`` / ``test_MB_decoder`` / ``test_SMB_decoder``."""

    def _load_model_via_registry(self, backbone: str, ckpt_path: str):
        """Load model + tokenizer through the generative backbone registry."""
        self.model, self.tokenizer = load_model_and_tokenizer(backbone, ckpt_path)
        self.model = self.model.to(self.device)
        self.config = self.model.config
        from transformers.generation import GenerationMixin
        assert isinstance(self.model, GenerationMixin), "Model must be a generation model."

    def _setup_ddp_for_datasets(self, datasets) -> list:
        """Return samplers (one per dataset). In DDP mode also wraps the model in SyncBN + DDP."""
        if self.ddp:
            samplers = [
                DistributedSampler(
                    d,
                    num_replicas=self.world_size,
                    rank=self.local_rank,
                    shuffle=False,
                )
                for d in datasets
            ]
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            self.model = DDP(self.model, device_ids=[self.local_rank])
            return samplers
        return [None] * len(datasets)

    def _gather_sum(self, value):
        """Return sum-across-ranks for a scalar value (or value itself when not DDP)."""
        if not self.ddp:
            return value
        gather_list = [None] * self.world_size
        dist.all_gather_object(obj=value, object_list=gather_list)
        return sum(gather_list)

    def _gather_concat(self, lst):
        """Return concatenated lists across ranks (or the list itself when not DDP)."""
        if not self.ddp:
            return lst
        gather_list = [None] * self.world_size
        dist.all_gather_object(obj=lst, object_list=gather_list)
        out = []
        for sub in gather_list:
            out += sub
        return out

    def validation(self):
        """Shared validation loss loop. Subclasses must populate ``self.loaders``."""
        for i, loader in enumerate(self.loaders):
            pbar = get_tqdm(desc=f"Validating {i}", total=len(loader))
            losses = []
            for batch in loader:
                batch = batch.to(self.device)
                output = self.model(**batch)
                assert "loss" in output, "Model output must contain 'loss' for validation."
                losses.append(output["loss"].item())
                if pbar:
                    pbar.set_postfix({"Average loss": f"{np.mean(losses):.4f}"})
                    pbar.update(1)
            if pbar:
                pbar.close()
            self.info(f"Validation loss: {np.mean(losses):.4f} for dataset {i}.")

    def _pbar_step(self, pbar, *, results: dict[str, float], total: int, extras: dict[str, str] | None = None):
        """Update the progress bar with the first two running metrics (rank 0
        only) and synchronize ranks afterwards. ``extras`` lets a subclass
        append task-specific postfix fields (e.g. SMB's duplicate ratio)."""
        if self.local_rank == 0:
            show_metric_dict = {
                m: f"{results[m] / total:.4f}"
                for m in self.metric_list[:2]
                if m in results
            }
            if extras:
                show_metric_dict.update(extras)
            pbar.set_postfix(show_metric_dict)
            pbar.update(1)
        if self.ddp:
            dist.barrier()

    def _finalize_loop_metrics(
        self,
        *,
        results: dict[str, float],
        total: int,
        user_metric_dict: dict[str, dict[int, float]],
        dataset_len: int,
        save_path: str,
    ):
        """End-of-loop wrap-up: divide accumulated metrics by total, then save
        per-uid metrics. ``results`` is mutated in place."""
        if self.ddp:
            dist.barrier()
        for m in results:
            if isinstance(results[m], (int, float)):
                results[m] = results[m] / total
        self._save_user_metrics(user_metric_dict, dataset_len, save_path, results)

    def _accumulate_batch_metrics(
        self,
        *,
        output_str: list[str],
        scores: torch.Tensor,
        targets,
        inputs,
        batch_size: int,
        num_beams: int,
        results: dict[str, float],
        user_metric_dict: dict[str, dict[int, float]],
        multi_target: bool = False,
    ) -> int:
        """Run the shared loop tail used by every ``test_single_*`` method.

        Steps (equivalent to the previous inlined code):
          1. ``get_topk_results`` from local outputs.
          2. ``_gather_sum`` the batch size and ``_gather_concat`` the topk list
             (and the targets list, only when ``multi_target=True``).
          3. ``_gather_concat`` ``inputs['uid']`` if present, and track per-uid
             metric values in ``user_metric_dict``.
          4. Sum the per-batch metric values into ``results`` (mutated in place).

        Returns the delta to add to the caller's running ``total`` count.
        ``multi_target=True`` is for SMB where ``get_metrics_results`` needs the
        gathered targets list (multi-label targets).
        """
        topk_res = get_topk_results(output_str, scores, targets, num_beams)
        delta_total = self._gather_sum(batch_size)
        topk_res = self._gather_concat(topk_res)
        if multi_target:
            targets = self._gather_concat(targets)
        uid = self._gather_concat(inputs["uid"]) if "uid" in inputs else None

        metric_args = (topk_res, self.metric_list) + ((targets,) if multi_target else ())
        if uid is not None:
            batch_metrics_res = get_metrics_results(*metric_args, list_output=True)
            for m in batch_metrics_res:
                for i in range(len(uid)):
                    user_metric_dict[m][uid[i]] = batch_metrics_res[m][i]
            batch_metrics_res = {m: sum(v) for m, v in batch_metrics_res.items()}
        else:
            batch_metrics_res = get_metrics_results(*metric_args, list_output=False)

        for m, res in batch_metrics_res.items():
            if m not in results:
                results[m] = res
            else:
                results[m] += res
        return delta_total

    def _save_user_metrics(
        self,
        user_metric_dict: dict[str, dict[int, float]],
        dataset_len: int,
        save_path: str,
        results: dict[str, float],
    ):
        """If any per-uid metric was tracked, sort by uid, dump JSON, and rewrite
        ``results[m]`` with the mean so the DistributedSampler duplicates don't
        get counted twice."""
        if len(user_metric_dict[self.metric_list[0]]) == 0:
            return
        ensure_dir(os.path.dirname(save_path))
        user_metric_list: dict[str, list[float]] = {}
        for m in user_metric_dict:
            sorted_uids = sorted(user_metric_dict[m].keys())
            user_metric_list[m] = [user_metric_dict[m][uid] for uid in sorted_uids]
            assert len(user_metric_list[m]) == dataset_len, "User-level metric length should match dataset length."
            results[m] = np.mean(user_metric_list[m])
        if self.local_rank == 0:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(user_metric_list, f, indent=4)
        self.info(f"Saved user-level metrics to {save_path}.")

    def _save_results_and_log(self, results, results_file: str, *, multiple: bool):
        """Print results to stdout and dump JSON (rank 0 only).

        ``multiple=False`` formats a flat metric→value mapping (single-eval-type case).
        ``multiple=True`` formats a list of {eval_type, ...metrics...} dicts.
        """
        logger.success("======================================================")
        logger.success("Results:")
        if multiple:
            for res in results:
                logger.success("======================================================")
                logger.success(f"{res['eval_type']} results:")
                for m in res:
                    if isinstance(res[m], float):
                        logger.success(f"\t{m} = {res[m]:.4f}")
        else:
            for m in results:
                logger.success(f"\t{m} = {results[m]:.4f}")
        logger.success("======================================================")
        if self.local_rank == 0:
            ensure_dir(os.path.dirname(results_file))
            with open(results_file, "w") as f:
                json.dump(results, f, indent=4)
        logger.success(f"Results saved to {results_file}.")
