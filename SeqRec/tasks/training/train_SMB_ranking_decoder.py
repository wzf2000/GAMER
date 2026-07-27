import os
from typing import Any

import torch
from transformers import TrainerCallback

from SeqRec.datasets.loaders.session_behavior import load_SMB_datasets
from SeqRec.evaluation.ranking import binary_auc
from SeqRec.models.generative.registry import backbone_uses_sessions
from SeqRec.tasks.training.base import BaseGenerativeTrainTask
from SeqRec.tasks.training.helpers import get_behavior_token_ids


def _is_epoch_strategy(strategy: Any) -> bool:
    return getattr(strategy, "value", strategy) == "epoch"


class _EveryNEpochSaveEvalCallback(TrainerCallback):
    def __init__(self, interval: int):
        self.interval = interval

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.interval <= 1 or not _is_epoch_strategy(args.eval_strategy):
            return control
        epoch = int(round(state.epoch or 0))
        if epoch > 0 and epoch % self.interval != 0:
            control.should_evaluate = False
            control.should_save = False
        return control


class _SampledCvrAucCallback(TrainerCallback):
    def __init__(
        self,
        *,
        valid_data: Any,
        tokenizer: Any,
        target_behavior_token_id: int,
        behavior_level: dict[str, int],
        max_behavior_level: int,
        sample_count: int,
        candidate_batch_size: int,
    ):
        self.valid_data = valid_data
        self.tokenizer = tokenizer
        self.target_behavior_token_id = target_behavior_token_id
        self.behavior_level = behavior_level
        self.max_behavior_level = max_behavior_level
        self.sample_count = sample_count
        self.candidate_batch_size = candidate_batch_size
        self.trainer = None

    def _align_sequence(self, sequence: list[int], length: int, pad_value: int = 0) -> list[int]:
        if len(sequence) > length:
            sequence = sequence[-length:]
        if len(sequence) < length:
            sequence = sequence + [pad_value] * (length - len(sequence))
        return sequence

    def _sample_indices(self) -> list[int]:
        if len(self.valid_data) == 0:
            return []
        total = min(self.sample_count, len(self.valid_data))
        step = max(1, len(self.valid_data) // total)
        return list(range(0, len(self.valid_data), step))[:total]

    def _score_batch(
        self,
        *,
        model,
        device: torch.device,
        samples: list[dict],
    ) -> torch.Tensor:
        texts = [sample["input_ids"] for sample in samples]
        inputs = self.tokenizer(
            text=texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        max_length = inputs["input_ids"].shape[1]
        relation_actions = [
            self._align_sequence(sample["relation_actions"], max_length)
            for sample in samples
        ]

        model_inputs = {
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device),
            "relation_actions": torch.tensor(relation_actions, dtype=torch.long, device=device),
            "actions": torch.tensor(relation_actions, dtype=torch.long, device=device),
            "session_ids": torch.tensor(
                [self._align_sequence(sample["session_ids"], max_length) for sample in samples],
                dtype=torch.long,
                device=device,
            ),
            "extended_session_ids": torch.tensor(
                [self._align_sequence(sample["extended_session_ids"], max_length) for sample in samples],
                dtype=torch.long,
                device=device,
            ),
            "use_cache": False,
        }
        output = model(**model_inputs)
        last_indices = model_inputs["attention_mask"].sum(dim=1) - 1
        last_logits = output.logits[torch.arange(output.logits.shape[0], device=device), last_indices]
        return torch.log_softmax(last_logits, dim=-1)[:, self.target_behavior_token_id].detach().cpu()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.sample_count <= 0:
            return control
        model = kwargs.get("model")
        if model is None:
            model = self.trainer.model if self.trainer is not None else None
        if model is None:
            return control

        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"
        was_training = model.training
        model.eval()
        labels = []
        scores = []
        sampled = [self.valid_data[index] for index in self._sample_indices()]
        with torch.no_grad():
            for start in range(0, len(sampled), self.candidate_batch_size):
                batch = sampled[start : start + self.candidate_batch_size]
                scores.extend(
                    self._score_batch(
                        model=model,
                        device=args.device,
                        samples=batch,
                    ).tolist()
                )
                labels.extend(
                    1 if self.behavior_level[sample["behavior"]] == self.max_behavior_level else 0
                    for sample in batch
                )
        if was_training:
            model.train()

        if scores and metrics is not None:
            metrics["eval_auc_sampled"] = binary_auc(labels, scores)
            metrics["eval_auc_sampled_n"] = len(labels)
            metrics["eval_auc_sampled_positive"] = sum(labels)
        if scores and self.trainer is not None and self.trainer.is_world_process_zero():
            self.trainer.log(
                {
                    "eval_auc_sampled": binary_auc(labels, scores),
                    "eval_auc_sampled_n": len(labels),
                    "eval_auc_sampled_positive": sum(labels),
                }
            )
        return control


class TrainSMBRankingDecoder(BaseGenerativeTrainTask):
    checkpoint_dir_name = "SMB-ranking-decoder"
    parser_help = "Train a SMB decoder for behavior-token ranking."
    parser_model_max_length = 1024
    include_find_unused_parameters = True
    include_debug = True
    replace_progress = True

    @staticmethod
    def parser_name() -> str:
        return "train_SMB_ranking_decoder"

    def invoke(self, **raw_args):
        if not raw_args["backbone"].startswith("Qwen3TemporalHierarchical"):
            raise ValueError("SMB ranking decoder requires a Qwen3TemporalHierarchical backbone.")
        return super().invoke(**raw_args)

    def load_train_data(self, data_args):
        train_data, valid_data = load_SMB_datasets(
            dataset=data_args.dataset,
            data_path=data_args.data_path,
            max_his_len=data_args.max_his_len,
            index_file=data_args.index_file,
            tasks=data_args.tasks,
            train_session=data_args.train_session,
        )
        self._ranking_valid_data = valid_data
        return train_data, valid_data

    def get_train_notes(self, data_args, model_args) -> str:
        return f"Training SMB ranking decoder on {data_args.data_path} with base model {model_args.base_model}"

    def prepare_training_context(self, first_dataset, tokenizer):
        target_behavior = first_dataset.target_behavior
        target_behavior_ids = tokenizer.encode(
            "".join(first_dataset.get_behavior_tokens(target_behavior)),
            add_special_tokens=False,
        )
        if len(target_behavior_ids) != 1:
            raise ValueError(f"Expected one behavior token for {target_behavior}, got {target_behavior_ids}.")
        self._ranking_target_behavior_token_id = target_behavior_ids[0]
        self._ranking_behavior_level = first_dataset.behavior_level
        self._ranking_max_behavior_level = first_dataset.max_behavior_level
        return {
            "behavior_tokens": get_behavior_token_ids(first_dataset, tokenizer),
        }

    def get_collator_kwargs(self, first_dataset, tokenizer, context):
        return {
            "ranking": True,
        }

    def get_model_prepare_kwargs(self, context):
        return {
            "behavior_token_ids": context["behavior_tokens"],
            "pba_uses_temperature": True,
        }

    def get_label_names(self, backbone: str) -> list[str]:
        if backbone_uses_sessions(backbone):
            return [
                "input_ids",
                "labels",
                "session_ids",
                "extended_session_ids",
                "split",
                "actions",
                "relation_actions",
            ]
        return ["input_ids", "labels", "split", "relation_actions"]

    def get_ddp_find_unused_parameters(self, script_args):
        if self.ddp:
            return script_args.find_unused_parameters
        return None

    def after_trainer_created(self, trainer: Any):
        super().after_trainer_created(trainer)
        eval_epochs = int(os.environ.get("SMB_RANKING_EVAL_EPOCHS", "1"))
        if eval_epochs > 1:
            trainer.add_callback(_EveryNEpochSaveEvalCallback(eval_epochs))
            self.info(f"Eval/save will run every {eval_epochs} epoch(s).")

        sample_count = int(os.environ.get("SMB_RANKING_TRAIN_AUC_SAMPLES", "16"))
        if sample_count <= 0:
            return
        callback = _SampledCvrAucCallback(
            valid_data=self._ranking_valid_data,
            tokenizer=trainer.processing_class,
            target_behavior_token_id=self._ranking_target_behavior_token_id,
            behavior_level=self._ranking_behavior_level,
            max_behavior_level=self._ranking_max_behavior_level,
            sample_count=sample_count,
            candidate_batch_size=max(1, int(os.environ.get("SMB_RANKING_TRAIN_AUC_BATCH_SIZE", str(trainer.args.per_device_eval_batch_size)))),
        )
        callback.trainer = trainer
        trainer.add_callback(callback)
        self.info(
            "Enabled sampled train-time CVR AUC: "
            f"{callback.sample_count} valid samples, "
            f"batch size {callback.candidate_batch_size}."
        )
