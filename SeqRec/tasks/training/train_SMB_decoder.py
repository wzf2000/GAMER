from SeqRec.datasets.session_behavior import SMBExplicitDatasetForDecoder, SMBFixedRatioDatasetForDecoder
from SeqRec.datasets.loaders.session_behavior import load_SMB_datasets
from SeqRec.models.generative.registry import backbone_uses_sessions
from SeqRec.tasks.training.base import BaseGenerativeTrainTask
from SeqRec.tasks.training.helpers import get_behavior_token_ids


class TrainSMBDecoder(BaseGenerativeTrainTask):
    """
    Train a SMB decoder for the SeqRec model.
    """

    checkpoint_dir_name = "SMB-decoder"
    parser_help = "Train a decoder for session-wise multi-behavior recommendation."
    parser_model_max_length = 1024
    include_find_unused_parameters = True
    include_debug = True
    replace_progress = True

    @staticmethod
    def parser_name() -> str:
        return "train_SMB_decoder"

    def load_train_data(self, data_args):
        augmentation_config = {
            "sequence_augmentation": data_args.sequence_augmentation,
            "augmentation_views": data_args.augmentation_views,
            "augmentation_seed": data_args.augmentation_seed,
            "augmentation_drop_original": data_args.augmentation_drop_original,
            "augmentation_config": {
                "augmentation_min_history_items": data_args.augmentation_min_history_items,
                "time_decay_type": data_args.time_decay_type,
                "time_decay_tau": data_args.time_decay_tau,
                "time_decay_severity": data_args.time_decay_severity,
                "time_decay_max_drop": data_args.time_decay_max_drop,
                "time_decay_min_recent_items": data_args.time_decay_min_recent_items,
                "time_decay_allow_target_level_drop": data_args.time_decay_allow_target_level_drop,
                "recent_session_count": data_args.recent_session_count,
                "session_keep_probability": data_args.session_keep_probability,
                "session_time_decay_tau": data_args.session_time_decay_tau,
                "session_high_level_bonus": data_args.session_high_level_bonus,
                "session_allow_target_level_drop": data_args.session_allow_target_level_drop,
                "dataset_proportion_preset": data_args.dataset_proportion_preset,
                "dataset_proportion_tolerance": data_args.dataset_proportion_tolerance,
                "dataset_proportion_allow_target_level_drop": data_args.dataset_proportion_allow_target_level_drop,
                "user_adaptive_smoothing": data_args.user_adaptive_smoothing,
                "user_adaptive_confidence_scale": data_args.user_adaptive_confidence_scale,
                "user_adaptive_min_ratio": data_args.user_adaptive_min_ratio,
                "user_adaptive_max_ratio": data_args.user_adaptive_max_ratio,
                "user_adaptive_tolerance": data_args.user_adaptive_tolerance,
                "user_adaptive_allow_target_level_drop": data_args.user_adaptive_allow_target_level_drop,
                "target_conditioned_base_policy": data_args.target_conditioned_base_policy,
                "target_conditioned_same_level_restore": data_args.target_conditioned_same_level_restore,
                "target_conditioned_precursor_restore": data_args.target_conditioned_precursor_restore,
                "multi_view_disable_recent": data_args.multi_view_disable_recent,
                "multi_view_disable_hierarchy": data_args.multi_view_disable_hierarchy,
                "multi_view_disable_session": data_args.multi_view_disable_session,
            },
        }
        return load_SMB_datasets(
            dataset=data_args.dataset,
            data_path=data_args.data_path,
            max_his_len=data_args.max_his_len,
            index_file=data_args.index_file,
            tasks=data_args.tasks,
            train_session=data_args.train_session,
            sequence_augmentation_config=augmentation_config,
        )

    def get_train_notes(self, data_args, model_args) -> str:
        return f"Training SMB decoder on {data_args.data_path} with base model {model_args.base_model}"

    def prepare_training_context(self, first_dataset, tokenizer):
        return {
            "behavior_tokens": get_behavior_token_ids(first_dataset, tokenizer),
        }

    def get_collator_kwargs(self, first_dataset, tokenizer, context):
        return {
            "decoder_response_dataset_types": (SMBExplicitDatasetForDecoder, SMBFixedRatioDatasetForDecoder),
            "ignore_behavior_tokens": context["behavior_tokens"],
        }

    def get_model_prepare_kwargs(self, context):
        return {
            "behavior_token_ids": context["behavior_tokens"],
            "pba_uses_temperature": True,
        }

    def get_label_names(self, backbone: str) -> list[str]:
        if backbone_uses_sessions(backbone):
            return ['input_ids', 'labels', 'session_ids', 'extended_session_ids', 'split', 'actions']
        return ['input_ids', 'labels', 'split']

    def get_ddp_find_unused_parameters(self, script_args):
        if self.ddp:
            return script_args.find_unused_parameters
        return None
