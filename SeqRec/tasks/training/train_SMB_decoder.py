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
        return load_SMB_datasets(
            dataset=data_args.dataset,
            data_path=data_args.data_path,
            max_his_len=data_args.max_his_len,
            index_file=data_args.index_file,
            tasks=data_args.tasks,
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
