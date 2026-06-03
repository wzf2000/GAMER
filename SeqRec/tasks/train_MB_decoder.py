from SeqRec.datasets.MB_dataset import MBExplicitDatasetForDecoder
from SeqRec.datasets.loading_MB import load_MB_datasets
from SeqRec.models.generative.registry import backbone_uses_actions
from SeqRec.tasks.generative_train_base import BaseGenerativeTrainTask


class TrainMBDecoder(BaseGenerativeTrainTask):
    """
    Train a MB decoder for the SeqRec model.
    """

    checkpoint_dir_name = "MB-decoder"
    parser_help = "Train a MB decoder for SeqRec."
    replace_progress = True

    @staticmethod
    def parser_name() -> str:
        return "train_MB_decoder"

    def load_train_data(self, data_args):
        return load_MB_datasets(
            dataset=data_args.dataset,
            data_path=data_args.data_path,
            max_his_len=data_args.max_his_len,
            index_file=data_args.index_file,
            tasks=data_args.tasks,
        )

    def get_train_notes(self, data_args, model_args) -> str:
        return f"Training MB decoder on {data_args.data_path} with base model {model_args.base_model}"

    def get_collator_kwargs(self, first_dataset, tokenizer, context):
        return {
            "decoder_response_dataset_types": (MBExplicitDatasetForDecoder,),
        }

    def get_label_names(self, backbone: str) -> list[str]:
        if backbone_uses_actions(backbone):
            return ['input_ids', 'labels', 'actions', 'split']
        return ['input_ids', 'labels', 'split']
