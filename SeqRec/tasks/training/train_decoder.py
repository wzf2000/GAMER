from SeqRec.datasets.loaders.sequential import load_datasets
from SeqRec.tasks.training.base import BaseGenerativeTrainTask


class TrainDecoder(BaseGenerativeTrainTask):
    """
    Train a decoder for the SeqRec model.
    """

    checkpoint_dir_name = "decoder"
    parser_help = "Train a decoder for SeqRec."

    @staticmethod
    def parser_name() -> str:
        return "train_decoder"

    def load_train_data(self, data_args):
        return load_datasets(
            dataset=data_args.dataset,
            data_path=data_args.data_path,
            max_his_len=data_args.max_his_len,
            index_file=data_args.index_file,
            tasks=data_args.tasks,
        )
