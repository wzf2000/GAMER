from SeqRec.tasks.base import Task
from SeqRec.utils.parse import SubParsersAction, parse_global_args, parse_dataset_args


class TestDecoder(Task):
    """
    Test a decoder for the SeqRec model.
    """

    @staticmethod
    def parser_name() -> str:
        return "test_decoder"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        """
        Add subparsers for the TrainDecoder task.
        """
        parser = sub_parsers.add_parser("test_decoder", help="Train a decoder for SeqRec.")
        parser = parse_global_args(parser)
        parser = parse_dataset_args(parser)
        parser.add_argument("--ckpt_path", type=str, default="./checkpoint", help="The checkpoint path")
        parser.add_argument(
            "--results_file",
            type=str,
            default="./results/test.json",
            help="result output path",
        )
        parser.add_argument("--test_batch_size", type=int, default=16)
        parser.add_argument("--num_beams", type=int, default=20)
        parser.add_argument(
            "--sample_num",
            type=int,
            default=-1,
            help="test sample number, -1 represents using all test data",
        )
        parser.add_argument(
            "--metrics",
            type=str,
            default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
            help="test metrics, separate by comma",
        )
        parser.add_argument("--test_task", type=str, default="SeqRec")

    def invoke(self, *args, **kwargs):
        """
        Test the decoder using the provided arguments.
        """
        # Implementation of the training logic goes here.
        pass
