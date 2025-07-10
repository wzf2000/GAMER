from SeqRec.tasks.base import SubParsersAction, Task


class Tokenize(Task):
    """
    Tokenize item semantic information for the dataset.
    """

    @staticmethod
    def parser_name() -> str:
        return "tokenize"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        """
        Add subparsers for the Tokenize task.
        """
        parser = sub_parsers.add_parser(
            "tokenize", help="Run item tokenization for the dataset"
        )
        parser.add_argument("--dataset", type=str, default="Instruments", help="dataset")
        parser.add_argument(
            "--root_path", type=str, default="./checkpoint/RQ-VAE", help="root path"
        )
        parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")
        parser.add_argument("--alpha", type=str, default="0.2", help="cf loss weight")
        parser.add_argument("--beta", type=str, default="0.0001", help="div loss weight")
        parser.add_argument("--epoch", type=int, default="20000", help="epoch")
        parser.add_argument(
            "--checkpoint",
            type=str,
            default="best_collision_model.pth",
            help="checkpoint name",
        )

    def invoke(self, *args, **kwargs):
        """
        Run the tokenization process for the dataset.
        """
        # Implementation of the tokenization logic goes here.
        pass
