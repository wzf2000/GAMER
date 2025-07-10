from SeqRec.tasks.base import Task
from SeqRec.utils.parse import SubParsersAction, parse_global_args, parse_dataset_args


class TrainDecoder(Task):
    """
    Train a decoder for the SeqRec model.
    """

    @staticmethod
    def parser_name() -> str:
        return "train_decoder"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        """
        Add subparsers for the TrainDecoder task.
        """
        parser = sub_parsers.add_parser("train_decoder", help="Train a decoder for SeqRec.")
        parser = parse_global_args(parser)
        parser = parse_dataset_args(parser)
        parser.add_argument(
            "--optim", type=str, default="adamw_torch", help="The name of the optimizer"
        )
        parser.add_argument(
            "--epochs", type=int, default=200, help="Number of training epochs"
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=5e-4,
            help="Learning rate for the optimizer",
        )
        parser.add_argument(
            "--per_device_batch_size",
            type=int,
            default=256,
            help="Batch size per device during training",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=2,
            help="Number of steps to accumulate gradients before updating the model",
        )
        parser.add_argument(
            "--logging_step", type=int, default=30, help="Logging frequency in steps"
        )
        parser.add_argument(
            "--model_max_length",
            type=int,
            default=2048,
            help="Maximum sequence length for the model",
        )
        parser.add_argument(
            "--weight_decay", type=float, default=0.01, help="Weight decay for regularization"
        )
        parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help="either training checkpoint or final adapter",
        )
        parser.add_argument(
            "--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler"
        )
        parser.add_argument(
            "--lr_scheduler_type",
            type=str,
            default="cosine",
            help="Type of learning rate scheduler to use",
        )
        parser.add_argument(
            "--save_and_eval_strategy",
            type=str,
            default="epoch",
            help="Strategy for saving and evaluating the model (e.g., 'epoch', 'steps')",
        )
        parser.add_argument(
            "--save_and_eval_steps",
            type=int,
            default=1000,
            help="Steps at which to save and evaluate the model",
        )
        parser.add_argument(
            "--fp16", action="store_true", default=False, help="Use mixed precision training (fp16)"
        )
        parser.add_argument(
            "--bf16", action="store_true", default=False, help="Use bfloat16 precision training"
        )
        parser.add_argument(
            "--deepspeed", type=str, default=None, help="Path to deepspeed configuration file"
        )
        parser.add_argument(
            "--temperature", type=float, default=1.0, help="Temperature for softmax scaling"
        )

        parser.add_argument(
            "--wandb_run_name",
            type=str,
            default="default",
            help="Name for the Weights & Biases run"
        )

    def invoke(self, *args, **kwargs):
        """
        Train the decoder using the provided arguments.
        """
        # Implementation of the training logic goes here.
        pass
