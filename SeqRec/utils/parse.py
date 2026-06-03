import argparse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    SubParsersAction = argparse._SubParsersAction[argparse.ArgumentParser]
else:
    SubParsersAction = Any


def parse_global_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--backbone",
        type=str,
        default="TIGER",
        help="The backbone model to use, e.g., TIGER, PBATransformer, etc.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="./config/s2s-models/TIGER",  # Default to use the TIGER (T5-based) model
        help="Basic model path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoint/decoder",
        help="The output directory",
    )

    return parser


def parse_dataset_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--data_path", type=str, default="./data", help="data directory"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="seqrec",
        help="Downstream tasks, separate by comma",
    )
    parser.add_argument(
        "--dataset", type=str, default="Instruments", help="Dataset name"
    )
    parser.add_argument(
        "--index_file", type=str, default=".index.json", help="the item indices file"
    )

    # arguments related to sequential task
    parser.add_argument(
        "--max_his_len",
        type=int,
        default=20,
        help="the max number of items in history sequence, -1 means no limit",
    )
    return parser


def parse_training_args(
    parser: argparse.ArgumentParser,
    *,
    model_max_length: int = 512,
    include_find_unused_parameters: bool = False,
    include_debug: bool = False,
) -> argparse.ArgumentParser:
    parser.add_argument("--optim", type=str, default="adamw_torch", help="The name of the optimizer")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
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
    parser.add_argument("--logging_step", type=int, default=30, help="Logging frequency in steps")
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=model_max_length,
        help="Maximum sequence length for the model",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="either training checkpoint or final adapter",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate scheduler",
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
        "--patience",
        type=int,
        default=20,
        help="Number of evaluation steps to wait before stopping training if no improvement",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use mixed precision training (fp16)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Use bfloat16 precision training",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to deepspeed configuration file",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for softmax scaling",
    )
    if include_find_unused_parameters:
        parser.add_argument(
            "--find_unused_parameters",
            action="store_true",
            default=False,
            help="Find unused parameters",
        )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="default",
        help="Name for the Weights & Biases run",
    )
    if include_debug:
        parser.add_argument(
            "--debug",
            action="store_true",
            default=False,
            help="Enable debug mode without logging to WandB",
        )
    return parser


def parse_generation_eval_args(
    parser: argparse.ArgumentParser,
    *,
    metrics: str,
    include_filter: bool = False,
    include_eval_types: bool = False,
    include_behaviors: bool = False,
    include_valid_loss: bool = False,
) -> argparse.ArgumentParser:
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
        "--metrics",
        type=str,
        default=metrics,
        help="test metrics, separate by comma",
    )
    parser.add_argument("--test_task", type=str, default="SeqRec")
    if include_filter:
        parser.add_argument(
            "--filter",
            action="store_true",
            help="Filter out the collision items from the test data",
        )
    if include_eval_types:
        parser.add_argument(
            "--eval_types",
            type=str,
            default="target_behavior,behavior_specific,behavior_item",
            help="Evaluation type, separate by comma, valid values: target_behavior, behavior_specific, behavior_item",
        )
    if include_behaviors:
        parser.add_argument("--behaviors", type=str, nargs="+", default=None, help="The behavior list.")
    if include_valid_loss:
        parser.add_argument("--valid_loss", action="store_true", help="Whether to calculate valid loss instead of testing.")
    return parser


def parse_analysis_args(
    parser: argparse.ArgumentParser,
    *,
    ckpt_required: bool = False,
    ckpt_help: str | None = None,
    results_file: str,
    test_task: str = "smb_explicit",
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None if ckpt_required else "./checkpoint",
        required=ckpt_required,
        help=ckpt_help,
    )
    parser.add_argument("--results_file", type=str, default=results_file)
    parser.add_argument("--test_task", type=str, default=test_task)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=20)
    return parser
