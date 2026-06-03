import argparse
from dataclasses import MISSING, asdict, dataclass, fields
from types import UnionType
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

if TYPE_CHECKING:
    SubParsersAction = argparse._SubParsersAction[argparse.ArgumentParser]
else:
    SubParsersAction = Any


@dataclass
class ModelArgs:
    seed: int = 42
    backbone: str = "TIGER"
    base_model: str = "./config/s2s-models/TIGER"
    output_dir: str = "./checkpoint/decoder"


@dataclass
class DatasetArgs:
    data_path: str = "./data"
    tasks: str = "seqrec"
    dataset: str = "Instruments"
    index_file: str = ".index.json"
    max_his_len: int = 20


@dataclass
class ScriptTrainingArgs:
    optim: str = "adamw_torch"
    epochs: int = 200
    learning_rate: float = 5e-4
    per_device_batch_size: int = 256
    gradient_accumulation_steps: int = 2
    logging_step: int = 30
    model_max_length: int = 512
    weight_decay: float = 0.01
    resume_from_checkpoint: str | None = None
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    save_and_eval_strategy: str = "epoch"
    save_and_eval_steps: int = 1000
    patience: int = 20
    fp16: bool = False
    bf16: bool = False
    deepspeed: str | None = None
    temperature: float = 1.0
    find_unused_parameters: bool = False
    wandb_run_name: str = "default"
    debug: bool = False


@dataclass
class GenerativeTrainingArgs:
    model: ModelArgs
    dataset: DatasetArgs
    training: ScriptTrainingArgs
    unused: dict[str, Any]

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "model": asdict(self.model),
            "dataset": asdict(self.dataset),
            "training": asdict(self.training),
            "unused": self.unused,
        }


ARGUMENT_HELP = {
    "seed": "Random seed",
    "backbone": "The backbone model to use, e.g., TIGER, PBATransformer, etc.",
    "base_model": "Basic model path",
    "output_dir": "The output directory",
    "data_path": "data directory",
    "tasks": "Downstream tasks, separate by comma",
    "dataset": "Dataset name",
    "index_file": "the item indices file",
    "max_his_len": "the max number of items in history sequence, -1 means no limit",
    "optim": "The name of the optimizer",
    "epochs": "Number of training epochs",
    "learning_rate": "Learning rate for the optimizer",
    "per_device_batch_size": "Batch size per device during training",
    "gradient_accumulation_steps": "Number of steps to accumulate gradients before updating the model",
    "logging_step": "Logging frequency in steps",
    "model_max_length": "Maximum sequence length for the model",
    "weight_decay": "Weight decay for regularization",
    "resume_from_checkpoint": "either training checkpoint or final adapter",
    "warmup_ratio": "Warmup ratio for learning rate scheduler",
    "lr_scheduler_type": "Type of learning rate scheduler to use",
    "save_and_eval_strategy": "Strategy for saving and evaluating the model (e.g., 'epoch', 'steps')",
    "save_and_eval_steps": "Steps at which to save and evaluate the model",
    "patience": "Number of evaluation steps to wait before stopping training if no improvement",
    "fp16": "Use mixed precision training (fp16)",
    "bf16": "Use bfloat16 precision training",
    "deepspeed": "Path to deepspeed configuration file",
    "temperature": "Temperature for softmax scaling",
    "find_unused_parameters": "Find unused parameters",
    "wandb_run_name": "Name for the Weights & Biases run",
    "debug": "Enable debug mode without logging to WandB",
}


def _strip_optional(arg_type: Any) -> Any:
    origin = get_origin(arg_type)
    if origin not in (Union, UnionType):
        return arg_type
    non_none_types = [item for item in get_args(arg_type) if item is not type(None)]
    if len(non_none_types) != 1:
        return str
    return non_none_types[0]


def _field_default(field: Any, default_overrides: dict[str, Any]) -> Any:
    if field.name in default_overrides:
        return default_overrides[field.name]
    if field.default is not MISSING:
        return field.default
    return None


def add_dataclass_arguments(
    parser: argparse.ArgumentParser,
    dataclass_type: type,
    *,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
    default_overrides: dict[str, Any] | None = None,
) -> argparse.ArgumentParser:
    default_overrides = default_overrides or {}
    exclude = exclude or set()
    for field in fields(dataclass_type):
        if include is not None and field.name not in include:
            continue
        if field.name in exclude:
            continue
        default = _field_default(field, default_overrides)
        arg_name = f"--{field.name}"
        arg_type = _strip_optional(field.type)
        if arg_type is bool:
            action = "store_false" if default else "store_true"
            parser.add_argument(arg_name, action=action, default=default, help=ARGUMENT_HELP.get(field.name))
        else:
            parser.add_argument(arg_name, type=arg_type, default=default, help=ARGUMENT_HELP.get(field.name))
    return parser


def _pack_dataclass(dataclass_type: type, values: dict[str, Any]) -> Any:
    return dataclass_type(**{
        field.name: values[field.name]
        for field in fields(dataclass_type)
        if field.name in values
    })


def build_generative_training_args(values: dict[str, Any]) -> GenerativeTrainingArgs:
    known_fields = {
        field.name
        for dataclass_type in (ModelArgs, DatasetArgs, ScriptTrainingArgs)
        for field in fields(dataclass_type)
    }
    return GenerativeTrainingArgs(
        model=_pack_dataclass(ModelArgs, values),
        dataset=_pack_dataclass(DatasetArgs, values),
        training=_pack_dataclass(ScriptTrainingArgs, values),
        unused={
            key: value
            for key, value in values.items()
            if key not in known_fields
        },
    )


def parse_global_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return add_dataclass_arguments(parser, ModelArgs)


def parse_dataset_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return add_dataclass_arguments(parser, DatasetArgs)


def parse_training_args(
    parser: argparse.ArgumentParser,
    *,
    model_max_length: int = 512,
    include_find_unused_parameters: bool = False,
    include_debug: bool = False,
) -> argparse.ArgumentParser:
    excluded_fields = {"find_unused_parameters", "debug"}
    if include_find_unused_parameters:
        excluded_fields.remove("find_unused_parameters")
    if include_debug:
        excluded_fields.remove("debug")
    return add_dataclass_arguments(
        parser,
        ScriptTrainingArgs,
        exclude=excluded_fields,
        default_overrides={"model_max_length": model_max_length},
    )


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
