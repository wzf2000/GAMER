import argparse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    SubParsersAction = argparse._SubParsersAction[argparse.ArgumentParser]
else:
    SubParsersAction = Any


def parse_global_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--base_model",
        type=str,
        default="./ckpt/s2s-models/TIGER",
        help="basic model path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoint/Recommender",
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
