import os
import time
import argparse
from loguru import logger

from SeqRec.tasks import task_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(
        dest="pipeline",
        title="Available pipelines",
        description="Choose a pipeline to run",
        help="Which pipeline to run",
        required=True
    )
    for task_class in task_list.values():
        task_class.add_sub_parsers(sub_parsers)

    return parser.parse_args()


def main():
    args = parse_args()
    task_name: str = args.pipeline
    # remove the pipeline attribute from args
    del args.pipeline
    if task_name in task_list:
        log_dir = os.path.join("logs", task_name)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{time.strftime('%Y%m%d_%H%M%S')}.log")
        logger.add(log_file, rotation="1 week", level="INFO", backtrace=True, diagnose=True)
        task = task_list[task_name]()
        task.invoke(**vars(args))
    else:
        raise ValueError(f"Unknown task: {task_name}")


if __name__ == "__main__":
    main()
