import os
import sys
import argparse
from loguru import logger

from SeqRec.utils.futils import ensure_dir
from SeqRec.utils.logging import init_logger
from SeqRec.tasks import task_list


def parse_args() -> tuple[argparse.Namespace, list[str]]:
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

    return parser.parse_known_args()


@logger.catch(onerror=lambda _: sys.exit(1))
def main():
    args, unknown_args = parse_args()
    task_name: str = args.pipeline
    # remove the pipeline attribute from args
    del args.pipeline
    logger.success(f"Parsed arguments for {task_name}: {vars(args)}")
    if task_name in task_list:
        log_dir = os.path.join("logs", task_name)
        ensure_dir(log_dir)
        init_logger(log_dir)
        logger.success("Initialized logger!")
        if len(unknown_args) > 0:
            logger.warning(f"Unknown args: {unknown_args}")
        else:
            logger.success("No unknown args found.")
        task = task_list[task_name]()
        task.invoke(**vars(args))
    else:
        raise ValueError(f"Unknown task: {task_name}")


if __name__ == "__main__":
    main()
