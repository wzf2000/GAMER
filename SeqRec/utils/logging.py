import os
import sys
import time
import logging
import inspect
from tqdm import tqdm
from loguru import logger
from typing import TYPE_CHECKING
from transformers.trainer_callback import TrainerCallback, ProgressCallback


if TYPE_CHECKING:
    from transformers.trainer import Trainer
    from transformers.training_args import TrainingArguments
    from transformers.trainer_callback import TrainerState, TrainerControl


class InterceptHandler(logging.Handler):
    def __init__(self, level: str | int = logging.NOTSET, filter_level: str = 'INFO') -> None:
        super().__init__(level)
        self.setLevel(logging.DEBUG)
        self.filter_level = logger.level(filter_level).no

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        if record.levelno < self.filter_level:
            return

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame:
            filename = frame.f_code.co_filename
            is_logging = filename == logging.__file__
            is_frozen = "importlib" in filename and "_bootstrap" in filename
            if depth > 0 and not (is_logging or is_frozen):
                break
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class LoguruCallback(TrainerCallback):
    """
    A custom callback for Hugging Face Trainer to log training progress using Loguru.
    """

    def on_log(self, args: 'TrainingArguments', state: 'TrainerState', control: 'TrainerControl', logs: dict | None = None, **kwargs):
        if state.is_world_process_zero and logs is not None:
            # make a shallow copy of logs so we can mutate the fields copied
            # but avoid doing any value pickling.
            shallow_logs = {}
            for k, v in logs.items():
                if isinstance(v, str) and len(v) > self.max_str_len:
                    shallow_logs[k] = (
                        f"[String too long to display, length: {len(v)} > {self.max_str_len}. "
                        "Consider increasing `max_str_len` if needed.]"
                    )
                else:
                    shallow_logs[k] = v
            _ = shallow_logs.pop("total_flos", None)
            # round numbers so that it looks better in console
            if "epoch" in shallow_logs:
                shallow_logs["epoch"] = round(shallow_logs["epoch"], 2)
            logger.info(shallow_logs)


class ProgressCallbackWithLoguru(ProgressCallback):
    def on_log(self, args: 'TrainingArguments', state: 'TrainerState', control: 'TrainerControl', logs: dict | None = None, **kwargs):
        if state.is_world_process_zero and logs is not None and self.training_bar is not None:
            # make a shallow copy of logs so we can mutate the fields copied
            # but avoid doing any value pickling.
            shallow_logs = {}
            for k, v in logs.items():
                if isinstance(v, str) and len(v) > self.max_str_len:
                    shallow_logs[k] = (
                        f"[String too long to display, length: {len(v)} > {self.max_str_len}. "
                        "Consider increasing `max_str_len` if needed.]"
                    )
                else:
                    shallow_logs[k] = v
            _ = shallow_logs.pop("total_flos", None)
            # round numbers so that it looks better in console
            if "epoch" in shallow_logs:
                shallow_logs["epoch"] = round(shallow_logs["epoch"], 2)
            with tqdm.external_write_mode(sys.stdout, nolock=False):
                logger.info(shallow_logs)


def replace_progress_callback(trainer: 'Trainer'):
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(ProgressCallbackWithLoguru())


def intercept_logging():
    """
    Intercept standard logging calls and redirect them to Loguru.
    """
    def _replace_handler(sub_logger: logging.Logger, filter_level: str = 'INFO'):
        for handler in sub_logger.handlers[:]:
            sub_logger.removeHandler(handler)
        sub_logger.addHandler(InterceptHandler(filter_level=filter_level))
        sub_logger.setLevel(logging.DEBUG)

    _replace_handler(logging.getLogger())
    _replace_handler(logging.getLogger("DeepSpeed"))
    _replace_handler(logging.getLogger("transformers"), filter_level='WARNING')


def init_logger(log_dir: str, level: str = "INFO"):
    log_file = os.path.join(log_dir, f"{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger.remove()
    logger.add(log_file, rotation="1 week", level="DEBUG", backtrace=True, diagnose=True, filter=lambda _: os.environ.get("LOCAL_RANK", "0") == "0")
    logger.add(sys.stdout, level=level, filter=lambda _: os.environ.get("LOCAL_RANK", "0") == "0")


def set_color(log: str, color: str) -> str:
    return f"<{color}>{log}</{color}>"


intercept_logging()
