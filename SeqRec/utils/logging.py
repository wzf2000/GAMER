import logging


def disable_deepspeed_logging():
    """
    Disable DeepSpeed logging to avoid cluttering the output.
    """
    ds_logger = logging.getLogger("DeepSpeed")
    ds_logger.setLevel(logging.WARNING)


def set_color(log: str, color: str, highlight: bool = True) -> str:
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except Exception:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"


disable_deepspeed_logging()
