from typing import Any

from SeqRec.utils.parse import SubParsersAction
from SeqRec.utils.func_util import log_arguments, create_meta_class


class Task(metaclass=create_meta_class("Task", ("invoke", ), log_arguments)):
    param_dict: dict[str, Any]

    def __init__(self):
        pass

    @staticmethod
    def parser_name() -> str:
        """
        Return the name of the task subparser.
        This method should be implemented by subclasses to return a unique name.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction) -> None:
        """
        Add subparsers to the provided subparsers object.
        This method should be implemented by subclasses to define specific subparsers.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def invoke(self, *args, **kwargs):
        """
        Invoke the task with the provided arguments.
        This method should be implemented by subclasses to define specific task behavior.
        """
        raise NotImplementedError("Subclasses should implement this method.")
