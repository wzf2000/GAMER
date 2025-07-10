from SeqRec.utils.parse import SubParsersAction


class Task:
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
