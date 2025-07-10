from SeqRec.tasks.base import Task
from SeqRec.tasks.RQVAE import TrainRQVAE  # noqa: F401
from SeqRec.tasks.tokenize import Tokenize  # noqa: F401
from SeqRec.tasks.train_decoder import TrainDecoder  # noqa: F401
from SeqRec.tasks.test_decoder import TestDecoder  # noqa: F401

task_list: dict[str, type[Task]] = {
    task.parser_name(): task for task in Task.__subclasses__() if task.__name__ != "Task"
}
