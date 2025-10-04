from SeqRec.tasks.base import Task
from SeqRec.tasks.semantic_emb import SemanticEmbedding
from SeqRec.tasks.RQVAE import TrainRQVAE
from SeqRec.tasks.tokenize import Tokenize
from SeqRec.tasks.train_decoder import TrainDecoder
from SeqRec.tasks.train_MB_decoder import TrainMBDecoder
from SeqRec.tasks.train_SMB_decoder import TrainSMBDecoder
from SeqRec.tasks.train_SMB_rec import TrainSMBRec
from SeqRec.tasks.test_decoder import TestDecoder
from SeqRec.tasks.test_MB_decoder import TestMBDecoder
from SeqRec.tasks.test_SMB_decoder import TestSMBDecoder
from SeqRec.tasks.test_SMB_rule import TestSMBRule
from SeqRec.utils.func_util import subclasses_recursive


task_list: dict[str, type[Task]] = {
    task.parser_name(): task for task in subclasses_recursive(Task) if not task.__name__.endswith("Task")
}
