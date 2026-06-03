from importlib import import_module
from typing import Iterator


# parser_name -> "module_path:ClassName"
TASK_SPECS: dict[str, str] = {
    "SemEmb": "SeqRec.tasks.tokenization.semantic_emb:SemanticEmbedding",
    "RQVAE": "SeqRec.tasks.tokenization.rqvae:TrainRQVAE",
    "tokenize": "SeqRec.tasks.tokenization.tokenize:Tokenize",
    "train_decoder": "SeqRec.tasks.training.train_decoder:TrainDecoder",
    "train_MB_decoder": "SeqRec.tasks.training.train_MB_decoder:TrainMBDecoder",
    "train_SMB_decoder": "SeqRec.tasks.training.train_SMB_decoder:TrainSMBDecoder",
    "train_SMB_rec": "SeqRec.tasks.training.train_SMB_rec:TrainSMBRec",
    "test_decoder": "SeqRec.tasks.evaluation.test_decoder:TestDecoder",
    "test_MB_decoder": "SeqRec.tasks.evaluation.test_MB_decoder:TestMBDecoder",
    "test_SMB_decoder": "SeqRec.tasks.evaluation.test_SMB_decoder:TestSMBDecoder",
    "test_MB_rule": "SeqRec.tasks.evaluation.rule:TestMBRule",
    "test_SMB_rule": "SeqRec.tasks.evaluation.rule:TestSMBRule",
    "analyze_behavior_dropout": "SeqRec.tasks.analysis.behavior_dropout:AnalyzeBehaviorDropout",
    "analyze_sparse_behavior": "SeqRec.tasks.analysis.sparse_behavior:AnalyzeSparseTargetBehavior",
}


def get_task_class(name: str) -> type:
    spec = TASK_SPECS[name]
    module_path, class_name = spec.split(":", maxsplit=1)
    return getattr(import_module(module_path), class_name)


class TaskRegistry:
    """Mapping from parser name to Task subclass. Each task module is imported on first access."""

    def __init__(self, specs: dict[str, str]):
        self._specs = specs
        self._cache: dict[str, type] = {}

    def __getitem__(self, name: str) -> type:
        if name not in self._cache:
            self._cache[name] = get_task_class(name)
        return self._cache[name]

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def keys(self):
        return self._specs.keys()

    def values(self) -> Iterator[type]:
        return (self[name] for name in self._specs)

    def items(self) -> Iterator[tuple[str, type]]:
        return ((name, self[name]) for name in self._specs)


task_list = TaskRegistry(TASK_SPECS)
