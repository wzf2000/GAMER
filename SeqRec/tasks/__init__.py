from SeqRec.tasks.base import Task
from SeqRec.tasks.registry import TASK_SPECS, TaskRegistry, get_task_class, task_list

__all__ = [
    "Task",
    "TASK_SPECS",
    "TaskRegistry",
    "get_task_class",
    "task_list",
]
