import os
import torch
from typing import Any
import torch.distributed as dist

from SeqRec.tasks.base import Task
from SeqRec.utils.pipe import set_seed


class MultiGPUTask(Task):
    @property
    def local_rank(self) -> int:
        if not hasattr(self, "_local_rank"):
            self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return self._local_rank

    @property
    def world_size(self) -> int:
        if not hasattr(self, "_world_size"):
            self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        return self._world_size

    @property
    def ddp(self) -> bool:
        if not hasattr(self, "_ddp"):
            self._ddp = self.world_size != 1
        return self._ddp

    @property
    def device(self) -> str:
        if self.ddp:
            return f"cuda:{self.local_rank}"
        else:
            return "cuda"

    def info(self, msg: str | list[str]):
        if self.ddp:
            dist.barrier()
        super().info(msg)

    def init(
        self,
        seed: int,
        wandb_init: bool = True,
        wandb_run_name: str | None = None,
        job_type: str | None = None,
        notes: str | None = None,
        args: dict[str, Any] | None = None,
    ):
        set_seed(seed)
        if self.ddp:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method="env://", rank=self.local_rank, world_size=self.world_size, device_id=torch.device(self.device))
        if self.local_rank == 0 and wandb_init:
            import wandb
            wandb.init(
                project=self.parser_name(),
                config=args,
                name=wandb_run_name,
                dir=f"runs/{self.parser_name()}",
                job_type=job_type,
                reinit="return_previous",
                notes=notes,
            )

    def finish(self, wandb_finish: bool = True):
        if self.ddp:
            dist.destroy_process_group()
        if self.local_rank == 0 and wandb_finish:
            import wandb
            wandb.finish()
