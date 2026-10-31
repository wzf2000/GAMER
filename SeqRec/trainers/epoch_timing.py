"""Opt-in complete-epoch timing while preserving the original Trainer schedule."""

from datetime import datetime, timezone
from pathlib import Path
import time

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from SeqRec.utils.rebuttal import write_json


def synchronize_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class EpochTimingCallback(TrainerCallback):
    def __init__(self, epochs, output_dir):
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError('The benchmark epoch limit must be a positive integer.')
        self.epochs = epochs
        self.output_dir = Path(output_dir)
        self.records = []
        self.current = None

    def on_train_begin(self, args, state, control, **kwargs):
        if args.eval_strategy != 'epoch' or args.save_strategy != 'epoch':
            raise ValueError('Complete-epoch benchmarking requires epoch evaluation and saving.')
        if args.eval_delay > 1:
            raise ValueError('Complete-epoch benchmarking requires evaluation from the first epoch.')
        synchronize_device()
        self.training_start = time.perf_counter()

    def on_epoch_begin(self, args, state, control, **kwargs):
        synchronize_device()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.epoch_start = time.perf_counter()
        self.current = {'epoch_start': float(state.epoch or 0), 'first_step': state.global_step}

    def on_epoch_end(self, args, state, control, **kwargs):
        synchronize_device()
        self.training_end = time.perf_counter()
        self.current.update(epoch=float(state.epoch), global_step=state.global_step,
                            train_seconds=self.training_end - self.epoch_start)
        if float(state.epoch) >= self.epochs:
            # Trainer still performs the scheduled epoch evaluation/save before exiting.
            control.should_training_stop = True
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        synchronize_device()
        self.evaluation_end = time.perf_counter()
        self.current.update(validation_seconds=self.evaluation_end - self.training_end,
                            trainer_eval_runtime_seconds=(metrics or {}).get('eval_runtime'),
                            eval_loss=(metrics or {}).get('eval_loss'))

    def on_save(self, args, state, control, **kwargs):
        if dist.is_initialized():
            dist.barrier()
        if getattr(args, 'load_best_model_at_end', False) and state.best_global_step is not None:
            # HF 4.51 can check this directory before rank0 creates it. An immediate
            # first-epoch stop must give every rank the same load-best barrier path.
            best = self.output_dir / f'checkpoint-{state.best_global_step}'
            if not best.is_dir():
                raise RuntimeError(f'The benchmark best checkpoint was not saved: {best}')
            state.best_model_checkpoint = str(best)
        synchronize_device()
        self.checkpoint_end = time.perf_counter()
        self.current.update(checkpoint_seconds=self.checkpoint_end - self.evaluation_end,
                            epoch_seconds=self.checkpoint_end - self.epoch_start,
                            max_memory_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
        self.records.append(self.current)
        self.current = None

    def on_train_end(self, args, state, control, **kwargs):
        synchronize_device()
        end = time.perf_counter()
        if not self.records or self.current is not None:
            raise RuntimeError('The benchmark did not finish a complete train/validation/save epoch.')
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        local = {'rank': rank, 'epochs': self.records,
                 'training_loop_seconds': end - self.training_start,
                 'loop_finalization_seconds': end - self.checkpoint_end,
                 'final_epoch': state.epoch, 'global_step': state.global_step,
                 'planned_num_train_epochs': args.num_train_epochs, 'planned_max_steps': state.max_steps,
                 'planned_warmup_steps': args.get_warmup_steps(state.max_steps)}
        write_json(self.output_dir / f'epoch-timing.rank-{rank}.json', local)
        gathered = [None] * world
        if dist.is_initialized():
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]
        if rank == 0:
            if len({len(record['epochs']) for record in gathered}) != 1:
                raise RuntimeError('Ranks recorded different epoch counts.')
            fields = ('train_seconds', 'validation_seconds', 'checkpoint_seconds', 'epoch_seconds')
            epochs = [{'epoch': gathered[0]['epochs'][index]['epoch'],
                       **{field: max(record['epochs'][index][field] for record in gathered) for field in fields}}
                      for index in range(len(self.records))]
            write_json(self.output_dir / 'epoch-timing.json', {
                'schema_version': 1, 'recorded_at': datetime.now(timezone.utc).isoformat(), 'world_size': world,
                'benchmark_epochs': self.epochs, 'planned_num_train_epochs': args.num_train_epochs,
                'planned_max_steps': state.max_steps, 'planned_warmup_steps': args.get_warmup_steps(state.max_steps),
                'epochs_max_across_ranks': epochs, 'per_rank': gathered,
                'scope': 'Complete training epoch (forward/backward/optimizer/DDP/data loading), scheduled validation loss and checkpoint save. CUDA synchronized only at phase boundaries. No standalone ranking. Initialization before train_begin and final root save after Trainer.train are excluded. Each reported phase is the maximum across ranks; phase maxima need not sum exactly to the maximum epoch duration.',
            })
