# Project Instructions

## Project Overview

- This repository is `GAMER`: Generative Augmentation and Multi-Level Behavior Modeling for Sequential Recommendation.
- The codebase is a Python/PyTorch recommendation project. Core source code lives under `SeqRec/`.
- `main.py` is the central task entrypoint. Task implementations live in `SeqRec/tasks/`.
- Shell workflows for training, tokenization, embedding generation, and evaluation live in `scripts/`.
- Documentation for datasets and scripts lives in `docs/datasets.md` and `docs/scripts.md`.

## Repository Layout

- `SeqRec/datasets/`: dataset loading, processing, and collators.
- `SeqRec/evaluation/`: ranking and evaluation metrics.
- `SeqRec/generation/`: generation utilities.
- `SeqRec/models/`: model definitions for discriminative models, generative models, and tokenizers.
- `SeqRec/modules/`: reusable model modules and losses.
- `SeqRec/tasks/`: task definitions and main training/evaluation logic.
- `SeqRec/trainers/`: trainer implementations.
- `SeqRec/utils/`: configuration, filesystem, logging, and utility helpers.
- `config/`: model and tokenizer configuration files.
- `scripts/`: runnable shell scripts. Run them from the repository root.
- `data/`, `checkpoint/`, `logs/`, `results/`, `runs/`, and `wandb/` are runtime/data/output areas; avoid committing generated artifacts unless explicitly requested.

## Environment

- Target Python version: Python 3.12+.
- Target PyTorch version: PyTorch 2.7+.
- Install dependencies with `pip install -r requirements.txt`.
- Install CUDA-specific PyTorch wheels separately when needed; see `README.md`.
- Do not add or upgrade heavyweight ML dependencies without asking first.

## Common Commands

- Inspect available task arguments through `python main.py <task> --help`.
- Single-GPU training/evaluation scripts usually accept `gpu=0`.
- Multi-GPU scripts use `torchrun` and derive per-device batch size from `batch_size / gpu_num`.
- Script environment variables use `name=value` prefixes, for example:
  - `gpu=0 dataset=Retail bash scripts/train_SMB_decoder.sh`
  - `dataset=ShortVideoAD original=1 batch_size=256 tasks=smb_explicit_decoder_4 gpu=0 backbone=Qwen3Multi bash scripts/test_SMB_decoder.sh`
- `extra_args` is comma-separated `key=value` with no spaces, for example `extra_args=max_his_len=100,warmup_ratio=0.04`.
- `extra_flags` is comma-separated flag names with no spaces, for example `extra_flags=foo,bar`.

## Coding Conventions

- Follow the existing Python style: straightforward modules, explicit names, argparse-based task configuration, and loguru logging.
- Prefer extending existing task, dataset, model, trainer, and utility patterns instead of introducing a parallel framework.
- Keep changes scoped to the requested task. Avoid broad refactors unless they are needed for correctness.
- Preserve existing command-line argument names and script environment-variable interfaces unless the user explicitly asks for a breaking change.
- Use structured parsers and existing utilities for config/data handling where available; avoid ad hoc string parsing when a local helper already exists.
- Keep comments brief and only add them when they clarify non-obvious logic.

## Data And Artifacts

- Follow dataset formats documented in `docs/datasets.md`.
- Do not commit raw datasets, checkpoints, logs, runs, wandb output, or large binary artifacts.
- `.gitignore` intentionally ignores most generated data and model artifacts while allowing lightweight dataset JSON metadata/index files.
- Be careful with commands that write to `data/`, `checkpoint/`, `results/`, `logs/`, `runs/`, or `wandb/`; explain the expected outputs before running expensive jobs.

## Testing And Verification

- For shell script edits, run `bash -n <script>` at minimum.
- For Python syntax-level checks, prefer `python -m compileall main.py SeqRec` when practical.
- For behavior changes, use the smallest relevant task/script invocation first. Full training jobs can be expensive and should not be run without user confirmation.
- If GPU, data, checkpoint, or network requirements prevent verification, state exactly what was not run and why.

## Agent Workflow

- Read `README.md`, `docs/scripts.md`, and the relevant `SeqRec/tasks/` file before changing a workflow.
- Use `rg`/`rg --files` for code search.
- Do not overwrite user changes. Check `git status --short` before and after edits when modifying files.
- Ask before deleting files, resetting git state, installing dependencies, downloading datasets/checkpoints, or launching long-running training.
- Keep generated project instructions concise and update this file when project-wide conventions change.
