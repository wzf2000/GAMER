# Scripts Maintenance Refactor

## Context

The training, evaluation, tokenization, and analysis workflows under `scripts/`
have accumulated repeated shell logic. Adding a new sequence-to-sequence
backbone, such as `Qwen3TemporalHierarchical` and its variants, previously
required touching several scripts for the same decisions:

- External script backbone name to internal model backbone mapping.
- Base model/config directory resolution.
- Tokenization/index-file naming.
- Output directory and result-file construction.
- `extra_args` and `extra_flags` expansion.
- GPU counting, per-device batch-size calculation, and single-GPU versus
  `torchrun` dispatch.

This makes model comparison work fragile because a new backbone can be wired in
for training but missed in testing or analysis.

## Design Goal

Keep the script command-line interface stable while moving repeated decisions
into small shell helpers and Python registries. Scripts should remain readable
entrypoints, but they should delegate shared naming and routing rules to common
functions.

The desired outcome is that most future model variants only require:

1. A config directory under `config/s2s-models/`.
2. A shell-level alias in `scripts/lib/s2s_backbone.sh` if the external script
   name differs from the internal Python backbone name.
3. A Python registry entry in `SeqRec/models/generative/registry.py` if the
   model class/config/tokenizer family is new.

## Completed Refactor Batches

### Backbone Resolution

Added `scripts/lib/s2s_backbone.sh` to centralize S2S backbone handling.

Current responsibilities:

- `resolve_s2s_backbone_arg`: maps external script names such as
  `Qwen3TemporalHierarchicalMultiView` to the internal task backbone
  `Qwen3TemporalHierarchical`.
- `resolve_s2s_base_model`: maps script backbone names to config directories
  under `config/s2s-models/`.

This removed repeated `case`/`if` handling from SMB training, SMB testing, and
analysis scripts.

### Shell Argument Helpers

Added `scripts/lib/args.sh`.

Current responsibilities:

- `parse_extra_args`: converts comma-separated `key=value` pairs into argparse
  options.
- `parse_extra_flags`: converts comma-separated flag names into argparse
  boolean flags.

This keeps the existing `extra_args=a=b,c=d` and `extra_flags=x,y` script
interfaces unchanged while avoiding repeated `awk` snippets.

### Tokenization Naming

Added `scripts/lib/tokenization.sh`.

Current responsibilities:

- Resolves tokenization mode from existing environment variables:
  `original`, `rid`, `cid`, `shuffle`, `chunk_size`, `rq_kmeans`, `cf_emb`,
  `reduce`, `alpha`, `beta`, and `epoch`.
- Produces a consistent `token_tag`, `index_file`, and `tokenization_desc`.

The helper preserves existing path conventions, for example:

- `original` -> `.index.json`
- `rid` -> `.index.rid.json`
- `cid` -> `.index.cid.chunk<N>.json`
- `rq-kmeans-cf-reduce` -> `.index.rq-kmeans-cf-reduce.json`
- default RQ-VAE -> `.index.epoch<E>.alpha<A>-beta<B>.json`

### Path Builders

Added `scripts/lib/paths.sh`.

Current responsibilities:

- `build_task_dir`: builds `${dataset}/${task_name}/${backbone}` with optional
  suffix support.
- `build_checkpoint_path`: builds checkpoint paths from scope, task directory,
  and tokenization tag.
- `build_result_path`: builds result-file paths under `./results`.

This keeps path naming centralized across train/test/analyze scripts.

### Runtime Helpers

Added `scripts/lib/runtime.sh`.

Current responsibilities:

- `count_gpus`: counts comma-separated GPU IDs.
- `compute_per_device_batch_size`: derives per-device batch size from global
  batch size and GPU count.
- `run_main_distributed`: dispatches to `python main.py` for one GPU and
  `torchrun` for multiple GPUs.

This removes repeated single-GPU/multi-GPU launch branches from decoder scripts.

### Python Generative Backbone Registry

Added `SeqRec/models/generative/registry.py`.

Current responsibilities:

- Records whether each generative backbone is decoder-only.
- Records whether a backbone consumes session IDs and action/behavior-level
  tensors.
- Resolves config/tokenizer loading families.
- Resolves model classes and model instantiation.
- Records the training profile used by `train_SMB_decoder.py`:
  `basic`, `pba`, `session`, or `multi_behavior`.

The SMB train/test/analysis tasks now use registry helpers instead of repeating
model class imports and backbone membership lists.

## Validation Rules

Every script refactor batch should run at least:

```bash
for f in scripts/*.sh scripts/lib/*.sh; do bash -n "$f" || exit 1; done
```

For helper changes, run focused shell checks for the touched helpers, for
example tokenization mode expansion, path construction, GPU counting, and
backbone alias resolution.

For Python registry or task changes, run:

```bash
python -m compileall main.py SeqRec
```

When config/tokenizer routing changes, also run a small registry load check in
the project environment, for example loading the target backbone config and
tokenizer from `config/s2s-models/<name>`.

Full training or evaluation jobs are not part of routine refactor validation
because they require GPU time, datasets, and checkpoints. They should be run as
experiment verification after the script-level checks pass.

## Remaining Optimization Opportunities

- Extend `scripts/lib/s2s_backbone.sh` into MB decoder scripts if future MB
  workflows need the same variant alias support.
- Consider a single task-launch helper for analysis scripts if they gain
  distributed inference support.
- Consider adding lightweight shell-unit checks under `scripts/tests/` once the
  helper surface stabilizes.
- Keep generated data, checkpoints, logs, results, runs, and wandb outputs out
  of these refactor commits.
