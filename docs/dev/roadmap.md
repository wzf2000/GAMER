# Development Roadmap

## Current Priorities

- Keep project-level agent instructions and development workflow documentation on the `dev` branch.
- Maintain a paper-to-code map for the arXiv GAMER version so future implementation work can be tied back to the submitted method and experiments.

## Planned Work

- Prototype temporal-hierarchical cross-level modeling variants, starting with learnable relation bias in Qwen3Multi cross-attention.
- Verify whether graph baselines mentioned in the paper appendix are external, omitted from this snapshot, or implemented under alternate names.

## Updates

- 2026-06-01: Created the development documentation structure.
- 2026-06-01: Added the initial paper-to-code map for arXiv:2511.03155.
- 2026-06-01: Added a temporal-hierarchical behavior modeling design plan for the next GAMER revision.
- 2026-06-03: Completed the SeqRec package structural refactor (`a738e15..4bd0010`, 9 commits, -606 net lines):
  - `SeqRec/datasets/`: split `SMB_dataset.py` into the `session_behavior/` subpackage, moved loaders into `loaders/`, moved collators into `collators/`, renamed `seq_dataset` to `sequential`, and moved the discriminative session dataset under `discriminative/`. All legacy shim files removed.
  - `SeqRec/models/generative/`: split `mixins.py` into the `common/` subpackage (`cache`, `decoder_loop`, `temperature`, `wrappers`, `attention`, `session_masks`), regrouped the 11 model variants into family subpackages `qwen3/`, `llama/`, `pba_transformer/`, `tiger/`, removed the unused `Qwen3ActionMoe` variant, and registered the previously orphan `Qwen3SessionMoe` backbone with a corrected base config.
  - `SeqRec/tasks/`: regrouped tasks into category subpackages `training/`, `evaluation/`, `analysis/`, `tokenization/`. Replaced the `subclasses_recursive` scan in `__init__.py` with an explicit `registry.py` that maps parser names to `module:Class` strings and resolves each task module lazily on first access.
  - `SeqRec/utils/`: renamed `futils.py` → `fs.py`, `pipe.py` → `runtime.py`, `func_util.py` → `decorators.py`, `parse.py` → `args.py`, and dropped the now-unused `subclasses_recursive` helper.
