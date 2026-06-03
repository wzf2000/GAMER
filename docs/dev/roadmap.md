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
- 2026-06-04: Continued the refactor with task / model reuse extraction (`1ef8122..692644f`, 9 commits, -811 net lines):
  - `SeqRec/tasks/evaluation/`: merged `test_MB_rule.py` and `test_SMB_rule.py` into a single `rule.py` behind a shared `_BaseRuleTask`. Added `_BaseDecoderTestTask` for the three generative test tasks and progressively moved more scaffolding onto it (model load via registry, DDP setup, gather helpers, validation loop, user-level metric save, results JSON save, per-batch metric accumulator, pbar step, after-loop finalize). The three subclasses now delegate all backbone branching to registry predicates (`is_decoder_only_backbone`, `backbone_uses_actions`, `backbone_uses_sessions`); no hard-coded `if backbone == 'Qwen3'` checks remain. `build_generation_kwargs` was made tolerant of the BatchEncoding attribute-vs-dict gotcha (see `docs/dev/notes/2026-06-03-batch-encoding-attr-vs-dict.md`) so the Qwen3Multi MB path can use the helper too.
  - `SeqRec/tasks/analysis/`: extracted `_BaseAnalysisTask` carrying the shared model-load / trie-build / beam-search / rank-extraction scaffolding that `sparse_behavior` and `behavior_dropout` were each copying.
  - `SeqRec/models/generative/qwen3/`: consolidated `Qwen3MultiModelBase`, `Qwen3SessionMultiModelBase`, and `Qwen3SessionMoeModelBase` (~300 lines each, ≤22-line diffs) into a single `Qwen3DecoderModelBase` in `_decoder_base.py`. Then extended it with `_pre_layer_setup` / `_layer_kwargs` hooks so `Qwen3TemporalHierarchicalModel` also lives on the same base (no more manual `_update_causal_mask = Qwen3MultiModelBase._update_causal_mask` borrow). Verified byte-identical results JSON and per-uid metric files against pre-refactor baselines on 1-epoch Toy ckpts for all four families (Qwen3Multi MB, Qwen3SessionMulti SMB, Qwen3SessionMoe SMB, Qwen3TemporalHierarchical SMB).
