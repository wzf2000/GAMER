# Framework Reuse Refactor Report

## Scope

This report records the remaining high-value reuse opportunities in the Python
framework after the scripts refactor and generative backbone registry work.

The analysis focuses on:

- Decoder training tasks.
- Decoder evaluation tasks.
- Dataset loading task routing.
- Collator auxiliary-field handling.
- Generative model-family duplication.

## Findings And Recommended Order

### 1. Collator Auxiliary Field Handling

`SeqRec/datasets/collator.py` repeats padding and optional-field copying across
encoder-decoder train, decoder-only train, encoder-decoder test, and
decoder-only test collators.

Repeated fields include:

- `behavior`
- `uid`
- `session_ids`
- `extended_session_ids`
- `actions`
- `time`
- `inters_item_list`

Recommendation:

- Add small helpers for sequence padding and optional batch-field attachment.
- Preserve the current collator classes and call signatures.
- Keep decoder-only test left-padding and generated behavior-token alignment as
  explicit options.

Risk:

- Low. This is mostly mechanical reuse, but test collator left-padding must stay
  exactly equivalent.

Validation:

- `python -m compileall SeqRec/datasets/collator.py`
- Small batch smoke checks comparing tensor shapes, padding values, and copied
  list fields.

### 2. Registry Adoption In `train_decoder` And `train_MB_decoder`

`train_SMB_decoder.py` now uses `SeqRec.models.generative.registry`, but
`train_decoder.py` and `train_MB_decoder.py` still manually branch on backbone
for config/tokenizer loading, collator selection, model class imports, and label
names.

Recommendation:

- Reuse `load_config_and_tokenizer`, `instantiate_generative_model`,
  `is_decoder_only_backbone`, and backbone capability flags.
- Keep task-specific dataset and behavior-config differences local until the
  common training builder is introduced.

Risk:

- Medium. The code path affects training initialization, but the behavior should
  remain equivalent if the registry entries match the old branches.

Validation:

- `python -m compileall` for the touched training tasks.
- Registry config/tokenizer load smoke checks for supported backbone families.

### 3. Common Generative Training Builder

The decoder training tasks share a large block:

- Add dataset-specific new tokens.
- Save tokenizer and config.
- Select encoder-decoder or decoder-only collator.
- Configure PBAT/MoE behavior-aware config attributes.
- Instantiate model, set temperature, resize embeddings, move to device.
- Build `TrainingArguments` and `Trainer`.

Recommendation:

- Add a helper module such as `SeqRec/tasks/generative_training.py`.
- Extract low-level helpers first:
  - `prepare_tokenizer_and_config`
  - `build_train_collator`
  - `configure_behavior_aware_config`
  - `prepare_generative_model`
  - `build_training_arguments`
- Keep task-specific dataset loading and run-name scope in each task.

Risk:

- Medium. This should be staged after registry adoption so each task can be
  compared before the larger extraction.

Validation:

- Full Python compile.
- Small model/config initialization smoke checks.
- Full training jobs should be run separately because they require GPU time.

### 4. Common Generative Evaluation Utilities

`test_decoder.py`, `test_MB_decoder.py`, `test_SMB_decoder.py`, and the analysis
tasks repeat constrained-generation logic:

- Candidate trie construction.
- Decoder-only versus encoder-decoder generation kwargs.
- Session/action forwarding.
- Output slicing for decoder-only models.
- DDP result gathering.
- Metric accumulation and optional user-level metric output.

Recommendation:

- Add a helper module such as `SeqRec/tasks/generative_eval.py`.
- First extract pure helpers for generation kwargs, output decoding, trie
  construction, and DDP gathering.
- Keep each task's dataset/evaluation-type orchestration local.

Risk:

- Medium to high. Evaluation output must remain numerically equivalent.

Validation:

- Compile.
- Smoke tests with a small checkpoint when available.
- Compare pre/post metrics for at least one fixed checkpoint before relying on
  experiment results.

### 5. Dataset Loader Task Registry

`loading.py`, `loading_MB.py`, and `loading_SMB.py` manually map task strings to
dataset classes and constructor kwargs. SMB has the largest branching surface.

Recommendation:

- Introduce a lightweight `DatasetTaskSpec` only after the training/evaluation
  reuse work stabilizes.
- Start with SMB because it has the most active experiment variants.

Risk:

- Medium. Task-string compatibility is important for existing scripts.

Validation:

- Compile.
- Loader-construction smoke checks for every task string currently used by
  scripts and documentation.

### 6. Generative Model-Family Micro-Reuse

The Qwen3 multi-behavior family shares attention, router, mask, wrapper, and
temperature-handling patterns. `Qwen3TemporalHierarchical` already reuses some
`Qwen3Multi` base behavior, which suggests more reuse is possible.

Recommendation:

- Defer broad base-class extraction.
- Prefer small helper/mixin extraction after experiments stabilize:
  - temperature mixin
  - action/session mask builders
  - behavior embedding projection helper
  - generation position/session extension helper

Risk:

- High if done aggressively. These modules are close to model behavior and
  checkpoint compatibility.

Validation:

- Compile.
- Forward-pass smoke checks.
- Generation smoke checks.
- Existing checkpoint load checks.

## Execution Plan

1. Collator helper extraction.
2. Registry adoption in `train_decoder.py` and `train_MB_decoder.py`.
3. Common generative training helper extraction.
4. Common generative evaluation helper extraction.
5. Dataset loader registry feasibility pass.
6. Model-family micro-reuse feasibility pass.

Each implementation step should be followed by compile-level validation and a
focused smoke check for the changed abstraction.

## 2026-06-03 Execution Notes

Implemented in the first refactor batch:

- Collator auxiliary-field helpers in `SeqRec/datasets/collator.py`.
- Registry adoption in `train_decoder.py` and `train_MB_decoder.py`.
- Common generative training helpers in `SeqRec/tasks/generative_training.py`.
- Common generative evaluation helpers in `SeqRec/tasks/generative_eval.py`.
- Helper adoption in SMB evaluation and SMB analysis tasks.

Kept for a later batch:

- Dataset loader task registry. The active script/documentation surface uses a
  small set of task strings, but `loading_SMB.py` has enough train/valid/test
  mode-specific behavior that it should be migrated with dedicated loader smoke
  coverage for every task variant.
- Generative model-family base-class extraction. The temperature wrappers and
  action/session mask builders are repeated, but they sit close to checkpoint
  compatibility and model behavior. Prefer micro-extraction after the current
  temporal-hierarchical experiments are stable.
