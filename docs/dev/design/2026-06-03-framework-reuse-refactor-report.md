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

## 2026-06-03 Post-P4 Follow-Up Refactor Options

After the P4 model refactor batch, the main decoder forward loops are much less
duplicated:

- Common decoder forward state preparation now lives in
  `SeqRec/models/generative/mixins.py`.
- Cross-level cache state and item-mask builders are centralized.
- Multi-cross decoder layer loops are shared by `Qwen3Multi`,
  `Qwen3SessionMulti`, and `LlamaMulti`.
- `Qwen3TemporalHierarchical` uses a separate temporal-hierarchical loop helper
  instead of being forced into the multi-cross abstraction.

The remaining high-value duplication is now mostly in parser definitions,
backbone metadata, model construction/profile setup, and model component
definitions.

### P1. Training Parser Argument Groups

`train_decoder.py` and `train_MB_decoder.py` still repeat almost identical
training arguments, including optimizer, epoch count, learning rate, batch size,
gradient accumulation, logging cadence, sequence length, weight decay, resume
path, scheduler, save/eval strategy, precision flags, DeepSpeed path,
temperature, and W&B run name.

Recommendation:

- Add parser group helpers in `SeqRec/utils/parse.py`, starting with:
  - `parse_training_args(parser)`
  - optionally later `parse_generation_eval_args(parser)`
  - optionally later `parse_analysis_args(parser)`
- Apply `parse_training_args` first to decoder training tasks because it is
  low-risk and immediately reduces boilerplate for new task variants.

Risk:

- Low. This should preserve the exact same argument names, defaults, and help
  text.

Validation:

- `python main.py train_decoder --help`
- `python main.py train_MB_decoder --help`
- Compile and argument snapshot comparison for the touched tasks.

### P2. Generative Training Profile Setup

`TrainDecoder.invoke` and `TrainMBDecoder.invoke` now share many helper calls,
but model/config setup still repeats profile-specific logic. The most sensitive
part is behavior-aware config preparation for `pba` and `multi_behavior`
profiles:

- behavior token collection
- `behavior_maps`
- `num_behavior`
- `use_behavior_token`
- behavior-injection disabling
- `num_positions`
- `num_experts`
- `n_positions`
- `model_max_length`

Recommendation:

- Add a focused helper in `SeqRec/tasks/generative_training.py`, for example:
  `prepare_generative_model_config(...)`.
- Keep dataset loading and trainer creation outside this helper initially.
- Treat full `invoke` orchestration extraction as a later step after profile
  setup is stable.

Risk:

- Medium. The MB and non-MB paths differ in behavior-item formatting and target
  behavior handling.

Validation:

- Compile.
- Profile-level config smoke checks for `basic`, `pba`, and `multi_behavior`.
- Existing lightweight training task construction checks where feasible.

### P3. Single Source of Truth for Backbone Metadata

Python uses `SeqRec/models/generative/registry.py`, while shell scripts still
use `scripts/lib/s2s_backbone.sh` for backbone aliases and base-model path
resolution. This leaves two metadata sources to update whenever a new backbone
or comparison variant is added.

Recommendation:

- Extend the Python registry with optional alias and default base-model path
  metadata.
- Add a small CLI entrypoint, for example:
  - `python -m SeqRec.models.generative.registry resolve-backbone <name>`
  - `python -m SeqRec.models.generative.registry resolve-base-model <name>`
- Let shell helpers call the Python registry, so Python becomes the source of
  truth.

Alternative:

- Store shared metadata in a JSON file and let both Python and shell read it.
  This is workable but introduces another file format and another migration
  point.

Risk:

- Medium. Shell workflows are user-facing and must preserve existing aliases
  such as `Qwen3Session2`, `Llama`, `Qwen3Multi*`, and
  `Qwen3TemporalHierarchical*`.

Validation:

- `bash -n` for all touched scripts.
- Alias resolution smoke tests for current documented script backbones.
- Registry load checks for representative backbone families.

### P4. Qwen/Llama Multi-Behavior Component Reuse

The P4 loop extraction reduced forward-loop duplication, but component classes
still repeat substantial structure:

- `Qwen3MultiAttention` and `LlamaAttention`
- `Qwen3MultiDecoderLayer` and `LlamaMultiDecoderLayer`
- cross behavior embedding and gating
- self-attention, optional cross-attention, MLP residual sequencing

Recommendation:

- Do not immediately merge these into one large base class.
- Prefer micro-reuse in this order:
  1. Extract cross behavior embedding and gating helpers.
  2. Extract a decoder-layer residual flow helper.
  3. Only then consider a shared decoder-layer base class.

Risk:

- Medium to high. Qwen and Llama differ in normalization, attention warnings,
  sliding-window behavior, and Q/K normalization details. An overly broad base
  class could make checkpoint or behavior compatibility harder to reason about.

Validation:

- Compile.
- Tiny forward smoke checks where configs can be constructed.
- Existing checkpoint load checks before using the refactor for experiments.

### P5. Lazy Model-Class Import Registry

`get_generative_model_cls` still uses a chain of explicit `if backbone == ...`
branches. Adding a new model requires editing both the metadata dict and the
class import branch.

Recommendation:

- Add a model import path to `GenerativeBackboneSpec`, for example:
  `model_cls_path="SeqRec.models.generative.Qwen3Multi:Qwen3MultiWithTemperature"`.
- Resolve it with `importlib` at call time.
- Keep imports lazy so registry import remains cheap and does not eagerly import
  every model family.

Risk:

- Low. The external behavior should remain the same if paths match the existing
  branches.

Validation:

- Compile.
- Registry smoke checks for every supported backbone:
  `get_generative_model_cls(backbone)`.

### Recommended Follow-Up Order

1. P1 training parser argument groups.
2. P5 lazy model-class import registry.
3. P3 Python-backed shell backbone metadata resolution.
4. P2 generative model config/profile setup helper.
5. P4 Qwen/Llama component-level reuse.

This order prioritizes low-risk reductions in new-task and new-model wiring
before touching model internals again.
