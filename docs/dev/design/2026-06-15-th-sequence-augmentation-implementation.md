# Temporal-Hierarchical Sequence Augmentation Implementation Plan

## Purpose

This document evaluates how the proposed temporal-hierarchical (TH) sequence-augmentation methods can be implemented in the current codebase. It records:

- implementation feasibility and expected cost;
- reusable code and required refactoring;
- concrete module, class, configuration, and testing paths;
- data-leakage and evaluation-protocol constraints;
- the recommended implementation order.

The scope is data-side augmentation for `train_SMB_decoder`. No implementation is included in this document.

## Current Code Assessment (Implemented, Refactoring Recommended)

The current implementation already exposes the information needed by most proposed methods:

| Required signal | Current source |
| --- | --- |
| Behavior identity and level | `history_behaviors`, `behavior_level` |
| Target behavior | `target_behavior` |
| Interaction order | List order in each user history |
| Session boundary | `session` |
| Timestamp | `time`, normalized in half-hour units |
| Split boundary | `valid_pos`, `test_pos` |
| Deterministic user randomness | `_user_seed(uid)` in the fixed-ratio dataset |

The main limitation is architectural rather than informational:

- augmentation logic is embedded in complete Dataset subclasses;
- the decoder augmentation and fixed-ratio implementations duplicate filtering and sample assembly;
- task names encode parameters such as augmentation count and level ratios;
- cache names are manually constructed by each Dataset;
- static preprocessing and pickle caching make epoch-dependent augmentation difficult;
- the fixed-ratio task also modifies validation and test histories, which is useful for robustness evaluation but should not be the default protocol for training-only augmentation comparisons.

Most static strategies are therefore straightforward after a small policy-layer refactor.

## Decoder Pretraining Semantics (Design Correction, Pending Code Alignment)

The intended training unit for `smb_explicit_decoder` is a full causal sequence, not a single fixed target. The existing `SMBExplicitDatasetForDecoder` stores each training sample as:

```text
inters = sequence[:-1]
item = sequence[-1]
```

but `DecoderOnlyCollator` concatenates `input_ids + labels` and, for decoder-response datasets, does not mask the `input_ids` portion during training. Therefore the actual loss is causal LM shift loss over the full sequence:

```text
sequence[:-1] + sequence[-1]
```

The original `smb_explicit_decoder_4` augmentation should be interpreted as full-sequence augmentation:

```text
original full sequence
+ 4 ratio-indexed full-sequence dropout views
```

Each view is then represented with the legacy `inters=view[:-1]`, `item=view[-1]` schema only to reuse the existing collator and evaluation code.

The earlier policy dataset implementation deviated from this intended protocol. It first fixed `sequence[-1]` as a prediction target, applied policy dropout only to `sequence[:-1]`, and then appended the fixed target back. This produced:

```text
policy(history) + original_tail
```

rather than:

```text
policy(full_sequence)
```

Consequently, the completed policy-augmentation runs from that version should be treated as history-only semantic-dropout diagnostics, not as the final policy version aligned with the original decoder augmentation. The implementation has since been updated so every policy generates views over the full training sequence. Each generated view is split back into `inters=view[:-1]` and `item=view[-1]` for compatibility.

Design requirements for the aligned protocol:

- Always include the original full sequence unless `augmentation_drop_original` is explicitly set.
- Apply augmentation policies to `items[:valid_pos]`, `behaviors[:valid_pos]`, `session_ids[:valid_pos]`, and `times[:valid_pos]`.
- Keep chronological order and aligned fields unchanged after filtering.
- Require each emitted view to contain at least one interaction. A one-interaction view is represented as `inters=""` and `item=view[-1]`, matching the original decoder behavior; empty policy views are skipped.
- Preserve protected behavior levels according to each policy, but do not globally force the original last interaction to remain the tail unless that is part of a named policy.
- Keep validation and test datasets unchanged: evaluation remains `history -> candidate target`.
- Change cache identity when switching from history-only policy views to full-sequence policy views.

## Implementation Status Summary (Living)

| Scheme or component | Implementation | Verification | Next role |
| --- | --- | --- | --- |
| Shared policy interface and structured sequence | Implemented | Unit tests, compileall, and flake8 passed | Foundation for further policies |
| Unified decoder dataset | Implemented and full-sequence aligned | Synthetic loader, sample-schema tests, compileall, and flake8 passed | Common entry for policy augmentation |
| Explicit arguments, cache key, and `smb_policy_decoder` | Implemented | CLI, cache isolation, and legacy task parsing verified | Continue using |
| Time-Decayed Behavior Dropout | Implemented and full-sequence aligned | Determinism, recency protection, level protection, and full-sequence dataset behavior verified | Re-run under corrected protocol |
| Session-Aware Dropout | Implemented and full-sequence aligned | Session atomicity, minimum-history protection, and full-sequence dataset behavior verified | Re-run under corrected protocol |
| Dataset-Level Fixed Proportion | Implemented and full-sequence aligned | Soft cap, top-level protection, and full-sequence dataset behavior verified | Corrected-protocol control experiment |
| Training-prefix behavior statistics | Implemented | Verified to use only `history[:valid_pos]` | Supplies global priors |
| User-Adaptive Ratio | Implemented and full-sequence aligned | Zero-target fallback, training-prefix prior, and full-sequence dataset behavior verified | Re-run under corrected protocol |
| Target-Conditioned Augmentation | Implemented as tail-conditioned full-sequence policy | Same-level/precursor restoration and full-sequence dataset behavior verified | Re-run under corrected protocol |
| Multi-View Sequence Augmentation | Implemented and full-sequence aligned | Semantic view generation, Dataset deduplication, and full-sequence dataset behavior verified | Re-run under corrected protocol |
| Curriculum Augmentation | Not implemented | Not verified | Deferred |

## Shared Architecture (Implemented And Verified)

### Policy Interface (Implemented And Verified)

Add a strategy module:

```text
SeqRec/datasets/session_behavior/augmentation_policies.py
```

Use a structured input rather than passing four parallel lists:

```python
@dataclass(frozen=True)
class BehaviorSequence:
    items: list[str]
    behaviors: list[str]
    session_ids: list[int]
    times: list[float]


@dataclass(frozen=True)
class AugmentedView:
    name: str
    keep_indices: list[int]
    metadata: dict[str, Any]


class SequenceAugmentationPolicy(Protocol):
    def generate_view(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> AugmentedView:
        ...
```

`AugmentationContext` should contain only causal information:

```text
uid
target_behavior
target_level
target_time
behavior_level
max_behavior_level
```

Policies should return indices, not copied item arrays. A common helper applies the mask once to every aligned field. This prevents `items`, `behaviors`, `session_ids`, and `times` from becoming misaligned.

### Unified Decoder Dataset (Implemented And Verified)

Add:

```text
SeqRec/datasets/session_behavior/augmented_decoder.py
```

Implemented class:

```text
SMBPolicyAugmentedDatasetForDecoder
```

It inherits from `SMBExplicitDatasetForDecoder`, so the existing decoder-only collator logic continues to recognize it through `isinstance`.

Responsibilities:

1. Construct the full causal training sequence for each user prefix.
2. Build a deterministic RNG from user, split, and optional view id.
3. Call the selected augmentation policy.
4. Validate returned indices.
5. Split each emitted full-sequence view into the existing decoder schema:
   `item`, `inters`, `session_ids`, `extended_session_ids`, `actions`, `time`, and `behavior`.
6. Emit the original full-sequence view when configured.

The policy should not construct tokenizer strings or token-level metadata.

### Policy Registry (Partly Implemented)

Supported policies are currently resolved by `SMBPolicyAugmentedDatasetForDecoder._build_policy()`. A separate registry module has not been added. It can be introduced later at:

```text
SeqRec/datasets/session_behavior/augmentation_registry.py
```

Currently supported:

```text
time_decay
session
dataset_proportion
user_adaptive_ratio
target_conditioned
multi_view
```

Not yet supported:

```text
none
uniform_level
fixed_ratio
```

Do not create one complete Dataset class per strategy.

### Explicit Arguments (Implemented And Verified)

Add augmentation fields to `DatasetArgs` rather than encoding all values into `--tasks`:

```text
--sequence_augmentation none
--augmentation_views 1
--augmentation_seed 42
--augmentation_drop_original
--time_decay_type exponential
--time_decay_tau 48
--recent_session_count 1
--dataset_proportion_preset natural
```

Explicit arguments are wired through `DatasetArgs` and `TrainSMBDecoder.load_train_data`. The unified task is:

```text
--tasks smb_policy_decoder
```

Current status:

- retain existing task names for backward compatibility;
- `smb_policy_decoder` has been added;
- select the policy and parameters through explicit dataset arguments;
- deprecate new parameter-encoded task names.

A compact JSON/config-file interface is not implemented and can be reconsidered if the flat argument list grows further.

### Cache Identity (Implemented And Verified)

Every static policy must expose a stable serializable configuration:

```python
policy.cache_config()
```

The dataset cache name should include:

- policy name;
- a short hash of normalized policy configuration;
- augmentation seed;
- number of views;
- whether the original view is included;
- split and index suffix.

This avoids stale cache reuse when a policy parameter changes.

### Training And Evaluation Protocol (Default Implemented)

Default protocol:

```text
train: augmented full training sequence
valid: original history
test: original history
```

This isolates augmentation as a training intervention.

The intended `smb_policy_decoder` protocol is the full-sequence training protocol above, and the current implementation follows this protocol. Earlier history-only policy results should remain diagnostic only. The following separate robustness protocol is not yet integrated into the policy dataset:

```text
train: augmented or original history
valid/test: explicitly corrupted history
```

The current fixed-ratio behavior across train, validation, and test should remain available as a named robustness experiment, not as the default comparison.

## Time-Decayed Behavior Dropout (Implemented And Verified, First Experiments)

### Reuse

Directly reuse:

- `times`;
- `behavior_level`;
- protected-level preservation;
- deterministic RNG;
- existing full-history plus augmented-view behavior.

No model, collator, tokenizer, or sample-schema changes are required.

### Policy

Add:

```text
TimeDecayDropoutPolicy
```

For each interaction in the full training sequence:

```text
p_drop(i) = severity * level_weight(level_i) * age_weight(delta_t_i)
```

Because the current `time` values increase from the user's first interaction, use:

```text
delta_t_i = target_time - time_i
```

Recommended first implementation:

```text
age_weight = 1 - exp(-delta_t / tau)
level_weight(l) = 1 / (l + 1)
```

Thus recent interactions have near-zero age weight, old low-level actions are most likely to be removed, and protected high-level interactions are preserved or assigned a very small weight.

### Required Safeguards

- Preserve at least the minimum number of interactions required to form a decoder sample.
- Preserve the most recent `min_recent_items`.
- Optionally preserve all target-level interactions.
- Clamp probabilities to `[0, max_drop_probability]`.
- Handle equal timestamps and zero time spans.

### Suggested Arguments

```text
time_decay_type = exponential | linear_rank | bucket
time_decay_tau = 48.0
time_decay_severity = 0.5
time_decay_max_drop = 0.9
time_decay_min_recent_items = 1
time_decay_preserve_target_level = true
```

### Tests

- old interactions have a higher empirical drop rate than recent interactions;
- low-level interactions have a higher drop rate than high-level interactions;
- aligned fields remain aligned;
- fixed seed produces identical views;
- no future timestamp is used;
- required recent and protected-level items are preserved;
- zero-span timestamps do not fail.

## Session-Aware Dropout (Implemented And Verified, First Experiments)

### Reuse

The current data already provides normalized session ids and session boundaries. No preprocessing change is required.

### Policy

Add:

```text
SessionAwareDropoutPolicy
```

First group full-sequence indices by session. Compute session-level features using only that session:

```text
recency
maximum behavior level
contains target-level behavior
interaction count
```

Recommended first rule:

1. Always keep the latest `recent_session_count` sessions.
2. Always keep the most recent session containing a high-level behavior, if one exists.
3. Sample older sessions with a probability based on recency and maximum behavior level.
4. Keep selected sessions intact.

Optional second-stage behavior dropout can be added later, but the first implementation should preserve all interactions within a selected session.

### Suggested Arguments

```text
recent_session_count = 1
session_keep_probability = 0.5
session_time_decay_tau = 7
session_high_level_bonus = 0.3
session_preserve_target_level = true
```

### Cache And Split Detail

Use user seed plus stable session id. Avoid assigning randomness based on a session's position in a truncated prefix, or the same historical session may receive different decisions across train and validation prefixes.

### Tests

- selected sessions are kept atomically;
- the latest required sessions are always preserved;
- historical decisions are stable across compatible prefixes;
- session order remains unchanged;
- single-session users remain valid;
- sparse users retain enough history.

## User-Adaptive Ratio (Implemented And Verified, Second Experiments)

### Feasibility

The strategy is implementable with current histories, but the statistical estimator needs more care than the filtering code.

### Policy

Add:

```text
UserAdaptiveRatioPolicy
```

Estimate a causal per-user behavior ratio from the current history prefix:

```text
r_user,l = (count_user,l + alpha * r_global,l) / (count_user,target + alpha)
```

Then shrink it toward a training-set global prior:

```text
r_final,l = confidence_u * r_user,l
          + (1 - confidence_u) * r_global,l
```

Use bounded caps:

```text
r_final,l = clip(r_final,l, min_ratio_l, max_ratio_l)
```

The policy downsamples only when a level exceeds its cap.

### Global Statistics

Add a training-only statistics helper:

```text
SeqRec/datasets/session_behavior/statistics.py
```

It should compute aggregate counts only from training prefixes:

```text
history[:valid_pos]
```

Store the result in a versioned lightweight JSON cache. Validation and test must reuse the training-derived prior rather than recomputing it from future data.

### Zero-Target Users

Do not return the original sequence unconditionally when the target count is zero. Use a smoothed denominator or a sequence-length cap:

```text
cap_l = min(
    ratio_cap_from_global_prior,
    level_share_cap_l * history_length,
)
```

### Risks

- User conversion propensity may be encoded too directly in the augmentation pattern.
- A strongly target-count-based rule can remove useful differences between users.
- Very sparse users require strong shrinkage.

For these reasons, compare against dataset-level fixed proportion before treating this as the main augmentation.

### Tests

- statistics use only training prefixes;
- sparse users fall back toward the global prior;
- zero-target users are handled;
- ratios stay within configured bounds;
- fixed seed and prefix rules are deterministic.

## Dataset-Level Fixed Behavior Proportion (Implemented And Verified, Control)

### Policy

Add:

```text
DatasetProportionPolicy
```

Reuse the same training-only statistics helper. Support two presets:

```text
natural: preserve the observed training distribution
balanced: cap dominant shallow behaviors
```

Apply a soft cap rather than forcing exact proportions:

```text
max_count_l = ceil(history_length * target_share_l * tolerance)
```

Only overrepresented levels are downsampled.

### Why It Is Useful

- It handles users with no target behavior.
- It is a cleaner global control than per-user target-count normalization.
- It separates the benefit of distribution balancing from personalization.

### Tests

- statistics are training-only;
- no level below its cap is modified;
- output approaches, but is not forced to equal, the configured distribution;
- short histories are protected.

## Tail-Conditioned Augmentation (Implemented As Target-Conditioned, Needs Protocol Review)

### Feasibility

Under the full-sequence decoder protocol, augmentation does not have a separate fixed prediction target. The original sequence tail can still be used as an anchor for a named tail-conditioned policy, so no model change is needed. The policy can use:

```text
context.target_level
```

Here `target_level` should be read as the original sequence-tail behavior level, not as a fixed single-token training label.

### Policy

Add:

```text
TargetConditionedPolicy
```

Recommended implementation is a wrapper around another base policy:

```text
TargetConditionedPolicy(base_policy=time_decay)
```

The current implementation supports only `time_decay` as the base policy. Other base-policy combinations are not implemented.

It modifies level-preservation weights according to the anchor behavior:

```text
same-level evidence
one-level-below evidence
upward-path evidence
general temporal evidence
```

For a high-level anchor, retain recent lower-level precursors. For a shallow anchor, favor same-level and recent temporal evidence.

### Leakage Constraint

Using the sequence-tail behavior as an augmentation condition is valid for training-time data generation, but it changes the augmentation distribution. If the policy is meant to mirror behavior-specific inference, the anchor behavior must correspond to a known behavior prompt. If the production task requires predicting behavior type itself, strongly target-conditioned retention patterns can still create distribution shortcuts.

This policy should be reported as a stronger augmentation prior rather than as the default data-side protocol. It also needs distribution diagnostics before being used as the main augmentation.

### Distribution Shortcut Risk

The number or type of retained interactions must not uniquely reveal the target level. Mitigations:

- use overlapping keep-probability ranges across target levels;
- keep expected sequence lengths similar;
- include an unconditioned full view;
- report a classifier probe that predicts target level from augmentation metadata or simple count features.

### Tests

- only the current sample target is used;
- no future behavior is inspected;
- expected lengths are comparable across target levels;
- target-conditioned weights change the intended relation categories.

## Multi-View Sequence Augmentation (Implemented And Verified, Second Experiments)

### Feasibility

The current original decoder dataset already emits multiple full-sequence augmented views per user. The policy dataset should keep this protocol and replace ratio-indexed full-sequence views with named semantic full-sequence policies.

### Policy

Add:

```text
MultiViewAugmentationPolicy
```

It composes existing policies or deterministic selectors:

```text
full
recent
same_level
upward_evidence
session_subsampled
```

Recommended first view set:

1. `full`: unchanged full training sequence.
2. `recent`: time-decayed or fixed recent window.
3. `hierarchy`: preserve target-level, same-level, and one-level-below evidence.
4. `session`: session-aware subsample.

Do not enable every possible view initially.

The dataset provides the optional original full-sequence view separately. The policy generates full-sequence semantic views:

```text
multi_view_recent
multi_view_hierarchy
multi_view_session
```

Views with identical keep indices are deduplicated by the dataset. `augmentation_views` denotes the number of repeated samples of the complete semantic-view set.

### Sample Weighting

Simply duplicating each user once per view changes both dataset size and user weighting. Support:

```text
view_sampling = all | one_per_epoch | weighted_static
view_weights
```

The initial cache-compatible implementation can use `all` or `weighted_static`. `one_per_epoch` requires online sampling and belongs to a later stage.

If Trainer-compatible sample weights are not currently consumed by the loss, implement weighted-static sampling at dataset construction rather than adding an unused `sample_weight` field.

### Relation To Model-Side MultiView

Data-side views should remain independently usable with TH Base and relation-bias models. The main experiment matrix should distinguish:

```text
data MultiView only
model MultiView only
both
```

This prevents attributing an improvement to the wrong component.

### Tests

- each configured full-sequence view has the declared semantics;
- the full view is unchanged;
- no duplicate view is emitted when two policies produce the same keep indices;
- maximum views per user is enforced;
- dataset-size growth is predictable;
- view ordering and seeds are deterministic.

## Curriculum Augmentation (Not Implemented, Deferred)

### Why Current Static Caching Is Insufficient

Current datasets preprocess all samples in `__init__` and pickle `inter_data`. A curriculum based on epoch or global step cannot change these cached samples.

### Required Architecture

Choose one of:

1. Online augmentation in `__getitem__`, with shared epoch state.
2. A sampler selecting among precomputed views based on epoch.
3. A Trainer callback updating the dataset policy or sampler.

The preferred future path is precomputed semantic views plus an epoch-aware sampler. It avoids repeatedly rebuilding token strings and keeps augmentation cost controlled.

Possible modules:

```text
SeqRec/datasets/session_behavior/view_dataset.py
SeqRec/trainers/callbacks/augmentation_schedule.py
SeqRec/datasets/samplers/curriculum_view_sampler.py
```

### Additional Requirements

- `set_epoch(epoch)` propagation under DDP;
- deterministic behavior across ranks;
- resume-from-checkpoint restoration of curriculum state;
- worker-safe state with `num_workers > 0`;
- cache identity independent of current epoch;
- logging of view distribution per epoch.

This should be attempted only after static semantic views demonstrate benefit.

## File-Level Implementation Progress

### Stage 1: Shared Static Policy Layer (Implemented And Verified)

Add:

```text
SeqRec/datasets/session_behavior/augmentation_policies.py
SeqRec/datasets/session_behavior/augmented_decoder.py
```

Modify:

```text
SeqRec/datasets/session_behavior/__init__.py
SeqRec/datasets/loaders/session_behavior.py
SeqRec/tasks/training/train_SMB_decoder.py
SeqRec/utils/args.py
```

Legacy uniform-level and fixed-ratio task parsing remains compatible, but those implementations have not been migrated into policy adapters.

### Stage 2: First Policies (Implemented And Verified)

Implement and test:

1. `TimeDecayDropoutPolicy`.
2. `SessionAwareDropoutPolicy`.
3. `DatasetProportionPolicy`.

These have clear semantics and low leakage risk.

### Stage 3: Adaptive And Composed Policies (Implemented And Verified)

Implement:

1. training-only behavior statistics;
2. `UserAdaptiveRatioPolicy`;
3. `TargetConditionedPolicy`;
4. `MultiViewAugmentationPolicy`.

### Stage 4: Online Sampling And Curriculum (Not Implemented, Deferred)

Only proceed if static MultiView is useful and cached dataset expansion becomes a practical bottleneck.

## Verification Status

### Implemented Tests

Create:

```text
tests/datasets/session_behavior/test_augmentation_policies.py
tests/datasets/session_behavior/test_policy_augmented_dataset.py
```

The current thirteen synthetic tests cover:

- one and multiple sessions;
- time-decay recency and top-level protection;
- fixed-seed reproducibility;
- atomic session retention and minimum-history protection;
- dataset-proportion soft caps;
- aligned-field validation;
- loader construction for all three policies;
- augmented training with original validation;
- legacy task parsing compatibility.
- training statistics restricted to training prefixes;
- User-Adaptive fallback when target count is zero;
- Target-Conditioned restoration of same-level and precursor evidence;
- named Multi-View outputs and hierarchy-view constraints;
- loader construction for all six static policies.

### Invariants

Every policy must satisfy:

- chronological order is unchanged;
- all aligned fields have equal length;
- all kept indices belong to the causal training prefix;
- each emitted training view has at least one interaction, and empty policy views are skipped;
- minimum sequence-length constraints are respected;
- fixed seeds are reproducible;
- no validation/test interaction contributes to training statistics;
- cache keys change when behavior changes.

### Completed Integration Checks

For each new task/policy:

1. Loader argument resolution and legacy task compatibility.
2. Tiny synthetic train/validation dataset construction.
3. Dataset output sample-schema validation.
4. `python -m compileall main.py SeqRec tests`.
5. Configured `flake8` on modified Python files.
6. CPU dataset construction and sample access.
7. CLI exposure through `train_SMB_decoder --help`.

Executed on ShortVideoAD:

- `session` augmentation with `Qwen3TemporalHierarchicalMultiViewSoft`, using `smb_policy_decoder` with `--sequence_augmentation session`; result path `results/ShortVideoAD/smb_policy_decoder/Qwen3TemporalHierarchicalMultiViewSoft_aug_session/results-smb_explicit-original.json`.

Not yet executed:

- a full collated batch forward pass;
- a one-step GPU smoke test;
- full training and recommendation-metric experiments for all other static policies.

### Experiment Logging (Partly Implemented)

Log at dataset construction:

```text
policy name and normalized configuration
input/output sequence-length distribution
keep rate by behavior level
keep rate by time bucket
number and frequency of views
fraction of unchanged samples
```

Policy configuration, mean input/output length, keep rates by level and time bucket, view counts, and unchanged-view ratio are implemented. Kept-session counts remain in policy metadata but are not yet aggregated in dataset logs.

## Current Conclusion And Next Step

The following foundation is implemented and has passed synthetic verification:

```text
shared policy abstraction
+ unified decoder dataset
+ explicit augmentation arguments
+ training-only default augmentation
```

The recommended experimental order is:

```text
Time-Decayed Dropout
→ Session-Aware Dropout
→ Dataset-Level Proportion control
→ User-Adaptive Ratio
→ Target-Conditioned augmentation
→ semantic Multi-View augmentation
→ Curriculum only if static views are effective
```

The current ShortVideoAD policy results below use the corrected full-sequence policy implementation and are evaluated on the `smb_explicit` test set. The primary comparison baseline is the same `Qwen3TemporalHierarchicalMultiViewSoft` backbone under the original `smb_explicit_decoder_4` task, which uses the original four-view/four-times explicit decoder full-sequence augmentation rather than no augmentation.

Additional test-set references:

- `Original GAMER / Old GAMER SID` preserves the previously provided original GAMER test-set numbers.
- `END4Rec` comes from local `results/ShortVideoAD/smb_dis/END4Rec/result-smb_dis.json`.
- `MBGen` comes from the local MBGen test-set result under `results/ShortVideoAD/smb_explicit/`.

Merged behavior test-set results:

| Model / Policy | HR@5 | HR@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|
| END4Rec (local `smb_dis` baseline) | 0.0958 | 0.1457 | 0.0382 | 0.0466 |
| MBGen | 0.1129 | 0.1696 | 0.0461 | 0.0564 |
| Original GAMER / Old GAMER SID | 0.1443 | 0.2129 | 0.0621 | 0.0753 |
| MultiViewSoft + original 4x augmentation (`smb_explicit_decoder_4`) | 0.1418 | 0.2102 | 0.0609 | 0.0742 |
| Full-sequence policy dataset proportion | 0.1365 | 0.2009 | 0.0580 | 0.0702 |
| Full-sequence policy session | 0.1390 | 0.2064 | 0.0592 | 0.0722 |
| Full-sequence policy multi-view | 0.1421 | 0.2101 | 0.0609 | 0.0740 |
| Full-sequence policy target-conditioned | 0.1373 | 0.2027 | 0.0586 | 0.0711 |
| Full-sequence policy user-adaptive | 0.1379 | 0.2038 | 0.0592 | 0.0718 |

CVR target-behavior test-set results:

| Model / Policy | HR@5 | HR@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|
| END4Rec (local `smb_dis` baseline) | 0.0757 | 0.1207 | 0.0385 | 0.0493 |
| MBGen | 0.0985 | 0.1576 | 0.0491 | 0.0637 |
| Original GAMER / Old GAMER SID | 0.1280 | 0.1944 | 0.0687 | 0.0856 |
| MultiViewSoft + original 4x augmentation (`smb_explicit_decoder_4`) | 0.1274 | 0.1958 | 0.0708 | 0.0885 |
| Full-sequence policy dataset proportion | 0.1249 | 0.1929 | 0.0668 | 0.0838 |
| Full-sequence policy session | 0.1268 | 0.1930 | 0.0676 | 0.0847 |
| Full-sequence policy multi-view | 0.1316 | 0.1935 | 0.0709 | 0.0869 |
| Full-sequence policy target-conditioned | 0.1265 | 0.1917 | 0.0674 | 0.0840 |
| Full-sequence policy user-adaptive | 0.1244 | 0.1880 | 0.0679 | 0.0845 |

Under the corrected full-sequence protocol, the policy variants are clearly above END4Rec and MBGen, but they do not stably beat the original 4x explicit-decoder augmentation baseline. Full-sequence multi-view is the strongest policy variant: it essentially matches the original 4x baseline on merged behavior (`HR@5/N@5` slightly higher or tied, `HR@10/N@10` slightly lower) and improves CVR `HR@5/N@5`, but it still trails the original 4x baseline on CVR `HR@10/N@10`. Dataset-level proportion, session, target-conditioned, and user-adaptive policies are weaker than the original 4x baseline on most reported metrics. Therefore, the current conclusion is that semantic policy augmentation is not yet a replacement for the naive 4x ratio augmentation; multi-view policy is the only promising direction worth refining. The earlier history-only policy runs should be treated only as diagnostic history. Curriculum remains unimplemented and deferred because it crosses the static-cache boundary and requires coordinated Dataset, sampler, Trainer, DDP, and resume behavior.
