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

## Feasibility Summary

| Scheme | Feasibility | Expected cost | Model changes | Recommended status |
| --- | --- | --- | --- | --- |
| Time-Decayed Behavior Dropout | Easy | Low | None | Implement first |
| Session-Aware Dropout | Easy | Low-medium | None | Implement first |
| User-Adaptive Ratio | Moderate | Medium | None | Implement after policy abstraction |
| Dataset-Level Fixed Proportion | Easy-moderate | Medium | None | Implement as control |
| Target-Conditioned Augmentation | Moderate | Medium | None for current target-aware samples | Implement after protocol review |
| Multi-View Sequence Augmentation | Moderate | Medium | None initially | Implement after basic policies |
| Curriculum Augmentation | Difficult under current cache design | High | Trainer/data pipeline integration | Defer |

## Recommended Shared Architecture (Priority Refactor)

### Policy Interface

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
    def generate_views(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> list[AugmentedView]:
        ...
```

`AugmentationContext` should contain only causal information:

```text
uid
mode
target_index
target_behavior
target_level
behavior_level
max_behavior_level
```

Policies should return indices, not copied item arrays. A common helper applies the mask once to every aligned field. This prevents `items`, `behaviors`, `session_ids`, and `times` from becoming misaligned.

### Unified Decoder Dataset

Add:

```text
SeqRec/datasets/session_behavior/augmented_decoder.py
```

Recommended class:

```text
SMBPolicyAugmentedDatasetForDecoder
```

It should inherit from `SMBExplicitDatasetForDecoder`, so the existing decoder-only collator logic continues to recognize it through `isinstance`.

Responsibilities:

1. Construct the causal history and prediction target.
2. Build a deterministic RNG from user, split, and optional view id.
3. Call the selected augmentation policy.
4. Validate returned indices.
5. Assemble the existing sample schema:
   `item`, `inters`, `session_ids`, `extended_session_ids`, `actions`, `time`, and `behavior`.
6. Emit the original full-history view when configured.

The policy should not construct tokenizer strings or token-level metadata.

### Policy Registry

Add a small registry in the same module or in:

```text
SeqRec/datasets/session_behavior/augmentation_registry.py
```

Example names:

```text
none
uniform_level
fixed_ratio
time_decay
session
user_adaptive_ratio
dataset_proportion
target_conditioned
multi_view
```

Do not create one complete Dataset class per strategy.

### Explicit Arguments

Add augmentation fields to `DatasetArgs` rather than encoding all values into `--tasks`:

```text
--sequence_augmentation none
--augmentation_views 1
--augmentation_seed 42
--augmentation_keep_original
--augmentation_eval_mode original
--behavior_keep_ratios 1.0,0.8,0.6
--time_decay_type exponential
--time_decay_tau 48
--recent_session_count 1
```

The exact policy-specific arguments can initially be parsed as a compact JSON/config file if the flat argument list becomes too long:

```text
--sequence_augmentation_config config/augmentation/time_decay.json
```

Recommended transition:

- retain existing task names for backward compatibility;
- introduce one new task, such as `smb_policy_decoder`;
- select the policy and parameters through explicit dataset arguments;
- deprecate new parameter-encoded task names.

This requires passing the new dataset arguments through `TrainSMBDecoder.load_train_data` and `load_SMB_datasets`.

### Cache Identity

Every static policy must expose a stable serializable configuration:

```python
policy.cache_key()
```

The dataset cache name should include:

- policy name;
- a short hash of normalized policy configuration;
- augmentation seed;
- number of views;
- whether the original view is included;
- split and index suffix.

This avoids stale cache reuse when a policy parameter changes.

### Training And Evaluation Protocol

Default protocol:

```text
train: augmented history
valid: original history
test: original history
```

This isolates augmentation as a training intervention.

Separate robustness protocol:

```text
train: augmented or original history
valid/test: explicitly corrupted history
```

The current fixed-ratio behavior across train, validation, and test should remain available as a named robustness experiment, not as the default comparison.

## Time-Decayed Behavior Dropout (Easy, Highest Priority)

### Reuse

Directly reuse:

- `times`;
- `behavior_level`;
- target-behavior preservation;
- deterministic RNG;
- existing full-history plus augmented-view behavior.

No model, collator, tokenizer, or sample-schema changes are required.

### Policy

Add:

```text
TimeDecayDropoutPolicy
```

For each historical interaction:

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

Thus recent interactions have near-zero age weight, old low-level actions are most likely to be removed, and target-level history is protected or assigned a very small weight.

### Required Safeguards

- Always preserve the prediction target.
- Preserve at least `min_history_items`.
- Preserve the most recent `min_recent_items`.
- Optionally preserve all historical target-level interactions.
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
- target and required recent items are preserved;
- zero-span timestamps do not fail.

## Session-Aware Dropout (Easy, Highest Priority)

### Reuse

The current data already provides normalized session ids and session boundaries. No preprocessing change is required.

### Policy

Add:

```text
SessionAwareDropoutPolicy
```

First group historical indices by session. Compute session-level features using only that session:

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

## User-Adaptive Ratio (Moderate, Medium-High Priority)

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

## Dataset-Level Fixed Behavior Proportion (Easy-Moderate, Control Baseline)

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

## Target-Conditioned Augmentation (Moderate, Protocol Review Required)

### Feasibility

For decoder training, the target behavior is already known while constructing each sample, so no model change is needed. The policy can use:

```text
context.target_level
```

However, the current `SMBExplicitDatasetForDecoder` creates one training target per user at the split boundary. This gives limited target-level diversity compared with datasets that create a target at every position.

### Policy

Add:

```text
TargetConditionedPolicy
```

Recommended implementation is a wrapper around another base policy:

```text
TargetConditionedPolicy(base_policy=time_decay)
```

It modifies level-preservation weights according to the target:

```text
same-level evidence
one-level-below evidence
upward-path evidence
general temporal evidence
```

For a high-level target, retain recent lower-level precursors. For a shallow target, favor same-level and recent temporal evidence.

### Leakage Constraint

Using target behavior is valid only if the same behavior prompt or target type is known at inference/evaluation. If the production task requires predicting the behavior type itself, this augmentation would leak target information.

The first experiment should therefore stay within behavior-specific next-item prediction, where the target behavior is part of the task definition.

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

## Multi-View Sequence Augmentation (Moderate, Medium-High Priority)

### Feasibility

The current decoder dataset already emits multiple augmented sequences per user. Replace ratio-indexed views with named semantic policies.

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

1. `full`: unchanged causal history.
2. `recent`: time-decayed or fixed recent window.
3. `hierarchy`: preserve target-level, same-level, and one-level-below evidence.
4. `session`: session-aware subsample.

Do not enable every possible view initially.

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

- each configured view has the declared semantics;
- the full view is unchanged;
- no duplicate view is emitted when two policies produce the same keep indices;
- maximum views per user is enforced;
- dataset-size growth is predictable;
- view ordering and seeds are deterministic.

## Curriculum Augmentation (Difficult, Deferred)

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

## Recommended File-Level Plan

### Stage 1: Shared Static Policy Layer (Implement First)

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

Provide compatibility adapters for the current uniform-level and fixed-ratio behavior.

### Stage 2: First Policies

Implement and test:

1. `TimeDecayDropoutPolicy`.
2. `SessionAwareDropoutPolicy`.
3. `DatasetProportionPolicy`.

These have clear semantics and low leakage risk.

### Stage 3: Adaptive And Composed Policies

Implement:

1. training-only behavior statistics;
2. `UserAdaptiveRatioPolicy`;
3. `TargetConditionedPolicy`;
4. `MultiViewAugmentationPolicy`.

### Stage 4: Online Sampling And Curriculum

Only proceed if static MultiView is useful and cached dataset expansion becomes a practical bottleneck.

## Verification Plan

### Unit-Level Tests

Create:

```text
tests/datasets/session_behavior/test_augmentation_policies.py
tests/datasets/session_behavior/test_augmented_decoder.py
tests/datasets/loaders/test_session_behavior_augmentation.py
```

Use synthetic users covering:

- one and multiple sessions;
- equal and irregular timestamps;
- missing target behavior;
- all interactions at one level;
- very short histories;
- multiple high-level behaviors;
- split prefixes sharing historical sessions.

### Invariants

Every policy must satisfy:

- chronological order is unchanged;
- all aligned fields have equal length;
- all kept indices belong to the causal history;
- prediction target is never inserted into history;
- minimum history constraints are respected;
- fixed seeds are reproducible;
- no validation/test interaction contributes to training statistics;
- cache keys change when behavior changes.

### Integration Checks

For each new task/policy:

1. Resolve loader arguments without constructing a full real dataset.
2. Build a tiny synthetic train/valid/test dataset.
3. Verify the collator receives the same sample schema.
4. Run `python -m compileall main.py SeqRec`.
5. Run configured `flake8` on modified Python files.
6. Run a short CPU DataLoader iteration.
7. If available, run a one-step GPU smoke test without launching full training.

### Experiment Logging

Log at dataset construction:

```text
policy name and normalized configuration
input/output sequence-length distribution
keep rate by behavior level
keep rate by time bucket
kept sessions per user
number and frequency of views
fraction of unchanged samples
```

These statistics are necessary to interpret recommendation results and to detect accidental differences in training volume.

## Final Recommendation

The current framework can support most proposed sequence augmentation without changing the TH model. The best implementation path is:

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

Time-decayed and session-aware strategies are the easiest and most aligned with the current TH framing. User-adaptive and target-conditioned methods are feasible but require stronger leakage and shortcut controls. Curriculum augmentation should remain deferred because it crosses the current static-cache boundary and requires coordinated Dataset, sampler, Trainer, DDP, and resume behavior.
