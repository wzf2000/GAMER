# Temporal-Hierarchical Method Status, Decisions, And Paper Plan

## Purpose

This document is intended for internal synchronization and subsequent paper writing. It summarizes:

- implemented and evaluated Temporal-Hierarchical (TH) variants,
- implemented variants still awaiting experiments,
- directions expected to be dropped or retained only as ablations,
- candidate auxiliary objectives,
- sequence-augmentation designs aligned with the current TH model.

The method target remains:

```text
Generative Temporal-Hierarchical Behavioral Modeling for Recommendation
```

The central view is that multi-behavior interaction histories have two dimensions: temporal order and behavior hierarchy. A causal model should retain ordinary next-item prediction, within-level next-item prediction, and cross-level action/conversion prediction as compatible special cases.

## Unified Modeling Framework

The main setting keeps the deepest observed behavior for each item and interprets it as a compact behavior-chain state. Complete behavior-chain flattening is not the default input.

The model uses replacement-style TH attention:

```text
Layer 0-1: standard causal attention
Layer 2-5: temporal-hierarchical attention
Layer 6-7: standard causal attention / generation refinement
```

All TH variants share:

- behavior-level Q/K/V embeddings,
- attention-output gating,
- causal and item-local semantic-token masks,
- behavior-aware MoE/FFN.

Consequently, the zero-relation-bias model is still substantially different from ordinary Qwen3.

## Current Method Status (Living Summary)

| Category | Variant | Status | Current role |
|---|---|---|---|
| TH Base | FixedBias / FixedZero | Implemented and evaluated | Current best result; effectively TH without scalar relation bias |
| Relation Bias | Factorized | Implemented and evaluated | Learnable relation-bias main-line candidate |
| Relation Bias | FixedSoft | Implemented, pending result | Fixed weak hierarchy prior |
| Relation Bias | FactorizedSoft | Implemented, pending result | Learnable bias initialized from weak prior |
| Relation Bias | FactorizedScale | Implemented, pending result | Fixed scale controlling relation-bias strength |
| Relation Bias | FactorizedAlpha | Implemented, pending result | Per-layer learnable alpha |
| Relation Bias | Naive trainable table | Profiled | Expected to be dropped because of backward cost |
| Multi-View | Hard MultiView | Implemented and evaluated | Effective structured ablation, weaker than relation-bias family |
| Multi-View | Soft MultiView | Implemented, pending result | Finite soft penalties instead of hard masks |
| Multi-View | Gated MultiView | Implemented, pending result | Learnable per-head view mixture |
| Objective | Behavior-level auxiliary objectives | Not implemented | Future TH supervision |
| Data | Existing ratio augmentation | Implemented and used | User-independent and time-independent; needs redesign |

## Relation-Bias Family (Main-Line Candidate, Partly Pending)

Relation-bias variants use:

```text
score(i, j)
  = q_i k_j / sqrt(d)
  + causal_mask(i, j)
  + alpha * relation_bias(level_i, level_j)
```

They keep all causal history visible and softly control the importance of behavior-level relations.

### TH Base / FixedZero (Implemented and Evaluated)

The current FixedBias config has a zero, frozen scalar table. Its gain comes from replacement TH attention, behavior Q/K/V embeddings, gating, and behavior-aware MoE.

Recommended terminology:

```text
TH Base
TH Attention w/o Relation Bias
TH Embedding-only
```

It should not be presented as evidence that a nonzero fixed relation prior is effective.

### Factorized Relation Bias (Implemented and Evaluated)

Factorized bias learns query/key level factors and computes their low-rank dot product. It provides learnable pairwise hierarchy modeling with practical backward cost.

It is the strongest current method-story candidate, although its zero-init result is slightly below TH Base.

### Soft Prior, Fixed Scale, And Learnable Alpha (Implemented, Pending Evaluation, High Priority)

- FixedSoft and FactorizedSoft initialize a weak shallow-to-deep hierarchy prior with scale `0.05`.
- FactorizedScale multiplies relation bias by a fixed `0.1`.
- FactorizedAlpha learns a per-layer scalar initialized to `0.1`.

These variants test whether the original Factorized result was hurt by an overly strong or poorly initialized logit perturbation.

Learned alpha values should be recorded per layer for interpretation.

### Naive Trainable Table (Profiled, Expected To Be Dropped)

The naive full trainable table produced approximately `12918 ms/step` versus roughly `121 ms/step` for Factorized under the profiling setup. Advanced-index backward scatters a dense `[B,H,L,L]` gradient into a small table.

Decision:

- drop it as a practical model direction,
- retain it only as engineering evidence motivating the factorized implementation.

## Multi-View Family (Partly Evaluated, Structured Comparison)

### Hard MultiView (Implemented and Evaluated, Likely An Ablation)

Hard MultiView assigns heads to temporal, same-level, upward, and downward views. It is interpretable and efficient, but hard visibility constraints reduce flexibility. It improves over old GAMER but is weaker than TH Base and Factorized.

It should remain an important structured ablation, not the current default main model.

### Soft MultiView (Implemented, Pending Evaluation)

Soft MultiView replaces negative-infinity blocks with finite negative penalties and uses a uniform mixture over views. It tests whether hard masking explains MultiView's weaker performance.

The soft-bias scale requires later ablation.

### Gated MultiView (Implemented, Pending Evaluation, Preferred Over Dynamic Gating)

Gated MultiView learns a softmax view mixture for each head. Gates are initialized from the previous hard head allocation.

It preserves interpretability while allowing heads to revise their assigned view. The current gate is static per head; query-, level-, or user-conditioned dynamic gates should only be considered if this static gated variant is clearly beneficial.

## Auxiliary Objectives (Not Implemented, Future Focus)

Auxiliary objectives are not yet implemented. Next-token generation should remain primary.

### Next Behavior-Level Prediction (High Priority)

```text
L = L_next_token + lambda_level * L_next_level
```

Predict the next behavior level at behavior-token or item-level positions. This uses existing labels and directly strengthens hierarchy awareness.

Recommended as the first auxiliary objective.

### Behavior Transition-Type Prediction (Future Consideration)

Predict sampled relations:

```text
same / up / down / temporal-mixed
```

This can supervise Gated MultiView or regularize Factorized relations. Dense pair classification should be avoided; sample item/behavior-token pairs.

### Upward Progression / Conversion Objective (Future Consideration)

Given a shallow interaction, predict whether the same item reaches a deeper level in a later observation window.

This directly matches conversion modeling but requires careful window and censoring definitions. It should be considered after next-level prediction.

### Relation Regularization (Optional, Dependent On Main-Model Results)

Potential regularizers include distance to a soft prior, layer consistency, low-rank penalties, or monotonic constraints. These should only be added after learned relation matrices are inspected.

## Existing Sequence Augmentation (Implemented, Redesign Needed)

The current explicit decoder augmentation gives every user the same downsampling schedule. It randomly drops non-target behaviors with:

```text
behavior_drop_ratio = augmentation_ratio / (level + 1)
```

The fixed-ratio dataset caps each user's lower-level counts relative to the target-level count with a shared ratio such as `[5,1,1]`.

Current limitations:

- one schedule or ratio is shared by all users,
- random dropping ignores interaction age and session boundaries,
- recent high-value shallow evidence may be removed,
- global ratios can erase genuine user funnel differences,
- target-count normalization is undefined when a user has no target behavior.

## Candidate Sequence-Augmentation Designs (Not Implemented)

These are design candidates only and are not implemented yet.

### 1. Time-Decayed Behavior Dropout (High Priority)

Use behavior level and temporal distance:

```text
p_drop(i)
  = base_ratio(level_i)
  * time_decay(delta_t_i)
```

Older interactions are easier to drop; recent and high-level evidence is preserved. Candidate decay functions include rank decay, exponential decay, and recent/mid/old buckets.

Priority: high.

### 2. Session-Aware Dropout (High Priority)

Sample or drop complete historical sessions rather than independent interactions:

- keep the latest session,
- retain sessions containing deep behaviors with higher probability,
- drop old shallow-only sessions more aggressively,
- preserve within-session ordering.

This aligns strongly with the current session IDs and TH temporal dimension.

Priority: high.

### 3. User-Adaptive Ratio (Medium-High Priority)

Estimate a smoothed per-user funnel ratio and shrink it toward a global prior:

```text
r'_u,l = beta_u * r_u,l + (1 - beta_u) * r_global,l
```

Active users receive more personalized ratios; sparse users rely more on global statistics. Ratios need clipping, and all statistics must use only past data to avoid leakage.

Priority: medium-high.

### 4. Dataset-Level Fixed Behavior Proportion (Medium Priority, Control Baseline)

Estimate a dataset-level target proportion and softly cap per-level counts. The target may preserve the natural distribution or intentionally rebalance toward deeper behaviors.

This is more stable than per-user target-count normalization, but remains user-independent and may create train/test distribution shift.

Priority: medium.

### 5. Target-Conditioned Augmentation (Medium-High Priority)

Choose history retention according to the current target behavior:

- conversion targets retain click/activation and recent shallow evidence,
- click targets retain same-level and general temporal history,
- merged tasks use target-level-specific retention.

This is tightly aligned with TH queries but risks allowing the input distribution to reveal the target.

Priority: medium-high after time/session-aware designs.

### 6. Multi-View Sequence Augmentation (Medium-High Priority, Future Consideration)

Generate a small number of semantically defined views:

```text
full temporal view
recent-window view
same-level-preserving view
upward-evidence-preserving view
session-subsampled view
```

These views correspond to ordinary temporal modeling, per-level next-item prediction, and action/conversion prediction. They provide a stronger TH narrative than a generic ratio schedule.

Priority: medium-high.

### 7. Curriculum Augmentation (Low Priority, Longer Term)

Begin with full or weakly dropped history, then introduce time-aware/session-aware or sparse-history views later in training.

This may improve robustness but complicates caching, resuming, and early stopping. It should follow static augmentation experiments.

## Directions To Drop Or Downgrade (Current Decisions)

- Drop naive trainable relation table because Factorized provides practical learnable relation modeling.
- Retain Hard MultiView as an ablation unless soft/gated variants materially improve it.
- Keep fully flattened behavior-event input as an input ablation, not the default main representation.
- Delay dynamic context-aware MultiView gating until static Gated MultiView is validated.
- Keep the uniform random ratio schedule only as an augmentation baseline, not the final TH-aware design.

## Recommended Execution Order (Action Plan)

### Stage 1: Finish Implemented Model Experiments (Current Priority)

1. FixedSoft.
2. FactorizedSoft.
3. FactorizedScale.
4. FactorizedAlpha.
5. MultiViewSoft.
6. MultiViewGated.

Record conversion, merged, per-behavior metrics, training time, memory, learned alpha, and learned view gates.

### Stage 2: Select The Final Model Line (After Pending Results)

- If FactorizedAlpha or FactorizedSoft reaches or exceeds TH Base, use it as the main model.
- If all relation-bias extensions remain below TH Base, define the main contribution as behavior-aware replacement TH attention rather than scalar relation bias.
- If Gated MultiView improves substantially, consider combining it with Factorized only after checking complexity and attribution.

### Stage 3: Implement The First Auxiliary Objective (Next Development Round)

Implement next behavior-level prediction first, with initial weights `0.05` and `0.1`.

### Stage 4: Redesign Sequence Augmentation (Future Focus)

Recommended first implementations:

1. Time-Decayed Behavior Dropout.
2. Session-Aware Dropout.
3. User-Adaptive Ratio.

Then consider target-conditioned and multi-view sequence augmentation.

## Paper-Writing Guidance

The method can be organized into:

1. compact behavior-chain-state sequence,
2. replacement-style Temporal-Hierarchical Attention,
3. TH-aware training enhancement through relation control, auxiliary objectives, or sequence augmentation.

Recommended names:

- `TH Base`: behavior-aware Q/K/V and gating without scalar relation bias.
- `TH-FRB`: Factorized Relation Bias.
- `TH-FRB-Soft` or `TH-FRB-Alpha`: final relation-bias candidate.
- `TH-MV`: Hard MultiView.
- `TH-MV-Gated`: Gated MultiView.

Current evidence supports:

- replacement TH attention improves over old added-cross-level GAMER,
- behavior-aware Q/K/V helps both conversion and merged behavior,
- soft relation modeling currently outperforms hard MultiView.

Claims still requiring experiments:

- learnable relation bias always outperforms TH Base,
- soft hierarchy prior is beneficial,
- gated MultiView outperforms hard MultiView,
- TH-aware auxiliary objectives and augmentation add further gains.

## Current Conclusion

The model-side main line should remain:

```text
TH Base
+ controlled learnable Factorized Relation Bias
```

FactorizedAlpha and FactorizedSoft are the highest-priority final-model candidates.

The data-side design should move beyond a user-independent random ratio schedule toward:

```text
time-aware
+ session-aware
+ behavior-level-aware
+ optionally user-adaptive
```

The model, objectives, and augmentation should ultimately express the same temporal-hierarchical view instead of adding unrelated complexity.
