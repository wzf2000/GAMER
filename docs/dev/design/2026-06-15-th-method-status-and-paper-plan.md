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
| Relation Bias | FixedSoft | Implemented and evaluated | Fixed weak hierarchy prior; no stable gain over TH Base |
| Relation Bias | FactorizedSoft | Implemented and evaluated | Learnable weak-prior initialization; does not outperform Factorized/FixedBias |
| Relation Bias | FactorizedScale | Implemented and evaluated | Best current relation-bias extension on several CVR coverage metrics |
| Relation Bias | FactorizedAlpha | Implemented, pending result | Per-layer learnable alpha |
| Relation Bias | Naive trainable table | Profiled | Expected to be dropped because of backward cost |
| Multi-View | Hard MultiView | Implemented and evaluated | Clearly weaker than TH Base/relation-bias variants; structured ablation |
| Multi-View | Soft MultiView | Implemented and evaluated | Improves Hard MultiView on merged metrics but remains weaker on CVR |
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

### Soft Prior, Fixed Scale, And Learnable Alpha (Partly Evaluated, High Priority)

- FixedSoft and FactorizedSoft initialize a weak shallow-to-deep hierarchy prior with scale `0.05`.
- FactorizedScale multiplies relation bias by a fixed `0.1`.
- FactorizedAlpha learns a per-layer scalar initialized to `0.1`.

The latest results show that soft prior initialization is not consistently helpful. FactorizedScale is more promising: it improves CVR `HR@5/R@5/N@5` over TH Base, although it still loses on `HR@1/R@1/HR@10/N@10`. This suggests that relation-bias strength control is useful, but not yet sufficient for a stable main-model win.

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

### Soft MultiView (Implemented and Evaluated)

Soft MultiView replaces negative-infinity blocks with finite negative penalties and uses a uniform mixture over views. It tests whether hard masking explains MultiView's weaker performance.

The latest result confirms that soft penalties improve over Hard MultiView on merged behavior, where Soft MultiView is best on `HR@5/N@5`. However, it remains weaker than TH Base on the CVR target behavior. The soft-bias scale may still deserve ablation, but Soft MultiView is currently a structured ablation rather than the final model line.

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

1. FactorizedAlpha.
2. MultiViewGated.
3. Factorized rank ablation under the scaled setting.
4. Optional Soft MultiView scale ablation.

Record conversion, merged, per-behavior metrics, training time, memory, learned alpha, and learned view gates.

### Stage 2: Select The Final Model Line (After Pending Results)

- If FactorizedAlpha reaches or exceeds TH Base, use it as the relation-bias main model.
- If relation-bias extensions remain mixed or below TH Base, define the main contribution as behavior-aware replacement TH attention rather than scalar relation bias.
- If Gated MultiView improves substantially, consider combining it with Factorized only after checking complexity and attribution.

### Stage 3: Implement The First Auxiliary Objective (Next Development Round)

Implement next behavior-level prediction first, with initial weights `0.05` and `0.1`.

### Stage 4: Redesign Sequence Augmentation (Future Focus)

Recommended first implementations:

1. Time-Decayed Behavior Dropout.
2. Session-Aware Dropout.
3. User-Adaptive Ratio.

Then consider target-conditioned and multi-view sequence augmentation.

## Completed ShortVideoAD TH Variant Results

Result path:

```text
results/ShortVideoAD/smb_explicit_decoder_4/
```

The evaluation task is `smb_explicit_valid` behavior-specific next-item prediction. The tables below report test-set merged behavior and conversion/cvr behavior.

### Merged Behavior

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TH-FixedBias | 0.0490 | 0.1502 | 0.2227 | 0.0237 | 0.0778 | 0.1220 | 0.0656 | 0.0799 |
| TH-FixedSoft | 0.0492 | 0.1497 | 0.2216 | 0.0236 | 0.0780 | 0.1221 | 0.0652 | 0.0795 |
| TH-Factorized | 0.0487 | 0.1500 | 0.2220 | 0.0234 | 0.0781 | 0.1215 | 0.0655 | 0.0796 |
| TH-FactorizedScale | 0.0482 | 0.1502 | 0.2221 | 0.0234 | 0.0779 | 0.1216 | 0.0654 | 0.0795 |
| TH-FactorizedSoft | 0.0486 | 0.1494 | 0.2199 | 0.0229 | 0.0773 | 0.1203 | 0.0649 | 0.0787 |
| TH-MultiView | 0.0463 | 0.1478 | 0.2162 | 0.0219 | 0.0760 | 0.1179 | 0.0632 | 0.0766 |
| TH-MultiViewSoft | 0.0496 | 0.1508 | 0.2218 | 0.0239 | 0.0780 | 0.1212 | 0.0657 | 0.0796 |

TH-MultiViewSoft is best on merged `HR@1/HR@5/R@1/N@5`, while TH-FixedBias remains best on `HR@10/N@10` and TH-FixedSoft is narrowly best on `R@10`. The differences among the top variants are small, but the result changes the earlier interpretation of MultiView: hard partitioning is weak, while soft view penalties are viable for merged behavior.

The overall merged ranking is now:

```text
FixedBias ~= MultiViewSoft ~= FactorizedScale ~= Factorized > FixedSoft > FactorizedSoft > Hard MultiView
```

Relative to TH-FixedBias:

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TH-Factorized | -0.64% | -0.12% | -0.31% | -1.34% | +0.40% | -0.41% | -0.08% | -0.34% |
| TH-FactorizedScale | -1.65% | +0.02% | -0.29% | -1.47% | +0.13% | -0.38% | -0.17% | -0.50% |
| TH-FactorizedSoft | -0.99% | -0.55% | -1.26% | -3.37% | -0.62% | -1.44% | -1.08% | -1.50% |
| TH-FixedSoft | +0.29% | -0.33% | -0.50% | -0.67% | +0.35% | +0.03% | -0.48% | -0.46% |
| TH-MultiView | -5.56% | -1.56% | -2.93% | -7.89% | -2.30% | -3.43% | -3.56% | -4.07% |
| TH-MultiViewSoft | +1.24% | +0.39% | -0.40% | +0.92% | +0.34% | -0.72% | +0.26% | -0.32% |

### Conversion / CVR Behavior

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TH-FixedBias | 0.0438 | 0.1368 | 0.2068 | 0.0348 | 0.1052 | 0.1597 | 0.0756 | 0.0936 |
| TH-FactorizedScale | 0.0428 | 0.1371 | 0.2046 | 0.0340 | 0.1062 | 0.1586 | 0.0758 | 0.0932 |
| TH-FactorizedSoft | 0.0427 | 0.1358 | 0.2045 | 0.0324 | 0.1054 | 0.1600 | 0.0748 | 0.0926 |
| TH-Factorized | 0.0419 | 0.1354 | 0.2044 | 0.0331 | 0.1052 | 0.1588 | 0.0747 | 0.0924 |
| TH-FixedSoft | 0.0405 | 0.1338 | 0.2048 | 0.0320 | 0.1044 | 0.1588 | 0.0735 | 0.0916 |
| TH-MultiView | 0.0394 | 0.1345 | 0.2018 | 0.0309 | 0.1028 | 0.1556 | 0.0723 | 0.0898 |
| TH-MultiViewSoft | 0.0417 | 0.1354 | 0.2038 | 0.0328 | 0.1036 | 0.1577 | 0.0739 | 0.0918 |

TH-FixedBias is still the most stable CVR model. FactorizedScale is slightly better on `HR@5/R@5/N@5`, and FactorizedSoft is slightly better on `R@10`, but both are weaker on top-rank quality and/or `N@10`. This suggests that controlled relation bias can improve candidate coverage but has not yet improved the primary ordering quality.

### Interpretation

1. TH-FixedBias gains are not from a scalar relation-bias value. The current FixedBias table is frozen zero, so the gain is better attributed to replacement-style TH attention, behavior Q/K/V, gating, and behavior-aware MoE.
2. Factorized and FactorizedScale are close to FixedBias, so learnable relation bias remains a feasible extension, but it does not yet prove a stable advantage over TH Base.
3. Soft prior is not consistently helpful. FixedSoft and FactorizedSoft do not outperform FixedBias; scale control is more promising than soft-prior initialization.
4. Hard MultiView is clearly weaker, but Soft MultiView recovers much of the loss and is strong on merged behavior. This supports soft view constraints over hard head partitioning.

### Method-Line Impact

The safer paper main line is now:

```text
TH Base / FixedBias
```

Learnable Factorized Relation Bias should be framed as an extension/ablation unless a later Alpha result clearly exceeds FixedBias. FactorizedScale is the strongest current relation-bias extension, but its gain is metric-specific.

Hard MultiView should be downgraded to a structured ablation. Soft MultiView should remain as a stronger structured-view comparison because it improves merged behavior without hard visibility cuts.

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
- controlled relation bias can improve coverage-oriented metrics, but not yet the full ranking profile.
- soft MultiView improves hard MultiView and is competitive on merged behavior.

Claims still requiring experiments:

- learnable relation bias always outperforms TH Base,
- soft hierarchy prior is beneficial,
- controlled relation bias or gated view mixture can stably beat TH Base,
- gated MultiView outperforms hard MultiView,
- TH-aware auxiliary objectives and augmentation add further gains.

## Current Conclusion

The model-side main line should remain:

```text
TH Base
+ controlled learnable Factorized Relation Bias
```

The latest ShortVideoAD results show that `FixedSoft`, `Factorized`, `FactorizedScale`, `FactorizedSoft`, `MultiView`, and `MultiViewSoft` do not stably outperform TH-FixedBias on the CVR target behavior. The final model should therefore prioritize `TH Base / FixedBias`. `FactorizedScale` and `MultiViewSoft` are worth retaining as stronger extensions than their unscaled/hard counterparts, while `FactorizedAlpha` and `Gated MultiView` remain the most informative pending model-side experiments.

The data-side design should move beyond a user-independent random ratio schedule toward:

```text
time-aware
+ session-aware
+ behavior-level-aware
+ optionally user-adaptive
```

The model, objectives, and augmentation should ultimately express the same temporal-hierarchical view instead of adding unrelated complexity.
