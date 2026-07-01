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
| TH Base | FixedBias / FixedZero | Implemented and evaluated | Clean TH base; effectively TH without scalar relation bias |
| Relation Bias | Factorized | Implemented and evaluated | Learnable relation-bias main-line candidate |
| Relation Bias | FixedSoft | Implemented and evaluated | Strongest new TH variant on merged behavior and most top-rank CVR metrics |
| Relation Bias | FactorizedSoft | Implemented and evaluated | Learnable weak-prior initialization; does not outperform Factorized/FixedSoft |
| Relation Bias | FactorizedScale | Implemented and evaluated | Best current relation-bias extension on several CVR coverage metrics |
| Relation Bias | FactorizedAlpha | Implemented, pending result | Per-layer learnable alpha |
| Relation Bias | Naive trainable table | Profiled | Expected to be dropped because of backward cost |
| Multi-View | Hard MultiView | Implemented and evaluated | Clearly weaker than TH Base/relation-bias variants; structured ablation |
| Multi-View | Soft MultiView | Implemented and evaluated | Improves Hard MultiView but remains weaker than FixedSoft/Factorized |
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

The latest test-set results show that soft prior initialization is helpful for the fixed/frozen relation family: FixedSoft is strongest among new TH variants on merged behavior and most top-rank CVR metrics. The same conclusion does not hold for the factorized family, where FactorizedSoft is weaker than Factorized. This suggests that prior direction, prior strength, and parameterization must be tuned together.

Learned alpha values should be recorded per layer for interpretation.

### Naive Trainable Table (Profiled, Expected To Be Dropped)

The naive full trainable table produced approximately `12918 ms/step` versus roughly `121 ms/step` for Factorized under the profiling setup. Advanced-index backward scatters a dense `[B,H,L,L]` gradient into a small table.

Decision:

- drop it as a practical model direction,
- retain it only as engineering evidence motivating the factorized implementation.

## Multi-View Family (Partly Evaluated, Structured Comparison)

### Hard MultiView (Implemented and Evaluated, Likely An Ablation)

Hard MultiView assigns heads to temporal, same-level, upward, and downward views. It is interpretable and efficient, but hard visibility constraints reduce flexibility. Compared with Original GAMER, it is not consistently better, and it is weaker than the main FixedSoft/Factorized TH candidates.

It should remain an important structured ablation, not the current default main model.

### Soft MultiView (Implemented and Evaluated)

Soft MultiView replaces negative-infinity blocks with finite negative penalties and uses a uniform mixture over views. It tests whether hard masking explains MultiView's weaker performance.

The latest result confirms that soft penalties improve over Hard MultiView, especially on CVR top-rank behavior. However, Soft MultiView remains weaker than FixedSoft on merged behavior and weaker than Factorized on deeper CVR metrics. The soft-bias scale may still deserve ablation, but Soft MultiView is currently a structured ablation rather than the final model line.

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

The evaluation task is `smb_explicit` behavior-specific next-item prediction on the test set. The tables below report test-set merged behavior and conversion/cvr behavior. MBGen is included as the main published baseline, and Original GAMER / Old GAMER SID is included as the previous method reference.

### Main Baseline Comparison

Merged behavior is reported with the four shared metrics available for the major baselines:

| Model | HR@5 | HR@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: |
| MBGen (SID) | 0.1179 | 0.1774 | 0.0493 | 0.0603 |
| Original GAMER / Old GAMER SID | 0.1443 | 0.2129 | 0.0621 | 0.0753 |
| TH-FixedBias | 0.1444 | 0.2116 | 0.0620 | 0.0750 |
| TH-FixedSoft | 0.1450 | 0.2121 | 0.0628 | 0.0756 |
| TH-Factorized | 0.1430 | 0.2117 | 0.0614 | 0.0746 |
| TH-FactorizedScale | 0.1432 | 0.2113 | 0.0614 | 0.0745 |
| TH-FactorizedSoft | 0.1434 | 0.2099 | 0.0615 | 0.0744 |
| TH-MultiView | 0.1391 | 0.2062 | 0.0595 | 0.0723 |
| TH-MultiViewSoft | 0.1418 | 0.2102 | 0.0609 | 0.0742 |

On merged behavior, all TH variants remain clearly above MBGen. Compared with Original GAMER, the result is more nuanced: TH-FixedSoft is the strongest new variant and slightly improves `HR@5/N@5/N@10`, but its `HR@10` is still slightly lower than Original GAMER. Therefore, the merged-behavior claim should be framed as competitive or slightly improved in top ranking, not as a broad win over the original GAMER line.

CVR target-behavior comparison:

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MBGen (SID) | 0.0276 | 0.1012 | 0.1622 | 0.0202 | 0.0736 | 0.1205 | 0.0518 | 0.0673 |
| Original GAMER / Old GAMER SID | 0.0394 | 0.1280 | 0.1944 | 0.0292 | 0.0966 | 0.1478 | 0.0687 | 0.0856 |
| TH-FixedBias | 0.0390 | 0.1283 | 0.1974 | 0.0290 | 0.0963 | 0.1507 | 0.0693 | 0.0873 |
| TH-FixedSoft | 0.0435 | 0.1349 | 0.1981 | 0.0326 | 0.1007 | 0.1513 | 0.0735 | 0.0900 |
| TH-Factorized | 0.0409 | 0.1342 | 0.2042 | 0.0302 | 0.1011 | 0.1565 | 0.0721 | 0.0902 |
| TH-FactorizedScale | 0.0393 | 0.1331 | 0.1987 | 0.0301 | 0.0988 | 0.1514 | 0.0706 | 0.0877 |
| TH-FactorizedSoft | 0.0385 | 0.1274 | 0.1947 | 0.0294 | 0.0972 | 0.1503 | 0.0690 | 0.0867 |
| TH-MultiView | 0.0381 | 0.1283 | 0.1949 | 0.0275 | 0.0958 | 0.1461 | 0.0678 | 0.0845 |
| TH-MultiViewSoft | 0.0427 | 0.1274 | 0.1958 | 0.0331 | 0.0966 | 0.1504 | 0.0708 | 0.0885 |

On the CVR target behavior, the new TH variants provide a clearer improvement over both MBGen and Original GAMER. TH-FixedSoft is best on most top-rank CVR metrics, while TH-Factorized is best on deeper ranking and coverage metrics. This is the strongest evidence that the Temporal-Hierarchical redesign improves the target behavior rather than only changing the merged-behavior tradeoff.

### TH Variant Details: Merged Behavior

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TH-FixedBias | 0.0465 | 0.1444 | 0.2116 | 0.0223 | 0.0731 | 0.1135 | 0.0620 | 0.0750 |
| TH-FixedSoft | 0.0476 | 0.1450 | 0.2121 | 0.0225 | 0.0743 | 0.1142 | 0.0628 | 0.0756 |
| TH-Factorized | 0.0465 | 0.1430 | 0.2117 | 0.0216 | 0.0725 | 0.1131 | 0.0614 | 0.0746 |
| TH-FactorizedScale | 0.0454 | 0.1432 | 0.2113 | 0.0218 | 0.0726 | 0.1133 | 0.0614 | 0.0745 |
| TH-FactorizedSoft | 0.0460 | 0.1434 | 0.2099 | 0.0218 | 0.0729 | 0.1126 | 0.0615 | 0.0744 |
| TH-MultiView | 0.0439 | 0.1391 | 0.2062 | 0.0210 | 0.0709 | 0.1105 | 0.0595 | 0.0723 |
| TH-MultiViewSoft | 0.0460 | 0.1418 | 0.2102 | 0.0220 | 0.0718 | 0.1130 | 0.0609 | 0.0742 |

Within the new TH variants, TH-FixedSoft is the strongest merged-behavior variant across the reported test metrics. TH-FixedBias remains a close and stable TH base, while hard MultiView is clearly weaker and MultiViewSoft only partially recovers the hard partition loss.

The overall merged ranking is now:

```text
FixedSoft > FixedBias ~= Factorized > FactorizedSoft ~= FactorizedScale > MultiViewSoft > Hard MultiView
```

Relative to TH-FixedBias:

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TH-FixedSoft | +2.53% | +0.42% | +0.26% | +1.31% | +1.54% | +0.62% | +1.19% | +0.79% |
| TH-Factorized | +0.07% | -0.94% | +0.06% | -2.77% | -0.85% | -0.36% | -0.98% | -0.63% |
| TH-FactorizedScale | -2.33% | -0.81% | -0.13% | -1.83% | -0.73% | -0.24% | -1.03% | -0.77% |
| TH-FactorizedSoft | -0.94% | -0.67% | -0.80% | -2.23% | -0.32% | -0.81% | -0.82% | -0.90% |
| TH-MultiView | -5.56% | -3.62% | -2.55% | -5.46% | -3.12% | -2.70% | -4.09% | -3.59% |
| TH-MultiViewSoft | -1.11% | -1.80% | -0.63% | -0.99% | -1.80% | -0.50% | -1.80% | -1.11% |

### TH Variant Details: Conversion / CVR Behavior

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TH-FixedBias | 0.0390 | 0.1283 | 0.1974 | 0.0290 | 0.0963 | 0.1507 | 0.0693 | 0.0873 |
| TH-FixedSoft | 0.0435 | 0.1349 | 0.1981 | 0.0326 | 0.1007 | 0.1513 | 0.0735 | 0.0900 |
| TH-Factorized | 0.0409 | 0.1342 | 0.2042 | 0.0302 | 0.1011 | 0.1565 | 0.0721 | 0.0902 |
| TH-FactorizedScale | 0.0393 | 0.1331 | 0.1987 | 0.0301 | 0.0988 | 0.1514 | 0.0706 | 0.0877 |
| TH-FactorizedSoft | 0.0385 | 0.1274 | 0.1947 | 0.0294 | 0.0972 | 0.1503 | 0.0690 | 0.0867 |
| TH-MultiView | 0.0381 | 0.1283 | 0.1949 | 0.0275 | 0.0958 | 0.1461 | 0.0678 | 0.0845 |
| TH-MultiViewSoft | 0.0427 | 0.1274 | 0.1958 | 0.0331 | 0.0966 | 0.1504 | 0.0708 | 0.0885 |

CVR test results change the previous conclusion. TH-FixedSoft is strongest on most top-rank CVR metrics (`HR@1/HR@5/N@5`), while TH-Factorized is strongest on the deeper coverage/ranking metrics `HR@10/R@5/R@10/N@10`. TH-MultiViewSoft has the highest `R@1`, but its remaining CVR metrics are weaker. TH-FixedBias remains a strong base, but it is no longer the best CVR target-behavior model under the corrected test-set comparison. Relative to Original GAMER, TH-FixedSoft gives a broad top-rank CVR gain, and TH-Factorized gives the clearest deeper-rank CVR gain.

### Interpretation

1. TH-FixedBias gains are not from a scalar relation-bias value. The current FixedBias table is frozen zero, so the gain is better attributed to replacement-style TH attention, behavior Q/K/V, gating, and behavior-aware MoE.
2. Learnable relation bias is useful on the CVR target behavior. Factorized improves the deeper CVR metrics, while FixedSoft improves most top-rank CVR metrics and merged behavior. Against Original GAMER, the most reliable improvement appears on CVR, not on every merged-behavior metric.
3. Soft prior is helpful for the frozen/fixed relation family, but not consistently helpful for the factorized family. This suggests that prior strength and parameterization should be tuned together.
4. Hard MultiView is clearly weaker. Soft MultiView recovers part of the loss, especially on top-rank CVR metrics, but it is still not competitive with FixedSoft or Factorized as the main model.

### Method-Line Impact

The safer paper main line should now be:

```text
TH Base
+ fixed soft hierarchy prior or factorized relation bias
```

TH-FixedBias should be kept as the base ablation, not the final default. TH-FixedSoft is the strongest merged-behavior and most-top-rank CVR candidate among the new TH variants, while TH-Factorized is the strongest candidate for deeper CVR ranking and coverage. After adding Original GAMER to the comparison, the final paper claim should emphasize target-behavior CVR gains and present merged behavior as competitive with small improvements on selected metrics.

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

Current test-set evidence supports:

- replacement TH attention improves over old added-cross-level GAMER,
- behavior-aware Q/K/V helps both conversion and merged behavior,
- controlled relation bias can improve CVR ranking and coverage,
- fixed soft hierarchy prior can improve merged behavior and most top-rank CVR metrics,
- soft MultiView improves hard MultiView but is not yet the strongest model line.
- compared with MBGen, the TH variants show large gains on both merged behavior and CVR; compared with Original GAMER, the most robust gains are on CVR.

Claims still requiring experiments:

- one relation-bias parameterization always outperforms the others,
- factorized soft prior is beneficial,
- gated view mixture can stably beat relation-bias variants,
- gated MultiView outperforms hard MultiView,
- TH-aware auxiliary objectives and augmentation add further gains.

## Current Conclusion

The model-side main line should be updated from FixedBias-only to:

```text
TH Base
+ FixedSoft hierarchy prior for merged/top-rank behavior
+ Factorized relation bias for CVR depth/coverage
```

The latest ShortVideoAD test-set results show that `TH-FixedSoft` is best among new TH variants on merged behavior and most top-rank CVR metrics, while `TH-Factorized` is best on deeper CVR metrics. Compared with MBGen, these variants are clearly stronger. Compared with Original GAMER, the improvement is clear on CVR but mixed on merged behavior, where FixedSoft slightly improves `HR@5/N@5/N@10` while trailing on `HR@10`. `TH-FixedBias` should remain the clean TH base ablation because its scalar relation bias is frozen zero, but it should no longer be described as the best final variant. `TH-MultiViewSoft` remains a useful structured-view ablation, and hard MultiView should be treated mainly as evidence that rigid view partitioning is too restrictive.

The data-side design should move beyond a user-independent random ratio schedule toward:

```text
time-aware
+ session-aware
+ behavior-level-aware
+ optionally user-adaptive
```

The model, objectives, and augmentation should ultimately express the same temporal-hierarchical view instead of adding unrelated complexity.
