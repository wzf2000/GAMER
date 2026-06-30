# TH Results Analysis And Next Design

## Background

The current ShortVideoAD `smb_explicit_decoder_4` experiments have tested the implemented Temporal-Hierarchical structural variants. This document now uses the `smb_explicit` test-set results, not the earlier validation-set comparison.

- `Qwen3TemporalHierarchicalFixedBias`
- `Qwen3TemporalHierarchicalFactorized`
- `Qwen3TemporalHierarchicalFactorizedScale`
- `Qwen3TemporalHierarchicalFactorizedSoft`
- `Qwen3TemporalHierarchicalFixedSoft`
- `Qwen3TemporalHierarchicalMultiView`
- `Qwen3TemporalHierarchicalMultiViewSoft`

This document summarizes the current results and recommends which direction should be improved if the final model needs to emphasize the Temporal-Hierarchical contribution.

## Current Results

### Conversion / CVR Target Behavior

Comparison against the previous `GAMER (SID)`:

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Old GAMER SID | 0.0394 | 0.1280 | 0.1944 | 0.0292 | 0.0966 | 0.1478 | 0.0687 | 0.0856 |
| FixedBias | 0.0390 | 0.1283 | 0.1974 | 0.0290 | 0.0963 | 0.1507 | 0.0693 | 0.0873 |
| FixedSoft | **0.0435** | **0.1349** | 0.1981 | 0.0326 | 0.1007 | 0.1513 | **0.0735** | 0.0900 |
| Factorized | 0.0409 | 0.1342 | **0.2042** | 0.0302 | **0.1011** | **0.1565** | 0.0721 | **0.0902** |
| FactorizedScale | 0.0393 | 0.1331 | 0.1987 | 0.0301 | 0.0988 | 0.1514 | 0.0706 | 0.0877 |
| FactorizedSoft | 0.0385 | 0.1274 | 0.1947 | 0.0294 | 0.0972 | 0.1503 | 0.0690 | 0.0867 |
| MultiView | 0.0381 | 0.1283 | 0.1949 | 0.0275 | 0.0958 | 0.1461 | 0.0678 | 0.0845 |
| MultiViewSoft | 0.0427 | 0.1274 | 0.1958 | **0.0331** | 0.0966 | 0.1504 | 0.0708 | 0.0885 |

Observations:

- The corrected test-set comparison is more mixed than the earlier validation-set comparison.
- FixedSoft is strongest on top-rank CVR metrics (`HR@1/HR@5/N@5`) and is a better final candidate than FixedBias if the target is early ranking.
- Factorized is strongest on deeper CVR metrics (`HR@10/R@5/R@10/N@10`), making it the strongest relation-bias candidate for coverage and depth.
- FixedBias remains the clean TH base, but it should no longer be described as the most stable final model on the CVR target behavior.
- Soft MultiView improves over Hard MultiView, but remains below FixedSoft/Factorized as a main model line.

### Merged Behavior-Specific Task

Comparison against the previous `GAMER (SID)`:

| Model | HR@5 | HR@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|
| Old GAMER SID | 0.1443 | 0.2129 | 0.0621 | 0.0753 |
| FixedBias | 0.1444 | 0.2116 | 0.0620 | 0.0750 |
| FixedSoft | **0.1450** | **0.2121** | **0.0628** | **0.0756** |
| Factorized | 0.1430 | 0.2117 | 0.0614 | 0.0746 |
| FactorizedScale | 0.1432 | 0.2113 | 0.0614 | 0.0745 |
| FactorizedSoft | 0.1434 | 0.2099 | 0.0615 | 0.0744 |
| MultiView | 0.1391 | 0.2062 | 0.0595 | 0.0723 |
| MultiViewSoft | 0.1418 | 0.2102 | 0.0609 | 0.0742 |

Observations:

- FixedSoft is now the strongest merged-behavior variant on all four reported metrics.
- FixedBias and Factorized remain close, but neither exceeds FixedSoft on the merged test-set comparison.
- MultiViewSoft is still materially better than Hard MultiView, but the soft-view line is no longer the strongest merged-behavior candidate.
- The test-set ranking therefore favors fixed soft hierarchy prior for merged behavior and factorized relation bias for CVR depth.

## Interpretation

### What FixedBias Actually Represents

Current FixedBias uses:

```json
"th_relation_bias_type": "table",
"th_relation_bias_trainable": false,
"th_relation_bias_init": "zero"
```

Therefore its scalar relation bias is zero and does not provide an explicit hierarchy prior.

Its gain likely comes from:

- replacement-style TH attention,
- behavior-level Q/K/V embeddings,
- attention output gating,
- behavior-aware MoE/FFN.

It should be described as:

```text
TH Attention without scalar relation bias
```

or:

```text
TH Embedding-only
```

It is a strong base variant, but it is not the best final-method story if the paper wants to emphasize learnable relation bias.

### What Factorized Represents

Factorized adds learnable low-rank behavior-level pair bias on top of the FixedBias base:

```text
bias(q_level, k_level, head)
  = query_factor[q_level, head] · key_factor[k_level, head]
```

It preserves learnable temporal-hierarchical relation modeling while avoiding the extremely slow backward path of the naive trainable table.

Its improvement on deeper CVR metrics, together with weaker merged behavior, may mean:

- TH Q/K/V embeddings already capture much of the hierarchy.
- zero-init factorized bias needs a better prior or regularization.
- a direct logit bias can mildly disturb an already effective attention distribution.
- bias strength matters, but the current fixed-scale variant is not uniformly better than the unscaled factorized version on the test set.
- rank=4 may not be optimal.

### What MultiView Represents

MultiView splits heads into:

- temporal,
- same,
- up,
- down.

It is the most interpretable structured version, directly corresponding to the two-dimensional temporal-hierarchical lattice. However, its hard masks restrict information flow, making it less flexible than relation-bias variants. Current results show Hard MultiView is effective but clearly weaker.

Soft MultiView partially fixes this issue, but the corrected test-set results show that it trails FixedSoft on merged behavior and Factorized on CVR depth. This means the MultiView story is useful as an interpretable structural comparison, but the current soft version is not yet strong enough to replace relation-bias variants as the final model.

## Recommended Main Direction

If the final model needs to emphasize the TH contribution while staying faithful to the current test-set results, the main line should use `TH Base` as the base ablation and choose between `FixedSoft` and `Factorized` as the final variant depending on the paper focus.

Reasons:

- FixedSoft is the best merged-behavior and most-top-rank CVR model.
- Factorized is the best deeper CVR ranking and coverage model.
- FixedBias remains necessary as the clean TH base because its scalar table is zero.
- MultiViewSoft shows that soft view constraints are better than hard masks, but its result is still weaker than the best relation-bias/prior variants.

Recommended paper positioning:

```text
Main candidates: TH-FixedSoft and TH-Factorized
Base ablation: TH Base / FixedBias
Relation-bias extension: Factorized Temporal-Hierarchical Relation Bias
Structured ablation: Soft/Hard Multi-View Temporal-Hierarchical Attention
```

The paper can describe relation control and soft hierarchy prior as TH-aware enhancements, but the latest test metrics do not support making hard MultiView or FixedBias-only the final story.

## Follow-Up Directions

### P1. Test Soft Prior Initialization

New configs:

- `Qwen3TemporalHierarchicalFixedSoft`
- `Qwen3TemporalHierarchicalFactorizedSoft`

Both use scale `0.05`.

Key questions:

- Does FixedSoft outperform FixedZero?
- Does FactorizedSoft outperform FactorizedZero?
- Does soft prior help CVR but hurt click/p3s or merged behavior?

### P2. Factorized Rank Ablation

Test:

```text
rank = 1, 2, 4, 8
```

If rank 1/2 is close to rank 4, the behavior-level relation may be very low-rank and easier to explain.

### P3. Relation Bias Scale Or Regularization

Factorized bias is added directly to attention logits. Add a scale:

```text
score = qk / sqrt(d) + alpha * relation_bias
```

Possible variants:

- fixed `alpha`, such as `0.1` or `0.3`,
- learnable scalar initialized to 0 or a small value,
- layer-specific alpha.

### P4. Soft/Gated MultiView

Current MultiView uses fixed hard head partitioning. A softer version could learn mixture weights over temporal/same/up/down views per head or per query level. This keeps interpretability while reducing the cost of hard masking.

### P5. Relation Bias Sharing Across Layers

Current TH layers each own their relation parameters. Test:

- shared factors across all TH layers,
- group-shared factors,
- independent factors with similarity regularization.

### P6. Behavior-Level Auxiliary Objective

Add:

```text
L = L_next_token + lambda_level * L_next_behavior_level
```

Only predict behavior level at behavior-token positions. Use a small weight such as `0.05` or `0.1`.

### P7. Attention Diagnostics

Add diagnostics for:

- attention mass over same/up/down/mixed relation types,
- relation mass by target behavior,
- learned factorized bias matrices,
- drift from FixedSoft / FactorizedSoft initialization.

These diagnostics are useful for proving that the model learns TH relations rather than only using extra embeddings.

## Recommended Priority

1. Decide the final test-set main variant between FixedSoft and Factorized according to target metric priority.
2. Learnable alpha for Factorized relation bias; record alpha values by layer.
3. Factorized rank ablation: `1/2/4/8`, preferably under the scaled or alpha-controlled setting.
4. Gated MultiView, because Soft MultiView improves Hard MultiView but is not yet strong enough.
5. Behavior-level auxiliary objective.
6. Attention/bias visualization diagnostics.
7. Sequence augmentation only after the model-side baseline is fixed.

## Current Recommendation

If merged behavior and most top-rank CVR metrics matter most, FixedSoft is currently the safest final model. If CVR depth/coverage is the priority, Factorized is currently the strongest final candidate.

If the final design needs to emphasize Temporal-Hierarchical modeling, the supported claim should be:

```text
Temporal-Hierarchical Attention
+ behavior-aware Q/K/V
+ attention gating
+ optional controlled relation/view bias
```

FixedBias should be treated as the base ablation rather than the default final model. Factorized is currently the most promising relation-bias extension; FixedSoft is the strongest fixed-prior variant; MultiViewSoft is a useful structured-view ablation showing that soft constraints are preferable to hard partitioning.
