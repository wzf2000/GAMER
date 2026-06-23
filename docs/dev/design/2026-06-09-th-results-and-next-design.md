# TH Results Analysis And Next Design

## Background

The current ShortVideoAD `smb_explicit_decoder_4` experiments have tested the implemented Temporal-Hierarchical structural variants:

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
| FixedBias | **0.0438** | **0.1368** | **0.2068** | **0.0348** | **0.1052** | **0.1597** | **0.0756** | **0.0936** |
| Factorized | 0.0419 | 0.1354 | 0.2044 | 0.0331 | 0.1052 | 0.1588 | 0.0747 | 0.0924 |
| FactorizedScale | 0.0428 | **0.1371** | 0.2046 | 0.0340 | **0.1062** | 0.1586 | **0.0758** | 0.0932 |
| FactorizedSoft | 0.0427 | 0.1358 | 0.2045 | 0.0324 | 0.1054 | **0.1600** | 0.0748 | 0.0926 |
| FixedSoft | 0.0405 | 0.1338 | 0.2048 | 0.0320 | 0.1044 | 0.1588 | 0.0735 | 0.0916 |
| MultiView | 0.0394 | 0.1345 | 0.2018 | 0.0309 | 0.1028 | 0.1556 | 0.0723 | 0.0898 |
| MultiViewSoft | 0.0417 | 0.1354 | 0.2038 | 0.0328 | 0.1036 | 0.1577 | 0.0739 | 0.0918 |

Observations:

- All tested TH structural variants outperform the previous GAMER SID result on most CVR metrics.
- FixedBias is still the most stable target-behavior model, especially on top-rank quality (`HR@1`, `R@1`, `N@10`).
- FactorizedScale is the strongest relation-bias extension on `HR@5`, `R@5`, and `N@5`, suggesting that controlling relation-bias strength is more useful than a soft prior alone.
- FactorizedSoft only wins on `R@10`, which looks more like broader candidate coverage than better top-rank ordering.
- Soft MultiView improves over Hard MultiView, but remains below FixedBias on the CVR target behavior.

### Merged Behavior-Specific Task

Comparison against the previous `GAMER (SID)`:

| Model | HR@5 | HR@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|
| Old GAMER SID | 0.1443 | 0.2129 | 0.0621 | 0.0753 |
| FixedBias | 0.1502 | **0.2227** | 0.0656 | **0.0799** |
| Factorized | 0.1500 | 0.2220 | 0.0655 | 0.0796 |
| FactorizedScale | 0.1502 | 0.2221 | 0.0654 | 0.0795 |
| FactorizedSoft | 0.1494 | 0.2199 | 0.0649 | 0.0787 |
| FixedSoft | 0.1497 | 0.2216 | 0.0652 | 0.0795 |
| MultiView | 0.1478 | 0.2162 | 0.0632 | 0.0766 |
| MultiViewSoft | **0.1508** | 0.2218 | **0.0657** | 0.0796 |

Observations:

- Most TH variants outperform old GAMER SID on the merged task.
- MultiViewSoft is now best on merged `HR@5` and `N@5`, showing that soft view penalties are materially better than hard head partitioning.
- FixedBias remains best on merged `HR@10` and `N@10`, so it is still the safest overall ranking model.
- Factorized and FactorizedScale remain very close to FixedBias; relation-bias control changes the metric balance but does not produce a stable overall win.

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

Its slight underperformance against FixedBias, together with the stronger FactorizedScale result on several CVR coverage metrics, may mean:

- TH Q/K/V embeddings already capture much of the hierarchy.
- zero-init factorized bias needs a better prior or regularization.
- a direct logit bias can mildly disturb an already effective attention distribution.
- bias strength matters; a fixed scale helps more than the current soft-prior initialization.
- rank=4 may not be optimal.

### What MultiView Represents

MultiView splits heads into:

- temporal,
- same,
- up,
- down.

It is the most interpretable structured version, directly corresponding to the two-dimensional temporal-hierarchical lattice. However, its hard masks restrict information flow, making it less flexible than relation-bias variants. Current results show Hard MultiView is effective but clearly weaker.

Soft MultiView partially fixes this issue. It is best on merged `HR@5/N@5`, but it still trails FixedBias on the CVR target behavior. This means the MultiView story is useful as an interpretable structural comparison, but the current soft version is not yet strong enough to replace the TH Base as the final model.

## Recommended Main Direction

If the final model needs to emphasize the TH contribution while staying faithful to the current results, the main line should use `FixedBias / TH Base` as the default model and keep controlled relation bias as an extension.

Reasons:

- FixedBias is still the most stable model on the target CVR behavior and on high-rank merged metrics.
- Its scalar table is zero, so the supported main claim should be behavior-aware replacement TH attention rather than nonzero scalar relation bias.
- FactorizedScale is the best current relation-bias extension and should be kept as the main relation-bias ablation/candidate, but it does not stably beat TH Base.
- MultiViewSoft shows that soft view constraints are better than hard masks, but its CVR result is still weaker than FixedBias.

Recommended paper positioning:

```text
Main model: TH Base / FixedBias
Relation-bias extension: Factorized Temporal-Hierarchical Relation Bias with scale control
Structured ablation: Soft/Hard Multi-View Temporal-Hierarchical Attention
```

The paper can still describe relation control as a TH-aware enhancement, but the latest metrics do not support making it the only core contribution.

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

1. Learnable alpha for Factorized relation bias; record alpha values by layer.
2. Factorized rank ablation: `1/2/4/8`, preferably under the scaled setting.
3. Gated MultiView, because Soft MultiView already improves Hard MultiView on merged metrics.
4. Behavior-level auxiliary objective.
5. Attention/bias visualization diagnostics.
6. Sequence augmentation only after the model-side baseline is fixed.

## Current Recommendation

If only metrics matter, FixedBias / TH Base remains the safest final model.

If the final design needs to emphasize Temporal-Hierarchical modeling, the supported claim should be:

```text
Temporal-Hierarchical Attention
+ behavior-aware Q/K/V
+ attention gating
+ optional controlled relation/view bias
```

FixedBias should be treated as the main model unless a later FactorizedAlpha or Gated MultiView result clearly exceeds it. FactorizedScale is currently the most promising relation-bias extension; MultiViewSoft is a useful structured-view ablation showing that soft constraints are preferable to hard partitioning.
