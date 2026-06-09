# TH Results Analysis And Next Design

## Background

The current ShortVideoAD `smb_explicit_decoder_4` experiments have tested three main Temporal-Hierarchical variants:

- `Qwen3TemporalHierarchicalFixedBias`
- `Qwen3TemporalHierarchicalFactorized`
- `Qwen3TemporalHierarchicalMultiView`

Two additional configs have been added for follow-up experiments:

- `Qwen3TemporalHierarchicalFixedSoft`
- `Qwen3TemporalHierarchicalFactorizedSoft`

This document summarizes the current results and recommends which direction should be improved if the final model needs to emphasize the Temporal-Hierarchical contribution.

## Current Results

### Conversion / CVR Target Behavior

Comparison against the previous `GAMER (SID)`:

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Old GAMER SID | 0.0394 | 0.1280 | 0.1944 | 0.0292 | 0.0966 | 0.1478 | 0.0687 | 0.0856 |
| FixedBias | **0.0438** | **0.1368** | **0.2068** | **0.0348** | **0.1052** | **0.1597** | **0.0756** | **0.0936** |
| Factorized | 0.0419 | 0.1354 | 0.2044 | 0.0331 | 0.1052 | 0.1588 | 0.0747 | 0.0924 |
| MultiView | 0.0394 | 0.1345 | 0.2018 | 0.0309 | 0.1028 | 0.1556 | 0.0723 | 0.0898 |

Observations:

- All three TH variants outperform the previous GAMER SID result.
- FixedBias is strongest, Factorized is very close, and MultiView is weaker but still effective.
- FixedBias improves over old GAMER SID by roughly `+6.4%` to `+19.3%` on CVR metrics.
- Factorized preserves learnable relation-bias modeling, but is slightly weaker than FixedBias in the current run.

### Merged Behavior-Specific Task

Comparison against the previous `GAMER (SID)`:

| Model | HR@5 | HR@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|
| Old GAMER SID | 0.1443 | 0.2129 | 0.0621 | 0.0753 |
| FixedBias | **0.1502** | **0.2227** | **0.0656** | **0.0799** |
| Factorized | 0.1500 | 0.2220 | 0.0655 | 0.0796 |
| MultiView | 0.1478 | 0.2162 | 0.0632 | 0.0766 |

Observations:

- All three TH variants also outperform old GAMER SID on the merged task.
- FixedBias and Factorized are almost tied, suggesting the relation-bias family is more stable than MultiView.
- MultiView improves less, but still validates the temporal/same/up/down decomposition.

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

Its slight underperformance against FixedBias may mean:

- TH Q/K/V embeddings already capture much of the hierarchy.
- zero-init factorized bias needs a better prior or regularization.
- a direct logit bias can mildly disturb an already effective attention distribution.
- rank=4 may not be optimal.

### What MultiView Represents

MultiView splits heads into:

- temporal,
- same,
- up,
- down.

It is the most interpretable structured version, directly corresponding to the two-dimensional temporal-hierarchical lattice. However, its hard masks restrict information flow, making it less flexible than relation-bias variants. Current results show it is effective but not strongest.

## Recommended Main Direction

If the final model needs to emphasize the TH contribution, the main line should continue from `Factorized`, not directly from FixedBias.

Reasons:

- FixedBias is currently best, but its scalar bias is zero, so it is hard to claim explicit relation-bias modeling.
- Factorized is only slightly weaker while preserving the main story: learnable temporal-hierarchical relation modeling.
- Factorized is stronger than MultiView, suggesting soft/continuous relation modeling is preferable to hard view partitioning.
- FixedBias can be used as a strong TH-base ablation.

Recommended paper positioning:

```text
Main model: Factorized Temporal-Hierarchical Relation Bias
Strong ablation: TH Attention w/o scalar relation bias
Structured ablation: Multi-View Temporal-Hierarchical Attention
```

If `FactorizedSoft` matches or exceeds FixedBias, the method story becomes stronger:

```text
learnable factorized TH relation bias initialized with a weak hierarchy prior
```

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

1. `FixedSoft` and `FactorizedSoft` with scale `0.05`.
2. Factorized rank ablation: `1/2/4/8`.
3. Relation bias scale or learnable alpha.
4. Soft/gated MultiView.
5. Behavior-level auxiliary objective.
6. Attention/bias visualization diagnostics.

## Current Recommendation

If only metrics matter, FixedBias is currently strongest.

If the final design needs to emphasize Temporal-Hierarchical modeling, continue from Factorized:

```text
Temporal-Hierarchical Attention
+ behavior-aware Q/K/V
+ learnable factorized relation bias
```

FixedBias should be used as a strong base ablation; MultiView should be used as a structured-view ablation; Factorized or FactorizedSoft is the best candidate for the final main method.
