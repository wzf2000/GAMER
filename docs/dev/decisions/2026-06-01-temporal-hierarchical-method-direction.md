# Temporal-Hierarchical Method Direction

## Status

Accepted for next GAMER revision planning.

## Context

The next GAMER revision is being reframed as:

```text
Generative Temporal-Hierarchical Behavioral Modeling for Recommendation
```

This changes the main modeling focus from generic cross-level behavior interaction to a two-dimensional view of user behavior sequences:

- Temporal dimension: interaction order across items and sessions.
- Hierarchical dimension: behavior depth from shallow intent to stronger preference.

Three early design choices affect the implementation path, experiment setup, and paper narrative:

- Whether the main input should remain a deepest-behavior sequence or switch to a flattened behavior-event sequence.
- Whether temporal-hierarchical modeling should be added as a second attention module or replace selected self-attention modules.
- How behavior injection, temporal-hierarchical attention, and MoE layers should be allocated.

Detailed method planning is kept in `docs/dev/design/2026-06-01-temporal-hierarchical-behavior-modeling.md`.

## Decision

Use the current deepest-behavior sequence as the main method input. Treat each behavior token as the final state of a behavior chain, not merely as a single isolated behavior label.

Use flattened behavior-event sequences as an ablation or baseline, not as the default main method input.

Prototype temporal-hierarchical relation modeling first inside the existing two-attention Qwen3Multi structure:

```text
self-attention -> temporal-hierarchical cross-attention -> MoE/FFN
```

For the main paper method, move toward replacement-style Temporal-Hierarchical Attention in selected Transformer layers:

```text
temporal-hierarchical attention -> MoE/FFN
```

This replacement attention should preserve ordinary causal next-token modeling and add hierarchy through relation bias or view-specific attention heads.

For an 8-layer model, prefer the following allocation:

```text
Layer 0-1: standard causal attention + behavior-injected MoE
Layer 2-5: temporal-hierarchical attention + behavior-injected MoE
Layer 6-7: standard causal attention + MoE
```

Keep MoE active in all layers by default. Make behavior injection cover at least all temporal-hierarchical attention layers.

## Rationale

The deepest-behavior sequence keeps the current training and evaluation protocol stable while supporting the new behavior-chain-state interpretation. It also avoids confounding method gains with a large input-format change.

Flattening all behavior events preserves raw temporal order, but it mixes within-item behavior progression with cross-item temporal transition. This can make the new temporal-hierarchical contribution less clean.

The existing two-attention structure is useful for fast implementation because the current code already has cross-level attention hooks. However, as a final main method, it increases attention capacity compared with a standard Transformer layer. A replacement-style attention module gives a fairer comparison because the total number of attention layers can remain fixed.

Middle layers are the preferred location for temporal-hierarchical attention because they can shape sequence representations before final generation. Final layers can remain closer to target-conditioned semantic ID decoding.

## Consequences

The first implementation can be incremental and low-risk by extending the existing cross-level attention module.

The paper's main method should eventually report a fair replacement-style variant, not only an added-attention variant.

Experiments should include input-format ablations:

- Deepest-behavior / behavior-chain-state sequence.
- Flattened behavior-event sequence.
- Optional chain-expanded sequence if implemented.

Experiments should also include architecture fairness ablations:

- Ordinary Qwen3 baseline.
- Current added cross-attention Qwen3Multi.
- Replacement-style temporal-hierarchical attention with the same number of Transformer attention layers.

