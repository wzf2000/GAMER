# Generative Temporal-Hierarchical Behavioral Modeling

## Background

The next GAMER revision should shift the method framing from generic multi-level behavior modeling to:

```text
Generative Temporal-Hierarchical Behavioral Modeling for Recommendation
```

The central object is no longer a one-dimensional behavior sequence, but a two-dimensional user behavior lattice:

- Temporal axis: the interaction order across items and sessions.
- Hierarchical axis: behavior depth from shallow intent to stronger preference, for example impression/PXS, click, activation, payment.

Current session-wise GAMER input already keeps one behavior token per interacted item, where the token represents the deepest observed behavior in the behavior chain. This is compact, but it means the model must infer both temporal transition and behavior-level transition from the same flattened token stream.

The motivating figure is stored at `docs/dev/asserts/temporal-hierachical.png`.

## Existing Cross-Level Implementation

Current implementation:

- Main model: `SeqRec/models/generative/Qwen3Multi/model.py`.
- Router: `SeqRec/models/generative/Qwen3Multi/router.py`.
- Cross-level layers are selected by `config.cross_attention_decoder`.
- `Qwen3MultiAttention(is_cross=True)` injects behavior embeddings into Q/K/V and gates the output.
- `Qwen3MultiModel._compute_action_block_mask()` builds the cross-attention mask.

The current `cross_mask_type` variants are:

- `causal`: no behavior-level restriction, equivalent to mixed flattened next-token modeling.
- `level`: query can attend only to strictly lower-level previous behavior tokens.
- `geq`: query can attend to same-level and lower-level previous behavior tokens.
- `reverse`: reversed level direction.
- `soft`: continuous bias favoring lower-level-to-higher-level direction.

This implementation mainly treats behavior hierarchy as an attention accessibility rule. That captures action-transition style modeling, but it underuses the idea that different behavior levels can mutually support next-token prediction as long as temporal causality is respected.

## Problem With One-Dimensional Baselines

Flattened multi-behavior sequences preserve global time order:

```text
pxs -> click -> pxs -> click -> activation -> payment -> ...
```

This is simple and compatible with next item prediction, but it mixes behavior semantics into one path. It can worsen high-level behavior sparsity and does not explicitly model behavior-level dependencies.

Max-depth-only sequences keep only the strongest behavior per item:

```text
click -> payment -> pxs -> click -> ...
```

This reduces sequence length, but discards intermediate behavior-chain evidence.

Per-level sequence modeling builds a separate sequence for each behavior:

```text
pxs:        i1 -> i2 -> i5 -> ...
click:      i1 -> i3 -> ...
activation: i3 -> ...
payment:    i4 -> ...
```

This makes each level's next item prediction clearer, but weakens temporal relations across levels because the original interleaving order is no longer a first-class signal.

The target method should unify these views: retain causal temporal order, expose behavior hierarchy, and allow learnable cross-level collaboration instead of fixing one hard information direction.

## Design Goal

Model the next token over a temporal-hierarchical behavior lattice:

```text
p(x_{t+1}, b_{t+1} | x_{\le t}, b_{\le t})
```

The model should:

- Respect temporal causality: no future interaction or future semantic token may be visible.
- Preserve ordinary next item prediction as a valid special case.
- Preserve per-level next item prediction as a valid special case.
- Preserve action-transition prediction as a valid special case.
- Learn when same-level, lower-to-higher, higher-to-lower, and mixed-level evidence is useful.
- Make behavior-level order explicit through embeddings, biases, objectives, or routing, not only through a binary mask.

## Directional Design Decisions

Decision record: `docs/dev/decisions/2026-06-01-temporal-hierarchical-method-direction.md`.

### Input Representation

Recommended main setting: keep the current deepest-behavior sequence, and frame each behavior token as the final state of a behavior chain.

Example:

```text
pxs -> click -> activation -> payment
```

is represented compactly as:

```text
payment
```

This should be interpreted as a behavior-chain state token, not simply as dropping lower-level evidence. Under this framing, the sequence remains compact while each item-level event still carries hierarchical meaning.

Reasons:

- It keeps the current training/evaluation protocol stable.
- It prevents the method contribution from being confounded with a large input-format change.
- It avoids mixing within-item vertical progression and cross-item temporal transition into the same flattened event stream.
- It makes the paper claim cleaner: the model learns temporal-hierarchical relations from compact behavior-chain states, instead of relying on raw sequence expansion.

Flattened behavior event sequences should be treated as an important ablation, not the primary method:

```text
pxs -> click -> pxs -> click -> activation -> payment -> ...
```

Optional behavior-chain expansion can be explored later as an auxiliary input view:

```text
(item i, payment) -> (i, pxs), (i, click), (i, activation), (i, payment)
```

Recommended experiment grouping:

- Main: deepest-behavior / behavior-chain-state sequence.
- Ablation: raw flattened behavior sequence.
- Ablation: per-level sequence modeling, if implemented or available as a baseline.
- Extension: chain-expanded input view.

### Attention Architecture

Current Qwen3Multi layer structure is:

```text
self-attention -> cross-level attention -> MoE/FFN
```

This is convenient for prototyping, but it increases the number of attention modules compared with a standard Transformer layer. If used as the final main method, improvements may be harder to attribute because the model has both temporal-hierarchical inductive bias and extra attention capacity.

Recommended route:

1. Short-term prototype: keep the existing two-attention structure and replace the current cross-level attention mask with temporal-hierarchical relation bias. This minimizes code changes and allows quick validation.
2. Main paper method: move toward replacement-style Temporal-Hierarchical Attention, where selected Transformer layers replace ordinary self-attention:

```text
temporal-hierarchical attention -> MoE/FFN
```

The replacement attention should still contain standard causal next-token modeling, with hierarchy entering as an additive relation bias or view-specific head bias:

```text
score = qk / sqrt(d) + causal_mask + temporal_hierarchical_bias
```

This makes the method fairer and conceptually cleaner: the model does not add extra attention depth; it upgrades part of the ordinary causal attention into temporal-hierarchical attention.

Recommended fairness comparisons:

- Qwen3 baseline: 8 ordinary attention layers.
- Current Qwen3Multi: 8 ordinary attention layers plus cross-level attention in selected layers.
- New main method: 8 total attention layers, with selected ordinary attention layers replaced by temporal-hierarchical attention.

### Layer Allocation

Current default config:

```json
"sparse_layers_decoder": [0, 1, 2, 3, 4, 5, 6, 7],
"behavior_injection_decoder": [0, 1, 2, 3],
"cross_attention_decoder": [4, 5, 6, 7]
```

This has a reasonable low-level/high-level split, but the new temporal-hierarchical framing suggests a more explicit allocation:

```text
Layer 0-1: local token and item representation.
Layer 2-5: temporal-hierarchical relation modeling.
Layer 6-7: target-level generation refinement.
```

Recommended main allocation for an 8-layer replacement-style model:

```text
Layer 0-1: standard causal attention + behavior-injected MoE
Layer 2-5: temporal-hierarchical attention + behavior-injected MoE
Layer 6-7: standard causal attention + MoE
```

MoE can remain active in all 8 layers because it handles semantic-token position and behavior-conditioned transformation. It is not necessarily tied to cross-level attention.

Behavior injection should at least cover all temporal-hierarchical attention layers. A practical config candidate is:

```json
"sparse_layers_decoder": [0, 1, 2, 3, 4, 5, 6, 7],
"behavior_injection_decoder": [0, 1, 2, 3, 4, 5],
"temporal_hierarchical_attention_decoder": [2, 3, 4, 5]
```

If keeping the current two-attention prototype, prefer moving cross-level layers from the final four layers to middle-high layers:

```json
"behavior_injection_decoder": [0, 1, 2, 3, 4, 5],
"cross_attention_decoder": [2, 3, 4, 5]
```

Rationale:

- Middle layers are early enough to shape sequence representations.
- Final layers can specialize in target-conditioned generation and semantic ID decoding.
- Behavior injection and temporal-hierarchical attention operate on overlapping layers, so the relation module receives explicit behavior information.

## Proposed Method Family

### 1. Temporal-Hierarchical Relative Attention Bias

Replace hard cross-level masks as the primary behavior relation mechanism with a learnable pairwise bias:

```text
score(i, j) = q_i k_j / sqrt(d)
            + temporal_bias(delta_t)
            + level_pair_bias(level_i, level_j)
            + level_gap_bias(level_i - level_j)
```

The causal mask remains mandatory. Behavior relations become additive attention biases, so every previous token can be used, while the model is aware of direction and distance in the behavior hierarchy.

Implementation direction:

- Add `cross_mask_type="temporal_hierarchical"` or a separate `cross_relation_type`.
- Keep `in_item_mask` and padding/session masking unchanged.
- Add learnable tables:
  - `level_pair_bias[num_behavior + 1, num_behavior + 1, num_heads]`.
  - optionally `level_gap_bias[2 * num_behavior + 1, num_heads]`.
- Apply the bias in cross-attention only, or in both self-attention and cross-attention as an ablation.

Expected benefit:

- `causal`, `level`, `geq`, and `soft` become fixed points inside a broader learnable relation model.
- High-level sparse behaviors can attend to lower-level evidence, but the model can also use high-level evidence to contextualize later low-level behaviors.

### 2. Multi-View Cross-Level Attention Heads

Instead of choosing one global cross mask, split cross-level attention heads into semantic views:

- Temporal heads: causal-only, all previous behaviors visible.
- Same-level heads: emphasize per-behavior next item prediction.
- Upward heads: lower-level history to higher-level query, close to CTR/CVR/action prediction.
- Downward heads: higher-level history to lower-level query, useful for post-conversion or repeated-intent context.

The layer output mixes these views through learned gates:

```text
h_cross = gate_temporal * A_temporal
        + gate_same     * A_same
        + gate_up       * A_up
        + gate_down     * A_down
```

Implementation direction:

- Keep one `Qwen3MultiAttention` module but build multiple additive masks/biases.
- Either partition heads by view or compute several attention outputs and fuse them.
- Start with head partitioning for lower cost.
- Add config fields:
  - `cross_view_types`: `["causal", "same", "up", "down"]`.
  - `cross_view_head_allocation`: optional list or automatic equal split.
  - `cross_view_gate`: `static`, `query`, or `query_level`.

Expected benefit:

- Makes the method story explicit: the model contains flattened, per-level, and action-transition views simultaneously.
- More interpretable ablations: remove one view at a time.

### 3. Behavior-Chain Expansion as an Optional Input View

Current data uses only the deepest behavior token per item. To better match the vertical axis, add an optional behavior-chain expansion:

```text
(item i, payment) -> (i, pxs), (i, click), (i, activation), (i, payment)
```

This expansion should not replace the compact max-depth input initially. It should be an auxiliary input construction controlled by task/config, because it increases sequence length.

Implementation direction:

- Add a dataset option such as `behavior_chain_view`.
- For each deepest behavior, expand to all levels up to that behavior.
- Assign identical item semantic IDs and monotonic behavior tokens.
- Keep the same item-local causal token mask so semantic ID tokens still decode left-to-right.
- Compare with current max-depth input and pure flattened raw behavior input if raw chains are available.

Expected benefit:

- Gives the model explicit vertical supervision for missing intermediate behavior-chain states.
- Makes action prediction and next item prediction share the same item identity.

Risk:

- Sequence length may grow by the average behavior depth.
- Need careful max length truncation so high-level target events are not disproportionately dropped.

### 4. Temporal-Hierarchical Auxiliary Objectives

Keep next-token generation as the primary objective, but add lightweight auxiliary losses that match the new framing.

Candidate objectives:

- Level transition prediction: predict the next behavior level from the current hidden state.
- Same-level next item contrast: for a query level, contrast the next item among same-level future targets.
- Upward conversion prediction: given a lower-level interaction hidden state, predict whether the same item reaches a higher level later in the session.
- Relation distillation: encourage attention mass over the four views to match weak labels from level relation type.

Recommended first auxiliary objective:

```text
L = L_next_token + lambda_level * L_next_level
```

Use a small `lambda_level` and predict the behavior token/level at behavior-token positions only. This is cheap, uses existing labels, and directly improves behavior-hierarchy awareness.

### 5. Level-Aware Generation Prompting

Evaluation currently appends the target behavior token and generates item semantic IDs. Keep this protocol, but interpret it as conditional generation on a target behavior level:

```text
generate item | history, target_behavior_level
```

The new method can add target-level embeddings to the final hidden state or decoding prompt, especially if cross-level attention is made less restrictive. This keeps the generation target explicit even when the model uses all causal history.

Implementation direction:

- Reuse existing behavior token prompting first.
- Later add a `target_action_index` to generation if behavior token prompting is insufficient.

## Recommended First Implementation

Start with Temporal-Hierarchical Relative Attention Bias because it is the smallest code change with the strongest conceptual alignment.

Step 1:

- Add `cross_mask_type="th_bias"` to `_compute_action_block_mask()` or create a separate bias builder.
- Keep the causal/in-item/session/padding masks exactly as they are.
- Add a learnable level-pair attention bias inside `Qwen3MultiAttention(is_cross=True)`.
- Initialize the bias to zero so the model starts from causal cross-attention.

Step 2:

- Add configs:
  - `Qwen3MultiTHBias`: causal cross-attention plus learnable level-pair bias.
  - `Qwen3MultiTHBiasSoftInit`: initialize level-pair bias from the current `soft` rule.

Step 3:

- Add ablations:
  - `causal`: flattened sequence view.
  - `level`: strict lower-level attention.
  - `geq`: same/lower-level attention.
  - `soft`: fixed hierarchy bias.
  - `th_bias`: learned temporal-hierarchical bias.

Step 4:

- Add optional attention diagnostics:
  - attention mass by relation type: same, lower-to-higher, higher-to-lower.
  - metrics by target behavior level.
  - sparse high-level target performance.

## Experiment Plan

Primary comparison:

- Current `Qwen3Multi` default.
- `Qwen3MultiCausal`.
- `Qwen3MultiSoft`.
- New `Qwen3MultiTHBias`.
- New `Qwen3MultiMultiView` if implemented.

Metrics:

- Existing HR, Recall, NDCG.
- Per-behavior metrics.
- High-level sparse behavior metrics through `scripts/analyze_sparse_behavior.sh`.
- Behavior dropout robustness through `scripts/analyze_behavior_dropout.sh`.

Datasets:

- Start with ShortVideoAD because README already uses it as the main Qwen3Multi example.
- Then extend to other session-wise multi-behavior datasets where behavior hierarchy is reliable.

Minimum verification for code changes:

- `python -m compileall main.py SeqRec`
- `bash -n scripts/train_SMB_decoder.sh`
- `bash -n scripts/test_SMB_decoder.sh`
- A tiny single-GPU smoke run only after confirming data/checkpoint availability.

## Paper Narrative

The revised paper can position GAMER as a generative model over a temporal-hierarchical behavior lattice.

Key claims:

- Flattened generative recommendation preserves temporal order but ignores behavior hierarchy.
- Per-level behavior modeling preserves hierarchy but weakens cross-level temporal order.
- Conversion/action prediction captures vertical progression but is not sufficient for sequential recommendation.
- GAMER unifies these views with causal temporal modeling and learnable hierarchical relation modeling.

Potential method naming:

- Temporal-Hierarchical Cross Attention (THCA).
- Temporal-Hierarchical Relation Bias (THRB).
- Multi-View Hierarchical Attention (MVHA).

Recommended first method name:

```text
Temporal-Hierarchical Relation Bias
```

It is precise, easy to connect to attention, and compatible with the current Qwen3Multi implementation.

## Open Questions

- Whether raw datasets contain complete behavior chains or only deepest behavior labels.
- Whether behavior levels are strictly ordinal across all datasets.
- Whether downward evidence from high-level behavior to later lower-level behavior improves recommendation or introduces popularity leakage.
- Whether level-pair bias should be shared across layers or layer-specific.
- Whether the auxiliary level objective improves hierarchy awareness without hurting item token generation.
