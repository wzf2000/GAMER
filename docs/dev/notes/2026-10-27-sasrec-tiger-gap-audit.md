# SASRec/TIGER gap: confirmed implementation defect and protocol differences

Chinese counterpart: `2026-10-27-sasrec-tiger-gap-audit.zh.md`. Audit checkout: `ca0d931`; no model/training source code was changed and no production training was launched.

**Subsequent correction:** The user authorized the fix and prioritized both Tmall SASRec reruns. Commit `d8d06c3` corrects the residual condition; CPU tests, required lint, and full-vocabulary GPU-step validation passed. Two fresh jobs were launched on GPUs 1 and 5, retaining the existing protocols. See `2026-10-27-sasrec-ffn-fix-reruns.md`. The observations below describe the pre-fix audit snapshot, not the current corrected forward.

## What the result gap establishes

The recovered Tmall alipay rows give TIGER/SASRec ratios of **7.5658× HR@10** and **8.7901× NDCG@10**. Both identified tasks ignore explicit behavior identity, but this does not establish matched architectures, objectives, training samples, selection protocols, or historical hyperparameters. These results alone cannot attribute the gap to generative recommendation.

## Confirmed FFN defect affecting the current SASRec path

`SeqRec/modules/layers/transformer.py` constructs `FeedForward` with `residual=True` by default and creates its dropout/LayerNorm only in that case. Its forward executes dropout, the residual addition, and LayerNorm under **`if not self.residual`**. `TransformerEncoderLayer`, used by both SASRec variants, does not override this flag. Consequently the default SASRec FFN skips all three operations.

This differs from the repository's pre-refactor implementation and the [RecBole FeedForward reference](https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/layers.py), which apply dropout and residual LayerNorm to the FFN output. The inversion was introduced by commit `7aae2a34f9d865a6db6fad153dc48b39a2e38cf5` on **2025-09-28**. Archive timestamps record SASRec result files on 2025-10-05, but timestamps are not proof of the actual training checkout.

CPU probes, seed 42:

- Zero only the FFN's two dense projections; keep LayerNorm weight=1 and bias=0. Actual output maximum absolute value is **0**; the expected residual LayerNorm output maximum is **2.3465452194**.
- Forward hooks record **zero calls** to this FFN's dropout and LayerNorm.
- A complete two-layer tiny SASRec loss/backward leaves all four FFN LayerNorm weight/bias tensors without gradients; the other registered parameters receive gradients.
- Explicit `residual=False` instead raises `AttributeError: 'FeedForward' object has no attribute 'dropout'`, confirming inconsistent constructor/forward branches.

This is a confirmed implementation error, not a measured explanation of the complete paper-score gap. The original training commit/checkpoint is absent. Its impact requires a controlled corrected training run; changing inference code alone cannot repair a model trained with the erroneous computation.

## Other confirmed differences under current code

| Aspect | SASRec `smb_dis_sample` | TIGER `smb` |
| --- | --- | --- |
| Item representation | Atomic item IDs and a flat item softmax | CID token sequence and autoregressive token prediction |
| Architecture source defaults | 2-layer causal encoder, FFN 256 | 4-layer encoder + 4-layer decoder, FFN 1024 |
| Source dropout | 0.5 | 0.1 |
| Default selection | Alipay sampled-validation ranking, default NDCG@10 | All-behavior validation token loss, unless filtered explicitly |
| Empty-history training samples | Skipped | Retained except for the very first event |

Source defaults do not establish the final historical settings. An injective CID changes parameter sharing and optimization, but adds no behavior identity or external semantic information merely by renaming items. Total parameter capacity also cannot be inferred from layer count: SASRec's large atomic item embedding dominates its parameter count.

Read-only counts from current Tmall event/session JSON, applying the present source rules with default session training and no cached-sample substitution:

- 217,374 users, 3,818,122 events, **361,932 unique items**, IDs 0 through 361,931 with no gaps.
- The currently available V2 CID index contains the same item-key set and **361,932 unique CID strings**, with no collisions. It is not an independently recovered historical index.
- Generative `smb` construction yields **1,782,306** samples; discriminative construction yields **1,057,649**. The difference, **724,657** samples, consists of first-session targets with no prior-session history; the generative builder includes them and the discriminative builder skips them. Session IDs were verified to be nondecreasing for all users.
- These are current construction-rule counts, not recovered historical training manifests or a measurement of performance effects. Different cached samples, history limits, or execution versions still need original records.

Current test code uses essentially the same item catalogue for ordinary SASRec and `smb` TIGER, aside from padding handling and exact full-sort versus beam search. The supplied TIGER result reports zero target SID collisions. All three archived alipay HR rows are consistent with integer hit counts over 6,329 users; SASRec lacks per-user UID records, so exact historical population identity remains unverified.

## Interpretation and next verification

Prioritize correcting and independently verifying the SASRec FFN, then align mixed-history input, nonempty-history training targets, history length, full-catalogue alipay validation/test, model selection, and tuning budget. Retrain SASRec and SASRec_B under an explicitly identified corrected implementation before attributing their low paper scores to model limitations. Measure the correction's effect rather than assuming it recovers a particular score. No corrected-score or significance claim is currently available.

Diagnostic output: `results/rebuttal/2026-10-27-sasrec-tiger-gap-audit/diagnostics.json`. Original result provenance is documented in `2026-10-27-tmall-baseline-provenance.md`. Runtime artifacts are not committed.
