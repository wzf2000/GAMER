# JData-V3 prescribed hierarchy intervention

Chinese counterpart: `2026-10-31-jdata-v3-hierarchy-intervention.zh.md`. Date/time: Asia/Shanghai. Implementation commit: `389c99a`.

## Scope and corrected reference

The author clarified that the paper's Tmall method uses weak auxiliary levels, rather than the legacy identity-order `full` generated in the earlier Tmall plan. Do not treat that generated reference as the paper run. Tmall interventions are deferred; the requested experiment is now JData.

The supplied `JData-V3.zip` contains TH-FixedSoft / `smb_explicit_decoder_4` / CID-shuffle-chunk64 aggregate and per-user results, without checkpoint/config/training arguments, UID sidecars, index file/hash or evaluation manifest. The nominal test buy row is **HR@10 0.726086957, NDCG@10 0.550338765**, with **230** users; merged is **0.516895038 / 0.322759241**. Per-user means were checked against all five aggregate behavior rows. The test split is inferred from `results-smb_explicit-cid-shuffle-64.json`, rather than independently recovered execution metadata.

## Frozen intervention

| Identity | Natural V3 level | Requested prior level |
| --- | --- | --- |
| pv | 0 | 2 |
| click | 1 | 1 |
| cart | 2 | 0 |
| collect | 2 | 0 |
| buy | 3 | 3 |

Variant: `random_auxiliary_reversed`, training seed **42**. The author explicitly selected the stronger reversal before training. This is one prescribed permutation of auxiliary level groups, not a random-map distribution or a mapping sampled from seed 42. Cart/collect remain tied and buy remains highest. The multiplicity of identities at each numeric level changes; this tests this particular prior assignment, including resulting bias distances, not an isolated tie effect.

Only TH attention's hierarchy lookup changes. Behavior token IDs, Q/K/V identity embeddings, gate, position experts/FFN routing, original inputs, augmentation, targets, candidates, split and causal mask are retained. Lambda remains **0.05**, the relation table is frozen, runtime scale is 1, and auxiliary objectives are disabled. The ordinary dataset level mapping remains `0,1,2,2,3`; it is not overwritten by the experimental ranks.

## V3 support and verification

The active checkout previously supported legacy identity-order priors, but lacked V3 schema handling. It now enables identity/level separation only through an explicit dataset schema. Identity indices are still produced by the existing router; a nonpersistent lookup converts them into hierarchy indices inside TH attention. This preserves the router's public return format and other backbones. No branch switch or cherry-pick was performed.

For V3, five identities use a four-level relation table: Q/K/V embeddings have six rows including special ID 0, and the frozen relation table is `5 × 5 × 6`. Internal natural lookup is `[0,1,2,3,3,4]`; the intervention is `[0,3,2,1,1,4]`. Special ID 0 remains 0. Checkpoint reload reconstructs this lookup from config. Legacy `th_prior_behavior_ranks` keeps its existing behavior-to-behavior table initialization semantics.

Both the natural and permuted tiny V3 models were compared against a read-only archive of historical V3 commit `eadc431`: all state tensors, logits, loss, Q-projection gradients and cached beam outputs matched exactly. Cached and uncached generation agreed. New tests cover planner/trainer agreement, hierarchy schema fingerprints, causal shifted supervision, frozen bias, gradients, reload and invalid ranks. The relevant suites passed **53 tests**: 18 model, 17 plan/queue/seed/pipeline and 18 session-behavior dataset tests. Four A100 ranks also passed two DDP backward steps and cached/uncached beam checks. Required flake8, compileall and whitespace checks passed.

JData-V3 metadata and source symlinks follow the historical V3 definition and reuse local JData events/session/time/item files. CID was independently generated on CPU with chunk64/shuffle/seed42 and matches the local V2 index byte-for-byte: `7270735f9a5109d7797b1c5bafe63dca73144291cd417a24d6f38d4d88fe5222`. This does not establish the historical training machine's CID hash. Separate V3 caches were freshly generated; V2 caches/checkpoints were not overwritten.

| Data | Samples |
| --- | ---: |
| Training, factor4 | 49,448 |
| Training-time all-behavior validation loss | 65,224 |
| Validation ranking: pv / click / cart / collect / buy | 4,651 / 9,098 / 1,255 / 233 / 263 |
| Test ranking: pv / click / cart / collect / buy | 4,361 / 9,087 / 1,137 / 219 / 230 |

All five test counts match the author-supplied result archive. Matching counts do not prove exact historical target/UID/index identity. Dataset sources, caches, archive members and index were SHA256-recorded in `results/rebuttal/2026-10-31-jdata-v3-preflight/inputs-and-historical-results.json`.

## Actual recipe and execution

The author recovered the main-method command: `dataset=JData-V3 cid=1 shuffle=1 chunk_size=64 batch_size=512 tasks=smb_explicit_decoder_4 backbone=Qwen3TemporalHierarchicalFixedSoft gpu=0,1,2,3 port=2314 bash scripts/train_SMB_decoder.sh --max_his_len 100 --learning_rate 5e-4 --num_train_epochs 200 --gradient_accumulation_steps 8 --warmup_ratio 0.04 --patience 20 --temperature 0.7`.

The script divides batch 512 by four GPUs, so **microbatch 128 × four GPUs × accumulation 8 = effective batch 4096**. The new run matches those explicit settings. Weight decay 0.01, cosine, AdamW, seed42, FP32 and maximum length1024 are resolved defaults; the old command does not prove those defaults were unchanged on its original checkout. `CUDA_LAUNCH_BLOCKING=1` is retained from the training shell; W&B is offline for this run. Ranking uses four GPUs, per-device batch32, beam20, all five behaviors and UID/original-target export.

The sole authorized variant started at **2026-10-31 19:59:13**, on GPU **0,1,2,3**, four **A100-SXM4-80GB** cards, port **24631**. Supervisor PID at launch: **2440452**. The start manifest confirms **20,381,184 trainable parameters**, SDPA, FP32, the intended five identities/four levels, ranks `[2,1,0,0,3]`, both cache counts and the effective batch. This document records launch and initial verification, not training completion.

Runtime root: `results/rebuttal/2026-10-31-jdata-v3-hierarchy/`.

- `plan.json`: reproducible three-configuration plan; `full` is the natural V3 mapping and `weak_auxiliary` is numerically identical here. Neither reference was launched.
- `random-only-plan.json`: the selected experiment, used by the running pipeline.
- `pipeline.log`, `job.start.json`, `launch.json`: live log and process/provenance records. `job.finish.json` appears when the supervisor exits.
- `random_auxiliary_reversed/seed_42/checkpoint/`: selected final weights, trainer state, config and training manifest.
- `random_auxiliary_reversed/seed_42/validation-ranking.json`: penultimate-session ranking (`smb_explicit_valid`).
- `random_auxiliary_reversed/seed_42/test-ranking.json`: last-session ranking (`smb_explicit`).
- Same-stem result directories: per-user metrics, UID sidecars and original target exports.
- After successful completion: `verified-results.json` and `metrics.csv`, verifying root/best weight identity, split provenance, per-user means, counts, target-export binding, weighted merged values and unchanged cache hashes.

From the repository root, the exact pipeline command is:

```bash
python scripts/run_rebuttal_experiment.py \
  --plan ./results/rebuttal/2026-10-31-jdata-v3-hierarchy/random-only-plan.json \
  --variants random_auxiliary_reversed --seeds 42 \
  --gpu 0,1,2,3 --port 24631 --execute
```

It is already running; do not launch a duplicate. It selects final weights by **all-behavior validation loss**, then runs standalone validation ranking and test ranking automatically. Failures stop subsequent phases. The supervisor exports the final verified CSV; no result is claimed before those phases finish. The obsolete V2 draft at `results/rebuttal/2026-10-31-jdata-random/` was never launched.

## Remaining comparability limits

The historical command and method/result family now match, and local V3 semantics were verified against the historical implementation. The supplied archive does not establish the historical dataset/index hashes, exact checkpoint/config, environment, beam/evaluation batch or actual split execution. Verify these before calling any difference a strict prior-only effect against that historical score. No historical UID pairing or significance test is available. The experiment is a single training seed and one deliberately selected hierarchy permutation; buy ranking is not purchase-probability estimation.
