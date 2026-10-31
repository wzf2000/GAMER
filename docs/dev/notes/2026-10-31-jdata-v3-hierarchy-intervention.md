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

The sole authorized variant started at **2026-10-31 19:59:13**, on GPU **0,1,2,3**, four **A100-SXM4-80GB** cards, port **24631**. Supervisor PID at launch: **2440452**. The start manifest confirms **20,381,184 trainable parameters**, SDPA, FP32, the intended five identities/four levels, ranks `[2,1,0,0,3]`, both cache counts and the effective batch. Completion and final results are recorded below.

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

This pipeline has completed; do not rerun it into the existing output directory. It selected final weights by **all-behavior validation loss**, then ran standalone validation ranking and test ranking automatically. The obsolete V2 draft at `results/rebuttal/2026-10-31-jdata-random/` was never launched.

## Completed test and main-method comparison

Training, validation ranking and test ranking all returned **0**. Training ended at **2026-10-31 21:12:22** after **35 epochs / 455 steps**. The selected checkpoint is **epoch15 / step195**, all-behavior validation loss **2.279292345**. Root weights match `checkpoint-195` byte-for-byte, SHA256 `328b51cabd9f0ef1fc0401a0987a40e7b06c6499e223f87bd489f964668a195b`. Final test/verification completed at **21:17:03**; GPU0–3 are released.

Training including periodic validation took **4,389.52 s (1h13m09.52s)**; standalone all-five-behavior validation ranking took **141.98 s**, test ranking **137.49 s**. The complete supervised process took approximately **1h17m50s**. These wall times include setup/export overhead and are not isolated training/inference efficiency measurements. The author subsequently supplied the historical V3 W&B final log: it also stopped at epoch35, with `train_runtime=18325.9655s` and final validation loss2.436602116. That excerpt does not recover the historical best epoch or checkpoint; do not substitute the new best epoch15 for the old unknown value.

**Test / buy, 230 users:** the reference below is the author-supplied JData-V3 main-method archive, not validation.

| Metric | Archived main method | Prescribed Random | Random minus main |
| --- | ---: | ---: | ---: |
| HR@1 | 0.395652174 | 0.395652174 | 0 |
| HR@5 | 0.652173913 | 0.656521739 | +0.004347826 |
| HR@10 | 0.726086957 | 0.726086957 | 0 |
| Recall@1 | 0.385403727 | 0.385403727 | 0 |
| Recall@5 | 0.633022774 | 0.637370600 | +0.004347826 |
| Recall@10 | 0.709834369 | 0.709834369 | 0 |
| NDCG@5 | 0.525383480 | 0.527333302 | +0.001949822 |
| NDCG@10 | 0.550338765 | 0.550869572 | +0.000530807 |

HR@10 is **167/230** in both runs; HR@5 is **150/230 → 151/230**. NDCG@10 increases **0.09645% relative**, or **0.05308 percentage points**. These point estimates are essentially unchanged and do not show a buy-ranking advantage for the natural auxiliary ordering over this prescribed reversal. This is not a significance or general hierarchy-necessity conclusion.

The completed checkpoint was independently loaded on CPU: all four TH layers resolve the requested lookup `[0,3,2,1,1,4]`. Replacing that lookup with the natural mapping in memory, while retaining the exact same weights, changes logits on a constructed five-behavior input (maximum absolute change **0.01322937**). The intervention is therefore active. Buy remains the highest level in both maps, so its query row has **zero direct soft-prior bias to every behavior** in both cases. Changes reach buy prediction through historical representations and training; this is narrower than scrambling all hierarchy relations or removing behavior conditioning.

Independent verification rechecked phase receipts, actual `test` metadata, root/best weights, unchanged raw/index/cache/archive hashes, all five per-user metric means/UID/target-export bindings, and merged weighted values for validation and test. The audit script passed prescribed flake8 and compileall; no production source or checkpoint was changed. Provenance and full eight-metric comparison: `results/rebuttal/2026-10-31-jdata-v3-hierarchy/verified-test-buy-comparison.json` and `test-buy-comparison.csv`. Actual test result: `random_auxiliary_reversed/seed_42/test-ranking.json` under the same root. Complete validation/test metrics are in `metrics.csv`.

## Remaining comparability limits

The historical command and method/result family now match, and local V3 semantics were verified against the historical implementation. The supplied archive does not establish the historical dataset/index hashes, exact checkpoint/config, environment, beam/evaluation batch or actual split execution. Verify these before calling any difference a strict prior-only effect against that historical score. No historical UID pairing or significance test is available. The experiment is a single training seed and one deliberately selected hierarchy permutation; buy ranking is not purchase-probability estimation.
