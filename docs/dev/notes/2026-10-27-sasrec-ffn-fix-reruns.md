# FFN correction and Tmall SASRec reruns

Chinese counterpart: `2026-10-27-sasrec-ffn-fix-reruns.zh.md`.

## Implementation and validation

Commit `d8d06c3` changes the shared `FeedForward.forward` condition from `if not self.residual` to `if self.residual`. Enabled residuals now apply dropout, input addition, and LayerNorm; disabled residuals return the projection without accessing absent modules. Constructor arguments, parameter shapes, and state-dict keys are unchanged. Old erroneous training is not repaired merely by reloading its weights under the corrected forward.

Four CPU regression tests passed: zero-projection residual preservation and module invocation, disabled-residual execution, SASRec normalization gradients/optimizer updates for atomic and expanded vocabularies, and forward/backward checks for PBAT/MB-STR FFN callers. Required flake8 (`--max-line-length=500 --ignore=F401,E203,W503,F841`), compileall for `main.py`/`SeqRec`, and `bash -n scripts/train_SMB_rec.sh` passed.

A synthetic single training step on idle GPU 5 used the actual expanded Tmall vocabulary, 1,447,728 item classes, batch 4096, history 20, FP32, and AdamW. Loss was finite (**14.21317482**), all FFN normalization gradients were finite/nonzero, and peak allocated memory was **69,731.25 MiB**. This is a capacity/function check, not a trained recommendation result or an efficiency benchmark.

The first real SASRec_B attempt subsequently failed with CUDA allocator fragmentation after its initial validation. The isolated single-step check had not covered that allocation sequence. A second GPU probe covered validation batches 4096/2821, training batches 4096/881, another validation, and another full training batch under `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. All steps passed with finite loss/gradients, peak allocated memory **71,154.40 MiB**, without changing batch size, model, or precision. The active SASRec_B retry uses this allocator setting; the failed attempt is preserved.

## Authorized reruns

The user requested prioritizing both Tmall SASRec reruns and postponing other protocol changes. Both runs therefore retain the recovered task identities, current source configuration, sampled-validation selection, and full-sort test workflow. Historical final hyperparameters remain unavailable; these are explicitly recorded fresh runs, not a proven one-variable reproduction of the old scores.

| Variant | Physical GPU | Training task | Test task | Item classes |
| --- | ---: | --- | --- | ---: |
| SASRec | 1 | `smb_dis_sample` | `smb_dis` | 361,932 |
| SASRec_B | 5 | `smb_dis_sample_diff` | `smb_dis_diff` | 1,447,728 |

Common settings: `Tmall-24-0.25` (not V2), seed 42, history 20, batch 4096 on one A100-SXM4-80GB per run, FP32, 200 maximum epochs, AdamW lr=0.001, weight decay=0.01, patience=20, 2 layers/2 heads/hidden 128/FFN 256/dropout 0.5/CE. No SID/CID index enters these discriminative models. `--add_uid` retains metadata for per-user exports; SASRec's loss/forward does not read UID. Validation uses alipay ranking and default NDCG@10 for selection. Final test evaluates the four behaviors and writes separate behavior rows plus merged results.

Confirmed dataset populations: 1,057,649 training samples, 6,917 validation alipay users, and 6,329 test alipay users. Separate atomic/diff caches and expected sorted test UID lists are retained locally.

Detached supervisors launched SASRec at **20:22:31** (worker PID 2592796) and the first SASRec_B attempt at **20:25:55** (worker PID 2595620), Asia/Shanghai on 2026-10-27. The failed SASRec_B attempt was retried at **20:29:46** (worker PID 2599878) under `sasrec_b/seed_42/retry-expandable/`. SASRec reached a saved best sampled-validation NDCG@10 checkpoint after epoch 3; no new full-test result is claimed yet.

## Cache preparation without changing sampling

The original cold validation builder reconstructs the entire 361,932-item candidate pool for every user and would take hours. Its newly started slow preparation process was stopped; no training process was stopped. A process-local preparation helper, retained with the runtime artifacts, maps sampled ranks into the same ascending dense-integer candidate pool. It leaves production dataset code unchanged and falls back to the original sampler outside its guarded case.

On CPython 3.12, 105 real-history/boundary cases verified identical sampled item IDs and identical post-sampling RNG state against the original method. All user histories satisfy the dense-integer small-exclusion guard. The sampler's seed, negatives, complete validation population, target encoding, and collation are retained. The helper does not change model code or persist monkeypatches into training processes. The normal task loads the resulting complete caches.

## Commands and local artifacts

Run root: `results/rebuttal/2026-10-27-sasrec-ffn-fixed/`. Config snapshots are in each variant's `seed_42/config/config.json`. The active SASRec run is `sasrec/seed_42/`; the active SASRec_B run is `sasrec_b/seed_42/retry-expandable/`. Each run has `run_manifest.start.json`, `train.log`, and outputs under `checkpoint/` and `results/`. `active-jobs.json` identifies current manifests. The supervisor writes `run_manifest.finish.json` when the complete train/test process exits. Training is detached from the chat process; a failed process records its exit status.

The exact commands are recorded in each start manifest. Equivalent root-directory commands, using the already prepared config snapshots and fresh output paths, are:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 WANDB_MODE=offline python main.py train_SMB_rec \
  --backbone SASRec --dataset Tmall-24-0.25 --data_path ./data \
  --base_model ./results/rebuttal/2026-10-27-sasrec-ffn-fixed/sasrec/seed_42/config \
  --output_dir ./results/rebuttal/2026-10-27-sasrec-ffn-fixed/sasrec/seed_42/checkpoint \
  --result_dir ./results/rebuttal/2026-10-27-sasrec-ffn-fixed/sasrec/seed_42/results \
  --tasks smb_dis_sample --test_task smb_dis --seed 42 --max_his_len 20 --add_uid \
  --epochs 200 --batch_size 4096 --learning_rate 0.001 --weight_decay 0.01 --optim adamw --patience 20

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py train_SMB_rec \
  --backbone SASRec --dataset Tmall-24-0.25 --data_path ./data \
  --base_model ./results/rebuttal/2026-10-27-sasrec-ffn-fixed/sasrec_b/seed_42/config \
  --output_dir ./results/rebuttal/2026-10-27-sasrec-ffn-fixed/sasrec_b/seed_42/retry-expandable/checkpoint \
  --result_dir ./results/rebuttal/2026-10-27-sasrec-ffn-fixed/sasrec_b/seed_42/retry-expandable/results \
  --tasks smb_dis_sample_diff --test_task smb_dis_diff --seed 42 --max_his_len 20 --add_uid \
  --epochs 200 --batch_size 4096 --learning_rate 0.001 --weight_decay 0.01 --optim adamw --patience 20
```

These paths are allocated to the running jobs: do not rerun these commands over them. Final JSON paths are `sasrec/seed_42/results/result-smb_dis.json` and `sasrec_b/seed_42/retry-expandable/results/result-smb_dis_diff.json` below the run root. Use the **Behavior alipay** row for the paper comparison. Runtime data, weights, logs, cache helpers, and manifests are not committed.

## Completed SASRec result

The ordinary SASRec process exited with code 0 at **2026-10-27 21:25:10 Asia/Shanghai**, including final full-sort test. Training stopped after epoch **37**; the saved best checkpoint was selected at epoch **17** by sampled-validation NDCG@10 (log-rounded **0.3514**, not a test score). The full train/validation/test process took 3,759.075 seconds. GPU 1 was released. At the subsequent check, SASRec_B was still training around epoch 37 and had no final test JSON.

Full-catalogue **test / Behavior alipay**, 6,329 users:

| Metric | Archived paper SASRec | Corrected fresh run | Relative increase |
| --- | ---: | ---: | ---: |
| HR@1 | 0.036182651 | 0.159424870 | +340.61% |
| HR@5 | 0.065887186 | 0.205719703 | +212.23% |
| HR@10 | 0.076789382 | 0.216621899 | +182.10% |
| Recall@10 | 0.072005407 | 0.202203450 | +180.82% |
| NDCG@5 | 0.049573397 | 0.174461469 | +251.93% |
| NDCG@10 | 0.053329522 | 0.178387853 | +234.50% |

HR@10 is **2.82099×** the archived value, an absolute increase of 0.139832517; NDCG@10 is **3.34501×**, an absolute increase of 0.125058330. Both remain below the archived TIGER row (0.580976458 / 0.468769537). Click, collect, and cart HR@10/NDCG@10 also increased; these auxiliary results are retained in the JSON.

Verified: successful exit, saved weights, unchanged source/config/raw-data/cache fingerprints, 6,329 finite per-user metric entries matching the aggregate means, and four learned FFN LayerNorm weight/bias tensors differing from initialization. Best-weight SHA256: `f2c79abe19f577f7fd5220457d8c26ae1cfb5dbd8c6ca8e5542ed0ed59e2810b`; result SHA256: `9c42b397f085f0baa180931e9863f31e9d221e3b0d4dd46f5d169997e81ca1f6`.

Evidence: `sasrec/seed_42/results/result-smb_dis.json`, its `result-smb_dis/user_level_metrics_behavior_alipay.json`, `run_manifest.finish.json`, and `verified-test-comparison.json` under the run root. The old reference is the matched `results_wzf.zip` artifact documented in `2026-10-27-tmall-baseline-provenance.md`.

This is a numerical comparison of the corrected fresh run against the paper's archived result. Missing historical hyperparameters/training revision prevent a strict FFN-only causal attribution; missing old UID-aligned metrics prevent a paired significance claim.

## Historical command recovered after the first rerun

The user subsequently supplied the original ordinary-SASRec command. It explicitly used history **50** and patience **10**, whereas the completed corrected run used **20 / 20**. The dataset, backbone, training/test tasks, batch 4096, lr=0.001, and maximum 200 epochs match. The current SASRec_B rerun also uses 20 / 20; the supplied historical command identifies only the ordinary SASRec run.

| Setting | Supplied historical command | Completed corrected run |
| --- | --- | --- |
| `max_his_len` | 50 | 20 |
| `patience` | 10 | 20 |
| `add_uid` | Absent / false | True, metadata only |
| FFN computation | Historical training checkout still unverified | Corrected in `d8d06c3` |

The config snapshot is byte-for-byte identical to the tracked SASRec config at `1c941cc` (2025-10-04), SHA256 `5d113d2c49cb06f47cdc72377f6c0b09ac6905af5e56901d363a920b87ee667a`. This supports the user's report that the config was unchanged, but is not a direct copy from the original machine. That historical parser confirms omitted defaults of seed 42, AdamW, weight decay 0.01, logging step 30, and the same metric list/default NDCG@10 selection.

History length changes both visible input and the learned position-embedding length, even when the JSON config is identical. Replaying this run's recorded best-update events with patience 10 would stop at epoch **27** and retain the same epoch-17 checkpoint; this is an inference about the observed history-20 trajectory, not a prediction for history 50.

Accordingly, **0.216621899 / 0.178387853 must remain labeled as the corrected history-20/patience-20 run**. A comparison matching the supplied historical recipe requires a fresh history-50/patience-10 training run; no such additional training was launched during this command audit. Original dataset/cache hashes, runtime environment, and exact training revision are still missing. Runtime evidence: `sasrec/seed_42/historical-command-comparison.json`.
