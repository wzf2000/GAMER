# Interim no-QKV ranking on a frozen checkpoint

Chinese counterpart: `2026-10-27-no-qkv-interim-ranking.zh.md`.

## Scope and checkpoint selection

At 2026-10-27 18:52:51 Asia/Shanghai, the latest saved no-QKV training state was epoch 44, step 5148. Its best checkpoint by **all-behavior validation loss** was `checkpoint-4914`, epoch 42, loss **2.034342050552368**. Training continues; this evaluation is an interim observation of that selected checkpoint, not the final no-QKV result. Test ranking does not select the checkpoint.

Source: `results/rebuttal/2026-10-26-p0/no_qkv/seed_42/checkpoint/checkpoint-4914/`.

An independent inference snapshot copies weights, config, tokenizer, trainer state, training arguments, and the training start manifest. It does not copy optimizer/scheduler/RNG states. This protects inference from checkpoint rotation without changing training files. Copied weight SHA256 matches the source:

`bfff940b2ea499a8cc1d99843296dea790b7cefa1fc2a0a2406bed2007588f3a`.

## Evaluation protocol

- Dataset: ShortVideoAD; original SID `.index.json`, SHA256 `0e71e2064bf2c0f95e639e511f4c81a00ef839ec12cd1e2e7ad43e485ab6e62d`.
- Model: 29,922,816 trainable parameters; `th_use_behavior_qkv=false`; frozen soft prior lambda 0.05; attention gate, position-specific FFNs, and FFN behavior injection retained.
- GPUs: previously idle devices 1 and 5, two A100-SXM4-80GB; FP32, SDPA, per-device ranking batch 8, beam 20, seed 42, history limit 100.
- Validation ranking: `test_SMB_decoder --test_task smb_explicit_valid --behaviors cvr`; actual dataset mode `valid_test`.
- Test ranking: `test_SMB_decoder --test_task smb_explicit --behaviors cvr`; actual dataset mode `test`.
- Candidates: 168,530 indexed item IDs, 151,925 unique SID representations, with the cvr behavior prefix. This is next-item ranking conditioned on cvr, not conversion probability prediction.
- Per-user metrics, UID sidecars, and original target IDs are exported with `--export_user_targets`.
- Evaluation code revision: `e7e94f3e75dff8e7bfbd4c7264b31b9dc87aff3f`; exact commands and selection provenance are recorded in `evaluation.json` below. The copied training start manifest separately identifies the training checkout and its dirty state.

## Results

| Split | Users | HR@1 | HR@5 | HR@10 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 9,408 | 0.043579932 | 0.137223639 | 0.205888605 | 0.158593993 | 0.075447820 | 0.093055472 |
| Test | 8,784 | 0.042008197 | 0.132058288 | 0.198315118 | 0.151055865 | 0.070861593 | 0.088230915 |

Validation has 17,788 held-out cvr target events; test has 16,812. Both processes exited successfully. JSON checksums, actual split, model flags, unchanged weight/data/index fingerprints, per-user metric means, unique UIDs, and target alignment were independently verified for both splits. Neither split has a target with an ambiguous SID collision under the existing collision check.

Validation ran from 18:53:03 to 19:06:05 (781.385 seconds); test ran from 19:06:05 to 19:18:11 (726.654 seconds), all Asia/Shanghai on 2026-10-27. These wall times include loading, candidate construction, inference, and output export. They are not isolated model-inference benchmarks. GPUs 1 and 5 were released after completion.

At the post-evaluation check around 19:18, training had saved a **new** best checkpoint: epoch 45, step 5265, all-behavior validation loss **2.0321006774902344**. The table above still refers exclusively to the epoch-42 / step-4914 snapshot chosen at evaluation launch; no ranking result for step 5265 is claimed.

## Local artifacts

Root: `results/rebuttal/interim-ranking/no_qkv-seed42-step4914-20261027-185251/`.

- `evaluation.json`: source checkpoint, selection state, snapshot hashes, commands, and phase timestamps/exit codes.
- `verified-summary.json`: independently checked metrics, user/target-event counts, result checksums, and wall times.
- `checkpoint-snapshot/`: independent inference weights/config/tokenizer and training provenance.
- `validation-ranking.json`, `test-ranking.json`: split-specific results; each has a `.run.json` provenance sidecar.
- `validation-ranking/`, `test-ranking/`: per-user metrics, UID sidecars, and original target exports.
- `validation.log`, `test.log`: complete process output.

Generated artifacts remain local and are not committed. No training/model source code changed. A controlled Full ranking result is still needed before drawing a Full/no-QKV effectiveness conclusion; historical submission scores are not substituted for it.
