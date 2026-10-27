# Interim parameter-matched decoder test ranking

Chinese counterpart: `2026-10-27-plain-matched-interim-ranking.zh.md`.

## Checkpoint and scope

At 2026-10-27 19:20:17 Asia/Shanghai, the parameter-matched plain decoder's latest saved state and best checkpoint by **all-behavior validation loss** were epoch 61, step 7137, loss **2.01444149017334**. An independent inference snapshot was copied from `results/rebuttal/2026-10-26-p0/plain_matched/seed_42/checkpoint/checkpoint-7137/`. Weight SHA256 was verified against the source:

`5faaade154a43edf21070fbcf51f51538a19669a7bb3e47bc643475da67e9287`.

Training continues. The same checkpoint was still the best saved checkpoint at the post-evaluation check around 19:33. This is an interim test-ranking result selected by validation loss, not by test metrics. No training files or model source code changed.

## Verified configuration

- Plain decoder: 29,935,616 trainable parameters; 8 layers, hidden size 256, shared FFN width 3501; temporal-hierarchical, sparse-FFN, and FFN-injection layer lists are empty; behavior injection is disabled. Serialized tensors contain no TH-specific behavior embeddings, relation tables, attention output gates, or FFN expert banks. Standard FFN gate/up/down projections remain.
- ShortVideoAD, original SID `.index.json`, history limit 100, seed 42. Index SHA256: `0e71e2064bf2c0f95e639e511f4c81a00ef839ec12cd1e2e7ad43e485ab6e62d`.
- Previously idle GPUs 1 and 5, 2×A100-SXM4-80GB; FP32, SDPA, per-device ranking batch 8, beam 20. Both this model and the earlier no-QKV snapshot use generation caching.
- `test_SMB_decoder --test_task smb_explicit --behaviors cvr --export_user_targets`; actual dataset mode `test`. Candidates use the same cvr-prefixed SID index as the no-QKV evaluation.
- Evaluation checkout: `7c0c6fb`; its only changes since the preceding no-QKV evaluation checkout `e7e94f3` are the no-QKV experiment documents.

## Test results

Both evaluations use **8,784 identical users and 16,812 identical held-out cvr target events**, including matching original target IDs and SID representations. These are cvr-conditioned next-item ranking metrics.

| Snapshot | HR@1 | HR@5 | HR@10 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Plain matched, epoch 61 / step 7137 | 0.040983607 | 0.135587432 | 0.197973588 | 0.149450960 | 0.072071882 | 0.087893460 |
| Earlier no-QKV, epoch 42 / step 4914 | 0.042008197 | 0.132058288 | 0.198315118 | 0.151055865 | 0.070861593 | 0.088230915 |

Plain-minus-no-QKV differences: HR@10 **-0.000341530**, NDCG@10 **-0.000337454**. The two interim snapshots have different training progress; this descriptive comparison is not a final Full/plain ablation or a multi-seed conclusion. See `2026-10-27-no-qkv-interim-ranking.md` for the earlier no-QKV evaluation.

The process exited successfully. Result/model/index/data checksums, actual split, per-user metric means, UID uniqueness, target alignment, and matching no-QKV evaluation settings were independently verified. The existing collision check found zero ambiguous-SID target samples.

Wall time was **776.889 seconds**, 19:20:26–19:33:23 Asia/Shanghai on 2026-10-27, including loading, candidate construction, inference, and output export. This is not an isolated inference-efficiency benchmark. GPUs 1 and 5 were released after evaluation.

## Local artifacts

Root: `results/rebuttal/interim-ranking/plain_matched-seed42-step7137-20261027-192017/`.

- `evaluation.json`: checkpoint selection, snapshot hashes, exact command, timestamps, and exit code.
- `verified-summary.json`: complete metrics, checked counts/checksums, and descriptive differences from the earlier no-QKV snapshot.
- `checkpoint-snapshot/`: independent inference weights/config/tokenizer and copied training provenance; optimizer/scheduler/RNG states are excluded.
- `test-ranking.json`, `test-ranking.run.json`, `test.log`: results, actual evaluation provenance, and complete process output.
- `test-ranking/`: per-user metrics, UID sidecar, and original target exports.

Generated runtime artifacts remain local and are not committed.
