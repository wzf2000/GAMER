# Final P0 test ranking on ShortVideoAD

Chinese counterpart: `2026-10-28-p0-final-test.zh.md`. This evaluation follows the user's request to test the completed runs and reuse previous tests of identical best checkpoints. No training was launched and no model, dataset or scoring implementation was modified.

## Checkpoint selection and comparability

All runs use ShortVideoAD, original SID `.index.json`, seed 42, `smb_explicit_decoder_4`, and the recorded effective training batch 2048. Training manifests identify the training checkout and its dirty-workspace flag. Model selection uses **all-behavior validation loss**, not test ranking.

| Variant | Parameters | Best epoch / step | Best validation loss | Final epoch |
| --- | ---: | --- | ---: | ---: |
| Full / TH-FixedSoft | 29,935,104 | 54 / 6318 | 2.0361568927764893 | 74 |
| Equal-capacity ordinary decoder (`plain_matched`) | 29,935,616 | 61 / 7137 | 2.01444149017334 | 81 |
| no-QKV (`no_qkv`) | 29,922,816 | 53 / 6201 | 2.029874086380005 | 73 |

The final root `model.safetensors` files match the selected best-checkpoint weights byte-for-byte. SHA256 values:

- Full: `90da57d7e077dbbce5d3654ab2bffc06504b4fa3a4ead096a84cb3eef0433a77`.
- Ordinary decoder: `5faaade154a43edf21070fbcf51f51538a19669a7bb3e47bc643475da67e9287`.
- no-QKV: `f6071ceeedbf699e814b7c8520f8bcf0633c657b6d4ddf610f62ace8221b054f`.

Training data/validation cache fingerprints and current data/index snapshots were checked against the common plan. Saved tokenizer/generation files and principal effective training arguments also match across all three after normalizing the relative/absolute data-path spelling; `verified-training-comparability.json` records this audit. The SID index SHA256 is `0e71e2064bf2c0f95e639e511f4c81a00ef839ec12cd1e2e7ad43e485ab6e62d`.

## Evaluation and reuse

Task: `test_SMB_decoder --test_task smb_explicit`, actual mode **test**; history limit 100, beam 20, FP32, SDPA, and unchanged behavior-specific candidate-prefix construction. Each new evaluation uses two A100-SXM4-80GB GPUs and per-device batch 64. `--export_user_targets` preserves UID-aligned metrics and original held-out target IDs. Test counts are **cvr 8,784**, **click 11,444**, **p3s 43,618**. Merged metrics weight the 63,846 behavior-user instances; this is not a distinct-user count. Cvr metrics describe behavior-conditioned next-item ranking.

Full runs on GPUs 2/3 and no-QKV on 6/7, evaluating cvr first. Full had no prior test; no-QKV's previous interim test used step 4914, so it cannot replace evaluation of final step 6201.

The ordinary decoder's final best step 7137 is identical to its earlier interim test. Model weights, complete config, tokenizer files, generation config, training manifest, data/index and the relevant scoring code were verified unchanged. Its earlier **test/cvr** result is therefore reused from `results/rebuttal/interim-ranking/plain_matched-seed42-step7137-20261027-192017/`, where inference used two GPUs and batch **8** per device. The original cvr row and all per-user/UID/target exports are retained. The previously reported cvr-only merged row is not reused as an all-behavior merged value.

At the user's explicit reuse instruction, the newly launched redundant ordinary-decoder test was stopped gracefully through its torchrun parent, PID 3321407; logs and nonzero cancellation receipts were retained. It was superseded rather than failing during training. Only click/p3s are newly evaluated on GPUs 4/5, under `plain_matched/remaining/`; their aggregate is `test-remaining-ranking.json`. The final ordinary-decoder result assembles the verified cvr row with these two rows and recomputes merged metrics. Its provenance records per-behavior source files and batch sizes (8 for reused cvr, 64 for new click/p3s), rather than asserting a uniform batch.

## Results

All final test results are complete. Full finished at **2026-10-28 15:19:41**, no-QKV at **2026-10-28 15:18:59**, and the ordinary decoder's new click/p3s phase at **2026-10-28 15:22:23**, all Asia/Shanghai. Each successful final evaluation process exited with code 0; the earlier redundant ordinary cvr attempt was explicitly cancelled for reuse as recorded above. The same **8,784 UIDs and 16,812 held-out target events** were verified across all three, together with metric/UID/target checksums, finite metric arrays and per-user means. The ordinary-decoder row is reused as documented above.

| Variant | HR@1 | HR@5 | HR@10 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 0.039845173 | 0.128756831 | 0.198542805 | 0.152507869 | 0.069265608 | 0.087807118 |
| Equal-capacity ordinary decoder | 0.040983607 | 0.135587432 | 0.197973588 | 0.149450960 | 0.072071882 | 0.087893460 |
| no-QKV | 0.041666667 | 0.131944444 | 0.203551913 | 0.154267208 | 0.071500847 | 0.089708056 |

Full and the ordinary decoder are close on HR@10/NDCG@10, with opposite signed differences; no-QKV has higher HR@10 and NDCG@10 in this seed. No test-based checkpoint reselection was performed. This is one seed, not a multi-seed uncertainty or significance result. `verified-cvr-summary.json` binds these values to the source exports and selected weights. The complete auxiliary and merged results are:

| Behavior | Full: HR@10 / NDCG@10 | Ordinary decoder: HR@10 / NDCG@10 | no-QKV: HR@10 / NDCG@10 |
| --- | --- | --- | --- |
| click | 0.160608179 / 0.068894657 | 0.162617966 / 0.069915867 | 0.165152045 / 0.070905045 |
| p3s | 0.223783759 / 0.072082603 | 0.227887569 / 0.073614020 | 0.226947590 / 0.074014237 |
| Merged | 0.208987251 / 0.073674579 | 0.212072800 / 0.074915730 | 0.212652320 / 0.075616105 |

Full is slightly above the ordinary decoder on cvr HR@10 (+0.000569217) and slightly below on cvr NDCG@10 (-0.000086342). No-QKV is higher than Full on those two cvr metrics (+0.005009107 / +0.001900938) and has the highest merged HR@10/NDCG@10 of these three runs. The ordinary decoder has the highest p3s HR@10. These are descriptive single-seed results; they do not establish component gains or multi-seed significance.

`verified-summary.json` verifies successful final phase receipts, immutable selected weights/configs, training/data/index provenance, matching training/validation caches, the actual test split, identical behavior-specific UID/target identities, all eight finite per-user metric arrays and aggregate means, and the merged weighted means. The existing `summarize_rebuttal_results.py` also generated complete cvr and merged summaries (`partial=False`). The full `test-metrics.csv` has 96 rows: 3 variants × 4 behavior/merged rows × 8 metrics. Reused ordinary cvr values exactly match the previous result. No further ranking or training was run to select a more favorable test score.

## Local artifacts

Runtime root: `results/rebuttal/2026-10-28-final-test/`. It contains the original test plan, the remaining-behavior plan for ordinary decoder reuse, supervised process records/logs, and reuse decisions. The final result paths are:

- `results/rebuttal/2026-10-26-p0/full/seed_42/test-ranking.json`.
- `results/rebuttal/2026-10-26-p0/plain_matched/seed_42/test-ranking.json`.
- `results/rebuttal/2026-10-26-p0/no_qkv/seed_42/test-ranking.json`.

Each result has `.run.json` provenance and a same-stem directory with per-user metrics, UID sidecars and target exports. Runtime preparation/supervision/assembly/verification helpers remain ignored artifacts and pass the prescribed flake8/compileall checks. Generated weights, datasets, logs and results are not committed. Wall time includes loading, candidate construction and exports, and the reused ordinary cvr component is from an earlier run; it is not an inference-efficiency comparison.
