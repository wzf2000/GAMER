# Startup audit of the two P0 seed42 runs

Checked at 2026-10-26 12:00 Asia/Shanghai. This verifies setup and early training activity, not convergence or completed validation.

## Outcome

Both parameter/data/output configurations match the prepared plan and training is advancing. Effective HF training arguments match after excluding output/logging/run-name paths. No unknown/unused CLI arguments, OOM, NaN/Inf, or NCCL errors were found.

| Field | FixedSoft full | Matched decoder |
| --- | --- | --- |
| GPUs / DDP port | 1/5, 2314 | 2/6, 2315 |
| DDP world size | 2 | 2 |
| Trainable parameters | 29,935,104 | 29,935,616 |
| Hidden / layers / vocabulary | 256 / 8 / 23675 | 256 / 8 / 23675 |
| FFN intermediate size | 512 | 3501 |
| TH attention layers | [2,3,4,5] | [] |
| Position-expert layers | [0,1,2,3,4,5,6,7] | [] |
| FFN behavior-injection layers | [0,1,2,3,4,5] | [], injection=false |
| Dropout / attention dropout | 0.2 / 0.2 | 0.2 / 0.2 |
| Latest progress | 49/23200 optimizer steps | 33/23200 optimizer steps |
| First record, step 30 | loss=10.1334, grad_norm=0.46155 | loss=10.1313, grad_norm=0.37762 |
| First recorded LR / epoch | 1.5625e-5 / 0.26 | 1.5625e-5 / 0.26 |

Counts differ by 512, approximately 0.00171%. Full retains the frozen soft table with lambda=0.05, default runtime multiplier 1, behavior Q/K/V, and output gate. The matched config inherits some prior fields but instantiates no TH attention layers, so those conditioning components are inactive.

## Shared inputs and training settings

- `ShortVideoAD`, `.index.json`, `smb_explicit_decoder_4`, history window 100, model_max_length=1024, seed=42.
- Source-config hashes match the corresponding plan entries. Input snapshots match the plan and each other.
- Train cache: `data/ShortVideoAD/ShortVideoAD.SMBExplicitDatasetForDecoder.100.SMB.aug4.train.pkl`, 238270 samples; validation: `data/ShortVideoAD/ShortVideoAD.SMBExplicitDataset.100.SMB.valid.pkl`, 193316 samples. Both jobs record identical paths, sizes, mtimes, and counts.
- Actual augmentation comes from the random4 cache. Parser defaults `sequence_augmentation=time_decay` and `augmentation_views=1` do not activate the policy branch for this task.
- Effective per-device train/eval batches are 128, accumulation 8, effective training batch 2048. Legacy `per_device_batch_size=256` did not override explicit HF batch arguments.
- FP32, SDPA, AdamW, LR=5e-4, warmup=0.04, cosine, weight_decay=0.01, loss temperature=0.7, up to 200 epochs, patience=20.
- Validation/saving occurs per epoch, selecting best by all-behavior validation loss. First validation has not occurred. Config/tokenizer/start manifests exist; epoch checkpoints are not expected yet.

## Evidence and outputs

- Full log: `logs/train_SMB_decoder/20261026_115245.log`, first loss at line 305.
- Decoder log: `logs/train_SMB_decoder/20261026_115328.log`, first loss at line 122.
- Full output: `results/rebuttal/2026-10-26-p0/full/seed_42/checkpoint/`.
- Decoder output: `results/rebuttal/2026-10-26-p0/plain_matched/seed_42/checkpoint/`.
- Each output's `run_manifest.start.json` records effective model/HF arguments and data/cache fingerprints. Old checkpoint paths were not used.
- The four workers' CUDA_VISIBLE_DEVICES, ranks, world sizes, and master ports confirm two independent two-GPU DDP groups.
- Code is `38a90d2` plus uncommitted changes. Tracked diffs and 23 changed/new source files, about 190KB, were saved under `results/rebuttal/2026-10-26-p0/code_snapshot/`. Running jobs were not changed or restarted.

Progress-bar ETAs exclude the complete periodic-validation overhead; no final duration is inferred. Assess results using validation and fixed-setting test cvr metrics rather than initial training loss.
