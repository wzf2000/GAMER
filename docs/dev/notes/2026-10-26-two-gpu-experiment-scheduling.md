# Scheduling rebuttal experiments on two A100s

Date: 2026-10-26. Code is the uncommitted experiment-support working tree after `38a90d2`. This note informs scheduling, not a publication efficiency claim.

## Resources and first experiments

The initial snapshot showed GPUs 1/5/6/7 without compute processes. Only the previously discussed GPUs 5/7, both A100-SXM4-80GB, were used for these probes.

Prefer two-GPU DDP with paired seed order:

```text
full42 → plain_matched42 → full43 → plain_matched43 → full44 → plain_matched44
```

This obtains the key capacity comparison before finishing all three full runs. Follow with one run each for no_qkv, no_gate, shared_ffn, and zero_prior. Raw comparisons await actual inputs; Tmall and lambda are later priorities; offline sparse buckets need no GPU.

`results/rebuttal/2026-10-26-p0/run_train_paired.sh` reuses the existing six commands, changing only queue order. It passed bash -n, preserves checkpoint overwrite guards, and has not been launched.

## Measurement method

- Configurations come from `results/rebuttal/2026-10-26-p0/configs/{full,plain_matched}/config.json`.
- FP32, TF32 off, SDPA, training mode/source dropout, 101 five-token item blocks plus EOS: length 506.
- Synthetic tokens and masked causal-LM labels, masking behavior tokens. No actual training samples, optimizer, or parameter updates.
- Single-GPU batches 32/64/128, one warmup plus three timed repetitions, CUDA synchronization, median reporting. Full and matched models used GPUs 5 and 7 respectively.
- Two-GPU DDP uses per-device batch 128, gradient all-reduce every step, and the slower rank's elapsed time; one warmup plus three measurements. Actual gradient accumulation reduces synchronization frequency.
- Timings include forward, LM loss, and backward; DDP includes gradient communication. They exclude optimizer steps, input loading, periodic validation, test, and saving.

## Measurements

| Model | Single-GPU batch | Median forward/backward | Peak allocated |
| --- | ---: | ---: | ---: |
| Full | 32 | 0.1881 s | 11.06 GiB |
| Full | 64 | 0.3370 s | 21.97 GiB |
| Full | 128 | 0.6387 s | 43.79 GiB |
| Matched decoder | 32 | 0.2536 s | 15.43 GiB |
| Matched decoder | 64 | 0.4968 s | 30.74 GiB |
| Matched decoder | 128 | 0.9724 s | 61.32 GiB |

At batch 128, peak allocator reservations were 49.96/68.37 GiB. Both fit; actual optimizer/data/full-workflow overhead remains separate.

| Model | DDP per-device batch | Two-GPU step | Throughput vs single GPU | Rank-0 peak allocated |
| --- | ---: | ---: | ---: | ---: |
| Full | 128 | 0.6439 s | 1.984× | 43.89 GiB |
| Matched decoder | 128 | 0.9736 s | 1.998× | 61.66 GiB |

Assuming equal sample counts for both groups, normalized by samples/128:

- One GPU per model concurrently: `max(0.6387, 0.9724) = 0.9724`.
- Two GPUs per model sequentially: `(0.6439 + 0.9736)/2 = 0.8088`.

The latter reduces paired compute waiting time by about 16.8% while retaining the prepared two-GPU setup. This is not a guaranteed full-workflow speedup.

Retain per-device batch 128, accumulation 8, effective batch 2048, FP32, SID `.index.json`, history 100, and seeds 42/43/44. Do not convert the historical 4×A800 22.57h or 8×H800 8.63h directly into a new full-run estimate.

Temporary raw records: `/tmp/gamer-full-capacity.json`, `/tmp/gamer-plain-capacity.json`, `/tmp/gamer-ddp-capacity.json`. Probe scripts: `/tmp/gamer_rebuttal_capacity_probe.py` and `/tmp/gamer_rebuttal_ddp_capacity_probe.py`. A formal efficiency study should preserve its scripts and complete conditions separately.
