# ShortVideoAD full GAMER epoch timing on eight A100s

Chinese counterpart: `2026-10-31-shortvideoad-eight-a100-timing.zh.md`. Date: 2026-10-31, Asia/Shanghai.

## Accepted measurement and 94-epoch estimate

The author explicitly accepted the completed **first training epoch plus its Evaluation**, excluding the subsequent finalization wait, and supplied **94 total epochs** for extrapolation. No second full-data epoch was run.

| Component | Measured first epoch | Conditional estimate for 94 epochs |
| --- | ---: | ---: |
| Training, including optimizer/DDP and batch preparation | approximately 228 s | approximately 5h57m12s |
| All-behavior validation-loss Evaluation | 136.9415 s | approximately 3h34m33s |
| Training + Evaluation | approximately 365 s (6m05s) | **approximately 9.53 h (9h32m)** |

Calculation: `(228 + 136.9415) × 94 / 3600 = 9.529028... hours`, approximately **76.23 A100 GPU-hours** across eight cards. This is a linear estimate from one first epoch, not an observed 94-epoch run or a steady-state mean. Training elapsed is reconstructed from the native tqdm timer with whole-second precision; the arithmetic does not imply subsecond measurement precision. Initialization, checkpoint save/load, finalization and standalone ranking are excluded from this accepted scope.

## Actual recipe

The original command specified ShortVideoAD, `Qwen3TemporalHierarchical` with `config/s2s-models/Qwen3TemporalHierarchicalFixedSoft`, `smb_explicit_decoder_4`, original `.index.json`, history100, learning rate5e-4, maximum200 epochs, per-device batch128, accumulation8, warmup0.04, patience20 and temperature0.7. The author's earlier hardware record identifies that original run as four GPUs: effective batch `4×128×8=4096`.

The measurement uses **8×NVIDIA A100-SXM4-80GB**, GPU0–7 on this host, retaining microbatch128 and changing accumulation to **4**, so effective batch remains **4096**. Both training and Evaluation use per-device batch128. It retains seed42, maximum length1024, FP32 (`fp16=False`, `bf16=False`), SDPA, AdamW, cosine, weight decay0.01, all-behavior validation, full Q/K/V conditioning/gate/position experts and frozen prior lambda0.05. Actual trainable parameter count is **29,935,104**. Input cache counts are **238,270 training** and **193,316 validation** samples; original SID SHA256 is `0e71e2064bf2c0f95e639e511f4c81a00ef839ec12cd1e2e7ad43e485ab6e62d`.

The optional `--benchmark_epochs 1` keeps `--num_train_epochs 200` and stops only after the first complete epoch and scheduled Evaluation/save. Actual optimizer updates are **59**; the preserved scheduler budget is **11,600 steps**, warmup **464 steps**. The supplied 94-epoch count is used only for time extrapolation; the benchmark's learning-rate schedule was not changed to either one or94 planned epochs.

Environment: PyTorch2.7.1+cu118, compiled CUDA11.8, cuDNN90100, Transformers4.51.0, Accelerate1.8.1, DeepSpeed0.16.5, NVIDIA driver550.144.03. W&B was offline, OMP/MKL/OpenBLAS threads were1, and `CUDA_LAUNCH_BLOCKING` was unset. Hardware/topology and exact environment/command were recorded alongside the artifacts.

## Evidence and finalization issue

Runtime root: `results/rebuttal/2026-10-31-shortvideoad-8xa100-epoch/`.

- Native progress before Evaluation: `59/11600 [03:48<...]`.
- Native Evaluation log at **21:42:36**: `eval_runtime=136.9415`, `epoch=1.0`; cumulative progress afterward: `59/11600 [06:05<...]`.
- `checkpoint/checkpoint-59/trainer_state.json` confirms epoch1, global_step59, planned max_steps11600 and a completed validation-loss entry.
- `accepted-epoch-summary.json` and `accepted-94-epoch-estimate.csv` contain the accepted phase measurement, conditional94-epoch estimate, command/config/cache/index bindings and explicit precision/scope limits.
- `pipeline.log`, `checkpoint/run_manifest.start.json`, `hardware-during-training.json` and `library-runtime.json` provide the raw evidence.

Training, Evaluation and the first checkpoint completed, but the original process stalled during post-checkpoint finalization and was interrupted. The original `job.finish.json` retains its nonzero return code; it must not be relabeled as a gracefully completed full pipeline. The accepted timing excludes that wait and does not depend on successful final root export.

A controlled tiny-model eight-GPU test reproduced a Transformers4.51 first-checkpoint race: before synchronization, rank0 had the new best checkpoint path while ranks1–7 had `None`. Immediate first-epoch stopping then risks different load-best barrier/store-FLOPs paths. The opt-in timing callback now synchronizes checkpoint completion and the best-checkpoint path before final loading. Eight ranks completed the forced delayed-save regression successfully; see `checkpoint-race-verification.json` and `smoke-checkpoint-sync.log`. Four CPU callback tests, including a real Trainer preserving the200-epoch schedule, passed; prescribed flake8 and compileall passed. Default training without `--benchmark_epochs` does not instantiate the callback.

The successful tiny regression validates the fix, but is not used to estimate full-model runtime. At the author's request, the full-data measurement was not repeated. The first full run used timing-feature commit `ccc09b2`; later synchronization fixes do not change the accepted raw first-epoch log.
