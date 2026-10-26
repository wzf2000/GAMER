# Run the FixedSoft rebuttal experiments on another machine

Date: 2026-10-27. Chinese counterpart: `.zh.md`. Published branch: `codex/rebuttal-fixedsoft-experiments-20261027`.

The remote `dev` contains the separate V3 behavior-hierarchy change `eadc431`. This experiment branch preserves the tested FixedSoft implementation used by the current ShortVideoAD jobs. Use this branch consistently for the new controlled series. No branch switch, remote history rewrite or V3 integration was performed in the active workspace.

The functional commit is `aaf63c2`. A clean export of that commit passed all 23 focused CPU tests, the prescribed flake8 check on 30 Python files and compileall, using only exported source/configuration/Toy data rather than ignored runtime files. Production GPU training on another machine has not been measured by this check.

## Obtain code and inputs

From the parent directory of the new checkout:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --single-branch \
  --branch codex/rebuttal-fixedsoft-experiments-20261027 \
  git@github.com:wzf2000/GAMER.git GAMER
cd GAMER
git lfs pull --include='data/ShortVideoAD/**' --exclude=''
```

Git LFS must already be available for the last command. The ShortVideoAD interaction, behavior, session, time and SID-index JSONs are LFS objects; a pointer-only checkout is insufficient. Existing behavior-level/item metadata and the FixedSoft tokenizer assets are tracked; tokenizer assets use repository-relative symlinks, so use a checkout with working symlink support. If LFS access is unavailable, copy the exact corresponding processed input files from an authorized source instead. Do not substitute reconstructed raw logs or another SID index.

`results/`, checkpoints, logs and `.pkl` caches are not part of this publication. Existing completed weights must be transferred separately when evaluating them. A new cache build uses the existing factor4/data-seed semantics, while the model seed is restored before initialization. For exact cross-machine comparability, record content SHA256 for inputs and, if transferring caches, the actual cache files. Current lightweight data snapshots use absolute paths/mtime and therefore cannot prove equivalence across machines by themselves.

Use the repository's existing installation instructions for dependencies; no installation is part of preparing a plan. The tested environment here was Python 3.12.13, PyTorch 2.7.1+cu118, Transformers 4.51.0, Accelerate 1.8.1 and DeepSpeed 0.16.5. Align the target machine's CUDA/PyTorch environment before comparing efficiency; backend defaults and hardware can differ.

## Regenerate a local plan

Run from the repository root with the target machine's GAMER Python environment active. This command only generates configuration/commands:

```bash
python scripts/prepare_rebuttal_experiments.py \
  --suite handoff --seeds 42 \
  --dataset ShortVideoAD --data_path ./data --index_file .index.json \
  --base_model ./config/s2s-models/Qwen3TemporalHierarchicalFixedSoft \
  --tasks smb_explicit_decoder_4 --max_his_len 100 --model_max_length 1024 \
  --gpu 0,1 --port 2314 --precision fp32 \
  --per_device_train_batch_size 128 --per_device_eval_batch_size 128 \
  --gradient_accumulation_steps 8 --num_train_epochs 200 \
  --learning_rate 5e-4 --weight_decay 0.01 --warmup_ratio 0.04 \
  --patience 20 --temperature 0.7 --test_batch_size 64 --num_beams 20 \
  --eval_behavior cvr \
  --output_dir ./results/rebuttal/2026-10-27-p0-local
```

GPU 0/1 and port 2314 are examples to replace with available target resources. The recipe has effective batch `128 × 2 × 8 = 2048`, FP32 and all-behavior validation-loss checkpoint selection, matching the current controlled series. Final ranking commands use separate validation/test tasks and cvr metrics; they are not conversion-probability prediction. This recipe is a documented new baseline, not a recovered submission command.

The generated plan is local: it records the local Python executable, checkout/data/config paths and fingerprints. Do not execute a plan or shell file copied unchanged from this server. Use `--reuse_plans` only for compatible plans on the same local paths/revision, not as an automatic cross-machine importer. Reuse of the already-running Full/no-QKV requires independently transferred and verified artifacts; the fresh seven-run plan does not claim they were completed locally. Allocate distinct configurations across machines to avoid duplicate work.

## Inspect and deliberately start one job

```bash
python scripts/manage_rebuttal_queue.py \
  --plan ./results/rebuttal/2026-10-27-p0-local/plan.json \
  --gpu 0,1 --port 2314 \
  --output_file ./results/rebuttal/2026-10-27-p0-local/status-001.json \
  --output_script ./results/rebuttal/2026-10-27-p0-local/next-001.sh
```

This refreshes state and prints/writes one command; it does not launch training. After checking the selected variant and free GPUs, execute the generated `next-001.sh` when ready. Use new snapshot/script names for later checks because existing outputs are never overwritten. Stage scripts also default to printing commands. Start manifests and epoch checkpoints alone are not treated as successful completion.

The handoff plan has seven unique configurations, reusing Full as lambda 0.05 and zero-prior as lambda 0. First finish P0-A (Full/no-QKV/no-gate/shared-FFN), then P0-B (0/0.01/0.1). The already-running capacity control remains a separate optional experiment. Full training and large ranking require the target machine's authorized resource allocation.

For parameter probes, UID/target exports, CSV results and scope limits, see [delivery tools](../design/2026-10-27-rebuttal-delivery-tools.md).
