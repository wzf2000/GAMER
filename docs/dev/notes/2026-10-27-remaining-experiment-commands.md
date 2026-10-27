# One-command pipelines for the remaining experiments

Date: 2026-10-27. Chinese counterpart: `.zh.md`. Use the latest `codex/rebuttal-fixedsoft-experiments-20261027` branch. All training commands below are for deliberate execution by the user on the target machine; no full training was launched while preparing this runbook.

## Status and order

Full, no-QKV and parameter-matched plain_matched seed 42 are already running on the original server and have not saved final root weights. Do not submit them again by default. The target machine first supplies no-gate/shared-FFN, then lambda 0.01/0.1.

Lambda 0 in the complete architecture is without prior bias. The historical FixedBias config freezes a zero-initialized table; its soft_scale=0.1 is inactive under zero initialization. It is computationally equivalent to FixedSoft soft_scale=0. The old `results/ShortVideoAD/smb_explicit_decoder_4/Qwen3TemporalHierarchicalFixedBias/results-smb_explicit-original.json` contains cvr HR@10=0.1974043716 and NDCG@10=0.0872569724. It lacks a run receipt/final training manifest/checkpoint, so its filename alone does not prove the actual historical split. Historical FixedSoft values are 0.1980874317/0.0899807344; these are also distinct from the currently running new Full. Treat lambda 0 as a pending reuse audit, not an automatic retraining job. Use the fallback command only if data/SID/views/budget/seed/selection cannot be matched.

Priority is P0-A components, P0-B new lambda points, Tmall prior-order, key seeds, conditional core-only, then optional capacity seeds. Extra component seeds are conditional on the intended claim; sparse diagnostics remain future work.

## 0. Prepare the target machine once

Use Bash, activate the GAMER Python environment and run from the repository root. See [machine setup](2026-10-27-rebuttal-other-machine.md) for cloning. Obtain the ShortVideoAD LFS inputs:

```bash
git lfs pull --include='data/ShortVideoAD/**' --exclude=''
```

Generate local configurations only. Examples assume two 80GB GPUs 0/1 and port 2314. Training per-device batch=128 and accumulation=8 give effective batch=2048, with FP32, history=100, model length=1024, random factor4 and seed 42. Smaller-memory devices need a separately adjusted microbatch/accumulation preserving the effective batch. Ranking uses per-device batch=8, beam=20 to reduce inference memory pressure; this is not a measured GPU-throughput guarantee. Later evaluation of the server baselines should use the same ranking batch.

```bash
export WANDB_MODE=offline
rebuttal_common=(
  --base_model ./config/s2s-models/Qwen3TemporalHierarchicalFixedSoft
  --data_path ./data --tasks smb_explicit_decoder_4
  --max_his_len 100 --model_max_length 1024
  --gpu 0,1 --port 2314 --precision fp32
  --per_device_train_batch_size 128 --per_device_eval_batch_size 128
  --gradient_accumulation_steps 8 --num_train_epochs 200
  --learning_rate 5e-4 --weight_decay 0.01 --warmup_ratio 0.04
  --patience 20 --temperature 0.7 --test_batch_size 8 --num_beams 20
)
python scripts/prepare_rebuttal_experiments.py \
  "${rebuttal_common[@]}" --suite handoff --seeds 42 \
  --dataset ShortVideoAD --index_file .index.json --eval_behavior cvr \
  --output_dir ./results/rebuttal/remaining-p0
python scripts/prepare_rebuttal_experiments.py \
  "${rebuttal_common[@]}" --suite components --seeds 43 44 --export_user_targets \
  --dataset ShortVideoAD --index_file .index.json --eval_behavior cvr \
  --output_dir ./results/rebuttal/remaining-seeds
```

Offline W&B avoids interactive login on a new machine without changing the training objective; use online if already configured. Later commands assume the same Bash session; restore this environment variable when opening another terminal.

Warm cold caches once in a single process before launching distributed jobs. This reads/processes inputs and writes several GB of caches but does not train a model; provide sufficient RAM/disk/time for input preparation:

```bash
python - <<'PY'
from SeqRec.datasets.loaders.session_behavior import load_SMB_datasets, load_SMB_test_dataset
settings = dict(dataset='ShortVideoAD', data_path='./data', max_his_len=100, index_file='.index.json')
load_SMB_datasets(**settings, tasks='smb_explicit_decoder_4', train_session=True)
for task in ('smb_explicit_valid', 'smb_explicit'):
    load_SMB_test_dataset(**settings, test_task=task)
PY
```

The plans contain candidates, not automatically scheduled runs. The runner executes only the selected entries. Remove `--execute` to validate/print commands. Each entry runs training, validation ranking and test ranking in order; training selects the best model by validation loss. Existing outputs are refused, and failure stops later phases/jobs. It neither infers other machines' state nor automatically enforces global stage completion.

For one GPU regenerate the plan with `--gpu 0 --gradient_accumulation_steps 16`; for four use `--gpu 0,1,2,3 --gradient_accumulation_steps 4`, preserving effective batch 2048. Do not change only the training GPU count at execution. Same-priority jobs can run on distinct GPU pairs and ports after cache warming; allocate each configuration to only one machine.

## 1. P0-A: no-gate and shared-FFN

```bash
python scripts/run_rebuttal_experiment.py \
  --plan ./results/rebuttal/remaining-p0/plan.json \
  --variants no_gate shared_ffn --seeds 42 --execute
```

This runs two new training jobs, each followed by validation/test. Reuse the server's Full/no-QKV jobs.

## 2. P0-B: lambda 0.01 and 0.1

After the component stage:

```bash
python scripts/run_rebuttal_experiment.py \
  --plan ./results/rebuttal/remaining-p0/plan.json \
  --variants lambda_0.01 lambda_0.1 --seeds 42 --execute
```

Lambda 0.05 is the existing Full. Only if historical FixedBias cannot be reused under matching conditions, run the lambda-0 fallback:

```bash
python scripts/run_rebuttal_experiment.py \
  --plan ./results/rebuttal/remaining-p0/plan.json \
  --variants zero_prior --seeds 42 --execute
```

## 3. P1: three Tmall prior-order runs

Use the previously audited Tmall-24-0.25-V2 data and the same CID index across all three variants, fixing identities, targets, views and candidates. The CID index is Git-ignored and must first be copied from an authorized source to:

`data/Tmall-24-0.25-V2/Tmall-24-0.25-V2.index.cid.shuffle.chunk64.json`

Expected current source SHA256: `1263e3b160f89e4f912feadd5d57e6aa1c9f37fe582ec670d7b5152d4bd6cbe2`. V2 JSON files are repository-relative symlinks to Tmall-24-0.25, whose underlying LFS inputs are required:

```bash
git lfs pull --include='data/Tmall-24-0.25/**' --exclude=''
python scripts/prepare_rebuttal_experiments.py \
  --suite hierarchy --seeds 42 \
  --base_model ./config/s2s-models/Qwen3TemporalHierarchicalFixedSoft \
  --dataset Tmall-24-0.25-V2 --data_path ./data \
  --index_file .index.cid.shuffle.chunk64.json \
  --tasks smb_explicit_decoder_4 --max_his_len 100 --model_max_length 1024 \
  --gpu 0,1 --port 2314 --precision fp32 \
  --per_device_train_batch_size 128 --per_device_eval_batch_size 128 \
  --gradient_accumulation_steps 8 --num_train_epochs 200 \
  --learning_rate 5e-4 --weight_decay 0.01 --warmup_ratio 0.04 \
  --patience 20 --temperature 0.7 --test_batch_size 8 --num_beams 20 \
  --eval_behavior alipay --export_user_targets \
  --hierarchy_maps ./config/rebuttal/tmall-hierarchy-screening.json \
  --output_dir ./results/rebuttal/remaining-tmall
python - <<'PY'
from SeqRec.datasets.loaders.session_behavior import load_SMB_datasets, load_SMB_test_dataset
settings = dict(dataset='Tmall-24-0.25-V2', data_path='./data', max_his_len=100, index_file='.index.cid.shuffle.chunk64.json')
load_SMB_datasets(**settings, tasks='smb_explicit_decoder_4', train_session=True)
for task in ('smb_explicit_valid', 'smb_explicit'):
    load_SMB_test_dataset(**settings, test_task=task)
PY
python scripts/run_rebuttal_experiment.py \
  --plan ./results/rebuttal/remaining-tmall/plan.json \
  --variants full weak_auxiliary random_auxiliary_seed42 --seeds 42 --execute
```

The variants are the legacy identity-order prior, tied auxiliary levels and a predefined nonidentity random auxiliary order. `random.Random(42).sample([0,1,2],3)` gives [2,0,1]; this changes click's rank, not merely collect/cart. Alipay retains its core identity. Only the behavior-to-behavior prior subtable changes; the existing special-ID-0 row/column and global behavior-level metadata remain. Tying also changes rank distances, whereas the permutation preserves the rank set. The V2 data metadata ties collect/cart/alipay at the maximum level; legacy target selection picks the last, alipay, and augmentation protects these max-level events. All three groups share that setting. Conclusions are limited to prior-order sensitivity for this dataset/CID, not all hierarchy mechanisms.

## 4. P1: key seeds 43 and 44

Predeclare Full versus no-QKV as the key repeat comparison, rather than choosing seeds after test results. Each already has seed 42 on the server:

```bash
python scripts/run_rebuttal_experiment.py \
  --plan ./results/rebuttal/remaining-seeds/plan.json \
  --variants full no_qkv --seeds 43 44 --execute
```

If the final claim needs other component repeats, additionally run the relevant entries. Expanding all three below adds six training jobs and is not mandatory in the first round:

```bash
python scripts/run_rebuttal_experiment.py \
  --plan ./results/rebuttal/remaining-seeds/plan.json \
  --variants no_gate shared_ffn zero_prior --seeds 43 44 --execute
```

## 5. Optional: two more matched-decoder seeds

The server already runs plain_matched seed 42. For the full three-seed capacity comparison:

```bash
python scripts/run_rebuttal_experiment.py \
  --plan ./results/rebuttal/remaining-seeds/plan.json \
  --variants plain_matched --seeds 43 44 --execute
```

## Evaluation of the three running jobs and outputs

Full/no-QKV/plain_matched need validation/test after their training finishes. To evaluate on another machine, copy the final checkpoint root (weights, tokenizer, config and start manifest), not merely an intermediate epoch checkpoint. A still-running job cannot be represented as final.

After copying to `./results/rebuttal/transferred/<variant>/checkpoint/`, these evaluation-only commands on the target machine avoid retraining:

```bash
for variant in full no_qkv plain_matched; do
  for split in validation test; do
    task=smb_explicit
    if [[ "$split" == validation ]]; then task=smb_explicit_valid; fi
    result="./results/rebuttal/transferred-eval/${variant}/${split}-ranking.json"
    test ! -e "$result" || exit 1
    CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python -m torch.distributed.run \
      --nproc_per_node=2 --master_port=2314 main.py test_SMB_decoder \
      --backbone Qwen3TemporalHierarchical \
      --ckpt_path "./results/rebuttal/transferred/${variant}/checkpoint" \
      --dataset ShortVideoAD --data_path ./data --index_file .index.json \
      --seed 42 --max_his_len 100 --test_batch_size 8 --num_beams 20 \
      --test_task "$task" --behaviors cvr --export_user_targets \
      --results_file "$result" || exit 1
  done
done
```

Normal runner outputs are `<plan-root>/<variant>/seed_<seed>/checkpoint/`, `validation-ranking.json` and `test-ranking.json`. Per-user metrics, UIDs, original target IDs, split/hash receipts and pipeline start/finish records accompany them. Validation/test remain separate; test never selects checkpoints or lambda.

For an explicitly partial target-machine P0 summary, with reference differences left blank when Full is absent:

```bash
python scripts/summarize_rebuttal_results.py \
  --plan ./results/rebuttal/remaining-p0/plan.json \
  --split test --behavior cvr --reference full --allow_partial \
  --output_file ./results/rebuttal/remaining-p0/summary-cvr-test.json \
  --csv_file ./results/rebuttal/remaining-p0/summary-cvr-test.csv
```

Once all Tmall results exist, omit `--allow_partial`, select its plan and behavior alipay to get differences against its local Full. Cross-machine aggregation requires both plans/manifests/results plus verification of input contents, SID, recipe and actual split. Path/mtime fingerprints cannot be automatically rebased across machines; editing JSON paths is not proof of comparability. Historical lambda-0 results are not silently imported into the controlled table.

## Items without unconditional runnable commands

- Strict core-only/raw/consolidated still need real raw events, consolidation provenance and a shared split/target manifest. The adapter exists, but never reconstruct raw by reversing the deepest state. Fix targets/candidates and validation selection once genuine inputs arrive.
- Matched runtime needs a timing protocol; MPC still needs an independent implementation/evaluation adapter. Neither belongs to this automatic P0 queue.
- Sparse/popularity/repeat diagnostics and separate Q/K/V ablations remain deferred/future work under the latest plan.

## Command verification

All 27 relevant CPU tests passed, including exact weight/logit equality between FixedBias zero initialization and FixedSoft lambda 0. An actual temporary Toy pipeline completed train, validation and test with JSON, UID/target exports and phase receipts. The CPU device override was test-only; no production training path changed. No production training or GPU evaluation was launched, and the three existing jobs continued running. Bilingual command blocks match and underwent Bash/embedded-Python syntax checks plus the prescribed flake8 check.
