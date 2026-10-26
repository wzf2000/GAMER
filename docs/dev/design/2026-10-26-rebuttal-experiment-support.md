# TH-FixedSoft rebuttal experiment support

Date: 2026-10-26. Audit baseline: `38a90d2`; this document describes subsequent working-tree changes. The Chinese counterpart has the `.zh.md` suffix. Development follows the user's order 1–6. No full production-data training was launched.

Forwardable evidence notes are in `../notes/2026-10-26-review-agent-handoff.md`. New support does not establish which settings historical submission runs actually used.

## 1. P0 component ablations and parameter matching

Two config options are added to `Qwen3TemporalHierarchicalAttention`, both defaulting to true:

- `th_use_behavior_qkv`: disabling removes behavior Q/K/V embedding parameters/additions while retaining prior, output gate, and FFN.
- `th_use_output_gate`: disabling removes attention-output gate parameters/multiplication while retaining Q/K/V conditioning, prior, and FFN.

Defaults preserve existing parameter names and computations. Options survive config/checkpoint reload for training and generation. The output gate is distinct from the SwiGLU FFN gate projection.

`scripts/prepare_rebuttal_experiments.py --suite components` prepares full, plain_matched, no_qkv, no_gate, shared_ffn, and zero_prior in that order:

- Full uses existing FixedSoft, keeping data, item index, task, and dropout consistent across groups.
- The plain decoder clears TH-attention, token-position expert, and FFN behavior-injection layer lists. It retains the same causal-mask/dropout path and automatically chooses the nearest integer FFN width for the actual vocabulary/model parameter count.
- Shared FFN clears only `sparse_layers_decoder`, preserving FFN behavior conditioning; it does not substitute a different backbone's dense configuration.
- Zero prior changes only frozen soft scale to zero; conditioning, gate, and position experts remain.

With the current ShortVideoAD index, full has 29,935,104 parameters and the matched decoder 29,935,616 (FFN width 3501), a difference of 512. Matching parameters does not match FLOPs/runtime. Default relative tolerance is 0.1%; tiny fixtures may explicitly relax it because integer width steps are relatively large.

## 2. P0 seeds, run manifests, and aggregation

The requested training seed is restored after data preparation and before model initialization. This fixes cold-cache augmentation's `set_seed(42)` affecting initialization. Existing augmentation seeding is retained; cache hits/misses no longer change model initialization for the same training seed.

From the repository root in the GAMER Python environment, prepare a priority-ordered plan:

```bash
python scripts/prepare_rebuttal_experiments.py \
  --suite components --seeds 42 43 44 --key_seeds_only \
  --dataset ShortVideoAD --data_path ./data --index_file .index.json \
  --tasks smb_explicit_decoder_4 --max_his_len 100 --model_max_length 1024 \
  --gpu 0,1,2,3 --precision fp32 \
  --per_device_train_batch_size 128 --per_device_eval_batch_size 128 \
  --gradient_accumulation_steps 4 --num_train_epochs 200 \
  --learning_rate 5e-4 --warmup_ratio 0.04 --patience 20 \
  --output_dir ./results/rebuttal/2026-10-26-p0
```

This writes configs, JSON manifests, and shell commands; **no training is launched**. Existing plan directories are rejected. These are explicit new experiment settings, not recovered final submission commands; align them with final settings before launching.

Artifacts include `configs/`, `plan.json`, `run_train_key.sh` (three seeds each for full/matched), `run_train_remaining.sh` (the first seed for four remaining ablations), `run_train.sh` (all ten), `run_validation.sh`, and `run_test.sh`. Stage outputs are separate and generated scripts reject existing checkpoint/result paths.

Commands pin the preparation interpreter; distributed execution uses that interpreter's `torch.distributed.run`. Batch arguments are per device; the example has effective batch `128×4×4=2048`. Random4 still includes the original plus up to four additional views and is not no augmentation.

`--suite seeds` prepares only the two key groups and defaults to seeds 42/43/44. `components` defaults to seed 42.

After user approval, execute the appropriate `bash .../run_train_key.sh`. Plans pass `--record_run_manifest`, recording effective model/HF training config, parameter count, item-index SHA256, seed, actual attention backend, parameter dtype, devices, and code revision before training. AMP precision is recorded separately; parameter dtype is not autocast precision. A start manifest does not prove training completion.

The index and behavior-level metadata use SHA256; large raw JSON/cache files record paths, sizes, and mtimes, with actual cache sample counts. Aggregation checks the planned data snapshot and cache consistency across runs. These lightweight snapshots are not full content hashes for large files.

SMB ranking writes a `.run.json` sidecar containing actual split/task, candidate-index fingerprint, checkpoint, training-manifest fingerprint, and result fingerprint. Aggregation validates these bindings and cannot silently summarize validation as test.

```bash
python scripts/summarize_rebuttal_results.py \
  --plan ./results/rebuttal/2026-10-26-p0/plan.json \
  --split test --behavior cvr --reference full \
  --output_file ./results/rebuttal/2026-10-26-p0/summary-cvr-test.json
```

All planned results are required by default. `--allow_partial` explicitly marks an interim report and lists missing runs. Reports include means, sample standard deviations (ddof=1), seed sets, and variant-minus-reference differences only for matching seeds. Single-seed ablations do not receive fabricated three-seed variance estimates.

## 3. P0 core-only / raw / consolidated

`scripts/prepare_rebuttal_history_views.py` requires supplied real raw events and consolidation provenance. It never reconstructs raw logs from deepest states.

Raw and consolidated inputs are JSONL, with each event containing:

```json
{"uid":"u1","event_id":"e1","item_id":0,"behavior":"click","session_id":0,"timestamp":"2026-01-01 00:00:00","available_at":"2026-01-01 00:00:00"}
```

UID/event IDs are strings; item/session IDs are integers; timestamps use canonical timezone-naive second precision. `timestamp` is the adapted occurrence time, and `available_at` is when the state can enter history. Upstream owners must establish their business meaning; the adapter does not infer it.

Consolidated events additionally require nonempty `source_event_ids` identifying real raw events for the same user/item/original session. Every raw event must belong to exactly one consolidated event, with no invented provenance or backdating before its source evidence.

The shared split manifest is:

```json
{
  "raw_provenance":"Actual original source and snapshot version",
  "core_behavior":"cvr",
  "users":{
    "u1":{"valid_session":2,"test_session":3,
          "valid_target_event_ids":["e5"],"test_target_event_ids":["e7"]}
  }
}
```

The last two original sessions must be the shared validation/test sessions, held-out core event IDs/order must match, and the index is shared exactly. Core-only filters raw histories to core events while retaining the common held-out core targets. Users without core training history remain in evaluation; empty-history collation is supported.

Static SMB cannot represent every case where earlier events are backfilled after a prediction while also serving as earlier labels. If `available_at` crosses the subsequent prediction cutoff, export is rejected and a query-time adapter is required. Late states are never silently inserted into history.

After actual inputs are supplied under `inputs/rebuttal/`:

```bash
python scripts/prepare_rebuttal_history_views.py \
  --raw_file inputs/rebuttal/raw.jsonl \
  --consolidated_file inputs/rebuttal/consolidated.jsonl \
  --split_manifest inputs/rebuttal/split.json \
  --behavior_levels data/ShortVideoAD/ShortVideoAD.behavior_level.json \
  --index_file data/ShortVideoAD/ShortVideoAD.index.json --index_kind SID \
  --dataset ShortVideoADViews --max_his_len 100 \
  --output_dir ./results/rebuttal/2026-10-26-history-inputs
```

Outputs are three data roots, `core_only/`, `raw/`, and `consolidated/`, with the same dataset name. Each index is named `<dataset>.index.json`; the manifest separately records actual SID/CID kind and source fingerprints. `coverage.json` reports per-user validation/test total/visible event counts, history tokens, raw-source coverage, core counts, and time span. History tokens exclude prompt/target tokens.

Prepare all groups with identical hyperparameters and `--validation_behavior cvr --eval_behavior cvr` so validation-loss target behavior does not vary with the view. For equal historical coverage, use the coverage report to choose appropriate windows; equal event budgets are not claimed to imply equal raw coverage. Production raw is unavailable; only synthetic fixtures were validated.

## 4. P1 weak hierarchy interventions

`th_prior_behavior_ranks` assigns one prior rank per identity in `behavior_order`. It changes only behavior-to-behavior entries of the frozen soft table; special-ID-0 rows/columns, identity tokens/embeddings/FFNs, core targets, and augmentation stay unchanged. Only frozen-table soft initialization is supported; invalid lengths, negative/nonfinite ranks, and other relation types are rejected.

Do not implement this by editing data `behavior_level.json`. The example `config/rebuttal/tmall-hierarchy-maps.json` includes tied auxiliary ranks, reversed auxiliary order, and flat auxiliary ranks, retaining alipay as the core target identity. This tests fixed-prior ordering/ties, not removal of all behavior conditioning.

```bash
python scripts/prepare_rebuttal_experiments.py \
  --suite hierarchy --dataset Tmall-24-0.25-V2 \
  --index_file .index.cid.shuffle.chunk64.json \
  --hierarchy_maps config/rebuttal/tmall-hierarchy-maps.json \
  --seeds 42 --output_dir ./results/rebuttal/2026-10-26-tmall-hierarchy
```

Full retains the original identity-order prior; other groups change only table initialization while sharing data/cache. Historical weak-level metadata does not prove the historical model prior used tied ranks.

## 5. P1 lambda grid

Use the same planner with a frozen table and runtime multiplier 1. Lambda is `th_relation_bias_soft_scale`, not a loss weight. Do not pass an unregistered config field directly to `main.py`.

```bash
python scripts/prepare_rebuttal_experiments.py \
  --suite lambda --lambdas 0 0.01 0.05 0.1 --seeds 42 \
  --output_dir ./results/rebuttal/2026-10-26-lambda
```

If P0 full/zero runs are complete under exactly matching conditions, use `--lambdas 0.01 0.1` for only the two missing points. For a grid reference use `--reference lambda_0.05`. Select configurations on validation and reserve test for reporting.

## 6. P1 offline sparse-user buckets

New SMB test per-user output retains existing metric arrays and adds `.uids.json`, preserving exact UID order, split, and history window and binding it to the metric file via SHA256. Offline aggregation checks these fields and directly uses saved multi-target NDCG/Recall, rather than reconstructing them from a best rank or rerunning inference.

```bash
python scripts/analyze_sparse_metrics.py \
  --ours ./results/ShortVideoAD/smb_explicit_decoder_4/Qwen3TemporalHierarchicalFixedSoft/results-smb_explicit-original/user_level_metrics_cvr.json \
  --baseline ./results/ShortVideoAD/smb_explicit_decoder_4/Qwen3TemporalHierarchicalFixedBias/results-smb_explicit-original/user_level_metrics_cvr.json \
  --behavior_file ./data/ShortVideoAD/ShortVideoAD.SMB.behavior.json \
  --session_file ./data/ShortVideoAD/ShortVideoAD.SMB.session.json \
  --behavior cvr --split test --max_his_len 100 --count_scope visible \
  --thresholds 1 3 6 --bootstrap_samples 1000 --seed 42 \
  --legacy_lexicographic_order \
  --output_file ./results/rebuttal/2026-10-26-sparse-cvr.json
```

Old files lack UID sidecars. This command explicitly assumes the current writer's lexical UID order and a population matching the reconstructed split; outputs record that limitation. Without the opt-in or a bound UID sidecar, joins are rejected. Re-export when old order cannot be established. New files need no legacy flag.

Default buckets are 0, 1–2, 3–5, and 6+. `visible` counts core events in the last N historical events; `full` counts the complete pre-prediction prefix. Outputs include bucket sizes, original metric means, paired differences, and reproducible bootstrap 95% intervals. These intervals do not estimate variation across training seeds.

## Validation and limits

Tests cover default compatibility, component gradients/shifted loss, KV caching, checkpoint reload, cold/warm-cache seeds, parameter matching, same-session provenance, target/candidate equality, empty core histories, late-evidence rejection, UID/hash alignment, visible-history buckets, and paired aggregation. A tiny actual Toy training-to-ranking execution chain is also checked.

Run project flake8 settings and compileall for Python, and bash -n on generated shell scripts. With user authorization, the then-idle GPU 7 (A100 80GB) completed actual CLI Toy single-epoch training, validation/test ranking, and manifest-checked aggregation. GPUs 5/7 passed two DDP forward/backward steps and cached/uncached beam checks for all six variants. The reusable entry is `tests/smoke_rebuttal_ddp.py`; recheck idle devices before running it.

The prepared P0 plan is at `results/rebuttal/2026-10-26-p0/` with ten runs, the hierarchy plan has four, and the extra-lambda plan has two. None were launched. These candidate plans explicitly use GPUs 5/7, per-device batch 128, and accumulation 8 (effective batch 2048); they are not recovered historical settings. Production full training and large ranking remain unrun. Original business definitions and final historical submission artifacts still require external evidence.
