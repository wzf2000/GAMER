# Rebuttal delivery tools: targets, parameters, CSV and queue

Date: 2026-10-27. Base commit `38a90d2`, existing workspace changes retained. Chinese counterpart: `.zh.md`. This implements the four gaps identified in the 2026-10-27 handoff audit. No commit, full training, dependency installation or production evaluation was performed.

## Compatibility and running jobs

No model, training entrypoint, collator, dataset builder, cache or existing experiment config was changed in this follow-up. The training tasks already loaded their code before this work. Full (GPU 1/5), plain_matched (2/6) and no-QKV (4/7) were left running. All new probes ran on CPU with one Torch thread; no GPU validation was attempted because the devices were occupied.

The only new evaluator path is guarded by `--export_user_targets`, which defaults to false. Other prepared-plan suites retain their existing launch-script behavior. Handoff is a new explicit suite whose stage scripts print pending commands by default. Existing summary fields remain; CSV is optional.

## Original targets and UID alignment

`SeqRec/evaluation/smb_targets.py` implements an optional in-memory dataset copy. It checks all cached target representations/behaviors and histories against the ordered source events before adding missing legacy validation UIDs. It rejects incompatible caches instead of rewriting them. Both `test` and `valid_test` ranking are supported; loss-only validation is not a target export.

For example, append `--export_user_targets` to an existing `main.py test_SMB_decoder` command. New handoff ranking commands include it automatically. Existing metric arrays and `.uids.json` sidecars remain. Each behavior additionally gets `user_level_metrics_<behavior>.targets.json` containing UID, original target item IDs, event/session indices, SID/behavior representations, per-user metrics and candidate IDs sharing each SID. Repeated events remain repeated, and item IDs come from held-out source events rather than inverse SID lookup. SHA256 binds metrics, UID sidecar, candidate index and the evaluation receipt; large source files have path/size/mtime snapshots, not content hashes.

Opening the flag does not alter candidates, prompts, history or labels. Legacy validation now has explicit UIDs under this flag, so DDP metrics use the existing UID deduplication path; the default path remains unchanged.

## Independent parameter audit

`scripts/audit_rebuttal_parameters.py` instantiates a separate resolved TH model, or loads an optional completed checkpoint. It never uses an optimizer or mutates training artifacts. CPU is the default. A synthetic all-behavior probe is explicit; actual collated JSON batches are accepted via `--batch_file`, including non-tensor metadata. `--temperature` defaults to the controlled recipe's 0.7 and is recorded.

```bash
python scripts/audit_rebuttal_parameters.py \
  --config results/rebuttal/2026-10-26-p0/configs/full/config.json \
  --device cpu --seed 42 --temperature 0.7 \
  --output_file results/rebuttal/full-parameters-new.json
```

The report separates unique registered/trainable tensors, buffers, calls with nonempty inputs, loss-connected tensors, nonzero-gradient tensors and observed zero/unobserved tensors. `active_participating` is the intersection of nonempty calls and loss connectivity, including tensors with zero gradient. All counts are whole-tensor counts; embedding counts are not accessed-row counts. Neither a zero gradient nor lack of observation proves global structural unreachability. Mode and gradients of the inspected instance are preserved; no global unused-capacity conclusion is inferred from one batch.

Seven production-vocabulary probes (23,675 tokens, random seed 42, temperature 0.7, training-style masked padding, eval mode with gradients) are retained under `results/rebuttal/2026-10-27-p0-supported/parameters-final/`. Registered counts remain Full 29,935,104; no-QKV 29,922,816; no-gate 29,672,960; shared-FFN 12,240,384; all lambda points 29,935,104. In these probes all registered tensors were called and connected, but Full had 27 zero-gradient tensors. Batch tensor digests are recorded. The report records that distinction instead of declaring them globally unused. These are synthetic diagnostics, not trained-model effectiveness results.

## JSON and CSV result summaries

`scripts/summarize_rebuttal_results.py` now accepts `--csv_file`. The long-form CSV includes actual per-run metric values, reference values for the same seed, absolute differences and relative percentages. Missing runs have an explicit status and blank numeric cells. A zero reference has a blank relative percentage. Existing JSON means/sample standard deviations and paired differences remain; single-seed standard deviation is null.

```bash
python scripts/summarize_rebuttal_results.py \
  --plan results/rebuttal/2026-10-27-p0-supported/plan.json \
  --split test --behavior cvr --reference full --allow_partial \
  --output_file results/rebuttal/cvr-test-summary-new.json \
  --csv_file results/rebuttal/cvr-test-summary-new.csv
```

`--allow_partial` explicitly permits an all-missing report. The prepared seven-run missing-only JSON/CSV are named `summary-cvr-test-pending.*`; they contain no measured production scores. No test result was used for selection.

## Handoff suite, reuse and live snapshots

`prepare_rebuttal_experiments.py --suite handoff --seeds 42` creates exactly seven configurations: Full, no-QKV, no-gate, shared-FFN, zero-prior, lambda 0.01 and 0.1. Full aliases lambda 0.05; zero-prior aliases lambda 0. Stage labels, resolved config differences, model/data seed and source reuse evidence are recorded. Data seed 42 reflects the existing random factor4 builder, not policy-augmentation seed controls.

Use `--reuse_plans <plan.json> ...` to preserve existing output/config/result paths. Reuse checks the code revision, resolved config SHA, data snapshots, training recipe and device count; start manifests add actual seed, parameter count, dataset/script/HF settings, precision and cache checks. The dirty workspace still requires the retained source snapshot to establish exact historical code; matching Git HEAD alone is not proof that arbitrary workspace edits are equivalent.

The concrete prepared plan is `results/rebuttal/2026-10-27-p0-supported/plan.json`. It reuses all seven existing planned paths from the earlier P0/lambda plans; Full and no-QKV are already running. Its stage scripts invoke the manager to print the next command, without launching it.

```bash
python scripts/manage_rebuttal_queue.py \
  --plan results/rebuttal/2026-10-27-p0-supported/plan.json \
  --parameter_audits results/rebuttal/2026-10-27-p0-supported/parameters-final/*.json \
  --output_file results/rebuttal/status-new.json
```

Use `--output_script <new.sh>` to save a single next command for later deliberate execution. `--gpu` may change IDs but must preserve device count; the tool does not claim those devices are free. At the recorded snapshot, next was no-gate; GPU 4/7 were occupied by no-QKV. No generated training command was executed.

`refresh_status` distinguishes running PIDs, unconfirmed started directories, completed final root weights/trainer state with `train_runtime`, missing results and verified result receipts. An epoch checkpoint or start manifest alone is insufficient for completion. Inaccessible PID namespaces retain uncertainty. Stage B waits until stage A training is complete; ranking queues also require the requested stage-A results. Invalid config/data/artifacts block progression. Existing started directories are never silently restarted or overwritten.

## Verification and limits

- Twenty-three focused tests passed, including default flags/model paths, seed/shifted loss, target collisions/repeats, invalid cache rejection, parameter graph/zero-gradient classification, reuse, queue gating, rejection of a different handoff training task and CSV missing/value differences.
- Actual Toy validation and test ranking on CPU produced identical parsed metrics with export off/on. Copied cache checksums were unchanged. Latest artifacts: `/tmp/gamer-four-support-cpu-n9oj87up/`; the actual-batch audit is in `/tmp/gamer-four-support-cpu-fh4tgk9d/`. A test-only CPU device and zero DataLoader workers were injected without changing the production device path.
- An actual Toy causal collator batch and completed Toy checkpoint passed the optional parameter-audit path. All seven final-vocabulary CPU synthetic probes passed separately.
- Flake8 uses the repository's `--max-line-length=500 --ignore=F401,E203,W503,F841`; compileall and generated-shell `bash -n` were run. Logs and source evidence are retained with the new plan.
- Full production ranking and GPU checks of the new delivery paths remain unrun. Existing model/training implementation and production data/cache/config fingerprints remain unchanged by this follow-up. All three user training jobs continued progressing.
