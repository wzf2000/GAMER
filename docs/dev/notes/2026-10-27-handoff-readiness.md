# Rebuttal handoff readiness and next run

Date: 2026-10-27, Asia/Shanghai. Checkout: `dev`, `38a90d2` plus the existing uncommitted experiment support. Chinese counterpart: `.zh.md`.

Follow-up: the four implementation gaps below have since been addressed; see [delivery tools](../design/2026-10-27-rebuttal-delivery-tools.md). The remainder preserves the earlier audit snapshot.

This audit follows the supplied 2026-10-27 handoff. Its queue supersedes the earlier seed/capacity-first queue: P0-A Full/no-QKV/no-gate/shared-FFN, then P0-B lambda 0/0.01/0.05/0.1. Full supplies lambda 0.05. Already-running jobs were left running; no new full training or large evaluation was launched. This pass adds an audit/registry and commands, without changing production Python or existing configurations.

## Ready capabilities

All seven configurations already have real model/configuration support. Each component configuration differs from Full in one intended key:

| Variant | Configuration change | Total parameters | Trainable parameters |
| --- | --- | ---: | ---: |
| Full | None | 29,935,104 | 29,935,104 |
| no-QKV | `th_use_behavior_qkv=false` | 29,922,816 | 29,922,816 |
| no-gate | `th_use_output_gate=false` | 29,672,960 | 29,672,960 |
| shared-FFN | `sparse_layers_decoder=[]` | 12,240,384 | 12,240,384 |
| lambda 0 | `th_relation_bias_soft_scale=0` | 29,935,104 | 29,935,104 |
| lambda 0.01 | `th_relation_bias_soft_scale=0.01` | 29,935,104 | 29,935,104 |
| lambda 0.1 | `th_relation_bias_soft_scale=0.1` | 29,935,104 | 29,935,104 |

Counts use the instantiated model with the expanded 23,675-token vocabulary. The frozen prior is a buffer, so it is not a registered parameter. These counts do not establish the number of active parameters under a particular loss.

No-QKV retains ordinary projections, normalization, output gate, FFN, prior and TH layers 2–5. No-gate omits only the attention output multiplication. Shared-FFN keeps `behavior_injection_decoder=[0,1,2,3,4,5]` and the existing injection implementation; no model decoupling is required. The prior stays frozen, lambda 0.05 and runtime multiplier 1 for all three component ablations.

Seed/cache isolation, start manifests, separate validation/test ranking commands, UID sidecars for test metrics, and JSON aggregation with differences from Full already exist. Old FixedBias results remain unverified reuse candidates; no completed lambda-0 run is credited here. The handoff's statement that 0.05 was an empirical preset is author-supplied provenance, not something executable code establishes.

## Remaining implementation and evidence gaps

1. **Ground-truth item IDs in per-user exports.** `SeqRec/tasks/evaluation/base.py::_save_user_metrics` saves metric arrays plus UID/checksum sidecars, but no original target item IDs. Export UID, behavior, split, original item IDs and SID strings from the actual held-out events, preserve repeated targets, and handle SID collisions explicitly. This can be added before final evaluation without retraining. Validation-ranking records also lack UID in `BaseSMBDataset._process_valid_test_data`; changing only the builder would not update existing cached records. Any fix must handle those caches or use an independent export.
2. **Active parameter statistics.** Existing helpers/manifests record registered trainable counts. Add a clearly defined execution/gradient audit for the final vocabulary and training loss, distinguish unreachable weights from zero gradients for a particular batch, and keep buffers separate. Registry `active_participating=null` deliberately marks this as unmeasured.
3. **Result CSV.** `scripts/summarize_rebuttal_results.py` produces JSON, including per-variant metrics and paired differences, but no metric CSV. Add a CSV export for actual values/differences and explicit missing runs; the new registry CSV contains configuration/status/counts only.
4. **Reusable queue/status tooling.** A seven-run registry and stage-specific scripts were prepared for this handoff, but the general tracked planner still uses the earlier components/seeds/lambda suites. A reusable handoff suite/status refresh could later automate stage ordering, Full/0.05 deduplication and completion detection. The current registry is a timestamped audit, not a live monitor.

These gaps do not prevent no-QKV training. Historical submission training provenance, final test scores and convergence cannot be inferred from current intermediate logs. This controlled series uses the documented new recipe; it must not be mixed with unverified historical Full/zero-prior results.

## Prepared artifacts and current jobs

Artifacts: `results/rebuttal/2026-10-27-handoff-audit/`:

- `registry.json`: seven configurations, resolved config/differences, source snapshots, data/cache fingerprints, model/data seed, selection rules, budgets, output paths and observed status. It is also compatible with the existing result summarizer.
- `registry.csv`: configuration/status/parameter summary.
- `P0-A-{train,validation,test}.sh` and `P0-B-{train,validation,test}.sh`: explicit stage commands, GPU 4/7, port 2316. Train scripts omit the already-running Full; all generated commands refuse an existing output path. Run ranking after the relevant training completes and saves its final best model.
- `run_next_no_qkv.sh`: the single next training command, without starting the other queued jobs.
- `gamer_handoff_audit_20261027.py` and verification logs: preparer and audit evidence.

At inspection, Full was running on GPU 1/5 and plain_matched on 2/6. Latest saved Full state was epoch 32, validation loss 2.052139; plain_matched was epoch 25, validation loss 2.062975. Both logs continued advancing without an observed training error. These are different training stages and are not a final performance comparison. GPU 4/7 were empty, and port 2316 was unused.

The next run is no-QKV, seed 42, with the same ShortVideoAD original SID `.index.json`, `smb_explicit_decoder_4`, train cache (238,270 examples) and validation cache (193,316 examples). History limit 100; model limit 1024; dropout 0.2; FP32/SDPA; per-device batch 128; two GPUs; accumulation 8; effective batch 2048; learning rate 5e-4; warmup 0.04; weight decay 0.01; temperature 0.7; maximum 200 epochs; early-stopping patience 20. Within-run selection remains all-behavior validation loss. Final core ranking uses `smb_explicit` test, with cvr HR@10/NDCG@10 as the primary report.

From the repository root:

```bash
bash results/rebuttal/2026-10-27-handoff-audit/run_next_no_qkv.sh
```

Output: `results/rebuttal/2026-10-26-p0/no_qkv/seed_42/checkpoint/`. The existing date is retained to keep this run in the same controlled series. After no-QKV, queue no-gate, shared-FFN, lambda 0, 0.01, 0.1. Extra seeds/hierarchy/core-only/capacity jobs are deferred by the updated handoff; the already-running plain_matched job is retained.

## Verification

- Ten existing component/seed/plan tests passed on CPU.
- Seven micro-model variants passed future-token independence, right-padding invariance, position/behavior group-boundary assertions and exact save/reload output checks. Shared-FFN retained a nonzero injection gradient. These supplementary checks were run inline; their logs are retained with the audit.
- On the actually free A100 80GB GPUs 4/7, `tests/smoke_rebuttal_ddp.py` passed two forward/backward steps and cached/uncached beam generation for Full, no-QKV, no-gate, shared-FFN, zero-prior and plain. Test processes exited and released both devices.
- The 23 source-file hashes in the earlier code snapshot still matched the workspace; no production source/configuration changed during this audit.
- Flake8 passed for all 22 modified/new project Python files with `--max-line-length=500 --ignore=F401,E203,W503,F841`. `bash -n` passed for all seven generated scripts. No full production training or ranking evaluation was run in this audit.
