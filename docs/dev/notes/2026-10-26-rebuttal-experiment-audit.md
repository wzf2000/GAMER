# Rebuttal experiment feasibility and evidence audit

Date: 2026-10-26. Code snapshot: `dev` / `38a90d2`.
Chinese counterpart: `2026-10-26-rebuttal-experiment-audit.zh.md`.

This audit describes the snapshot above. Subsequent implementation and verification are documented in `../design/2026-10-26-rebuttal-experiment-support.md`; new interfaces do not resolve missing historical artifacts.

## Scope

- The submission model is **TH-FixedSoft**, rather than Original GAMER / Qwen3Multi or the latest augmentation variant.
- The user identifies submission Eq. (8) as `r_soft(a,b) = lambda * min(a-b, 0)`, with `lambda=0.05`. This is a fixed hierarchy-prior scale, not an auxiliary-loss coefficient.
- This audit inspects local code, processed data, results, logs, and available checkpoints. The macOS paper/resource paths quoted by the user were not read on this host; their formula and hardware/timing information are user-provided evidence.
- Only the previously validated loader default fix was committed. No new ablation implementation, full training, or large evaluation was performed.

## Four evidence questions

### 1. ShortVideoAD timestamps, attribution, and delayed conversions

Confirmed: `BaseSMBDataset._load_data()` parses strings in `*.SMB.time.json` with `%Y-%m-%d %H:%M:%S`, subtracts each user's first interaction time, and divides seconds by 1800. Cached `time` values use half-hour units. Sample time features are subsequently expressed relative to the target time.

Unresolved: whether each timestamp denotes an impression, click, conversion, attribution, ingestion, or a representative time after consolidation; the attribution window; delayed-conversion censoring/correction; and whether earlier clicks are backfilled. The processed-JSON reader provides no business definition. Half-hour normalization does not establish a half-hour attribution window.

Required evidence: upstream field definitions, attribution/snapshot rules, or the implementation producing these JSON files.

Evidence: `SeqRec/datasets/session_behavior/base.py:77`, `:126`, and `:_generate_times`.

### 2. Tmall/JData behavior-chain construction

No actual raw-event-to-deepest-state preprocessing script was found in the current repository or visible Git history. `analysis/ShortVideoAD_behavior_analysis.ipynb` reads existing JSON/pickle artifacts; it is not an upstream raw-log construction script.

Confirmed downstream order: load preconstructed interaction / behavior / session files → hold out the penultimate session for validation and the last session for test → construct training prefixes and augmented views → truncate histories and collate inputs.

Unresolved: the same-user-item consolidation window, whether consolidation crosses sessions, and the upstream order of consolidation versus sessionization. The directory name `Tmall-24-0.25` does not prove a 24-hour consolidation window.

Usable processed inputs and CID indices exist in `data/Tmall-24-0.25-V2/` and `data/JData-V2/`. Their interaction / behavior / session / time files link to the original processed directories; the V2 names do not establish separate raw logs.

The hierarchy files also differ:

| Directory | `behavior_level.json` |
| --- | --- |
| Tmall | click=0, collect=1, cart=1, alipay=2 |
| Tmall-24-0.25-V2 | click=0, collect=1, cart=1, alipay=1 |
| JData-V2 | pv=0, click=1, cart=2, collect=2, buy=2 |

When multiple behaviors share the maximum level, the base dataset warns and selects the last as `target_behavior`. This does not establish that TH prior lookup preserves those ties; see the hierarchy-intervention discussion below.

Evidence: `SeqRec/datasets/session_behavior/base.py:98`, `:140`, and the listed data directories.

### 3. CVR/CTR execution paths

If CTR refers to the click row in the result files, the current `test_SMB_decoder` behaves as follows:

- cvr uses the cvr prompt and candidate prefix; click uses the click prompt and prefix. Both are not forced to the conversion prefix.
- Each behavior's candidate item catalogue comes from the entire `index_file`, with its behavior token attached. Candidates are not dynamically built from the third-last session. Whether the index itself was constructed from train-only items cannot be established by this reader.
- This task computes behavior-conditioned next-item ranking, not conversion/click probability AUC, LogLoss, or calibration.
- `test_task=smb_explicit` ranks on test; `smb_explicit_valid` ranks on validation; `--valid_loss` computes validation loss. The Target Behavior / Behavior Specific / Behavior Item modes of `test_MB_decoder` are a different protocol.

Three training concepts must be separated:

1. `smb_explicit_decoder_4` uses the complete prefix before validation and constructs the original sequence plus up to four augmented sequences. Its terminal target is the last retained event, normally in the third-last session for the original view.
2. Decoder-only training uses whole-sequence causal shifted loss with padding and behavior tokens masked. Earlier training sessions retained within the history window are supervised, not only the terminal target.
3. Events outside that window do not enter the sample. Training cross entropy uses the token vocabulary rather than an item-candidate sample drawn from selected sessions.

Original random4 uses deletion ratios `1/4, 2/4, 3/4, 4/4`, in addition to the original view. Too-short views are skipped, so exactly five samples per user are not guaranteed.

Located configuration/cache paths:

- `config/s2s-models/Qwen3TemporalHierarchicalFixedSoft/config.json`
- `data/ShortVideoAD/ShortVideoAD.index.json`
- `data/ShortVideoAD/ShortVideoAD.SMB.data.pkl`
- `data/ShortVideoAD/ShortVideoAD.SMBExplicitDatasetForDecoder.100.SMB.aug4.train.pkl`
- `data/ShortVideoAD/ShortVideoAD.SMBExplicitDataset.100.SMB.valid.pkl`
- `data/ShortVideoAD/ShortVideoAD.SMBExplicitDataset.100.SMB.test.pkl`

These paths do not replace the final submission training command or effective checkpoint configuration.

Evidence: `SeqRec/tasks/evaluation/test_SMB_decoder.py:93`, `:104`; `SeqRec/tasks/evaluation/helpers.py:62`; `SeqRec/datasets/session_behavior/base.py:398`; `SeqRec/datasets/session_behavior/decoder.py:28`, `:71`; `SeqRec/datasets/collators/generative.py:129`; `SeqRec/models/generative/common/temperature.py`.

### 4. Lambda, final checkpoints, and baseline records

- Verified FixedSoft settings: `th_relation_bias_soft_scale=0.05`, `th_relation_bias_init=soft`, and `th_relation_bias_trainable=false`. Runtime `th_relation_bias_scale` defaults to 1 and learnable scaling defaults to off. The relation table is a buffer, not a trainable parameter.
- The frozen bias matches the algebraic form of Eq. (8). Whether 0.05 was predetermined or selected using validation requires search/design records; a fixed configuration and its result do not establish the selection process.
- FixedSoft retains `results-smb_explicit-original.json`, `results-smb_explicit_valid-original.json`, and test per-user metrics. Its cvr test population has 8784 users, HR@5=0.13490437, and NDCG@10=0.08998073.
- This test/validation classification follows filenames and current task routing. The result files do not preserve complete run configurations/split manifests; without the corresponding historical evaluation logs, filenames alone cannot independently prove which split was actually evaluated.
- No final ShortVideoAD FixedSoft / FixedBias checkpoint directories were found locally. The submission's effective seed, optimizer, precision, batches, stopping epoch, attention backend, and actual parameter count cannot be verified.
- Weights exist under `checkpoint/SMB-decoder/ShortVideoAD/smb_explicit_decoder_4/Qwen3Multi/original/`, but the trainer state's best path points to `MB-decoder/.../mb_explicit_decoder_4/...` and its metadata matches the MB directory. This cannot be identified as the submission FixedSoft checkpoint.
- No actual SASRecB / TIGER model training, tuning, final evaluation, or checkpoint records were found. Several test logs contain an unused default `base_model=.../TIGER`; a default backbone in a rule task also does not establish a TIGER training run.
- SASRec with `smb_dis_diff` supports behavior-item remapping, but no retained record establishes that this was the submitted SASRecB setup.
- Source defaults are SASRec dropout=0.5, TIGER dropout=0.1, and FixedSoft residual/FFN and attention dropout=0.2. These are source defaults, not proof of final baseline settings or equal search budgets.
- Current generative training follows HF Trainer validation-loss model selection; the discriminative trainer selects on `metrics[-1]`, NDCG@10 by default. Final selection metrics and tuning protocols still need verification.

External artifacts needed: the final FixedSoft checkpoint/config/training_args/command and code version; lambda selection records; and SASRecB/TIGER search spaces, validation scores, model-selection rules, and final test records.

## Support for six experiment groups

### P0 component ablations and a parameter-matched plain decoder

Full, zero-prior, shared FFN, and a plain decoder already have execution paths; independent configurations and a common run manifest are needed. Disabling Q/K/V conditioning and the attention output gate needs new independent options in TH attention, preserving defaults and validating training, cached generation, and compatibility.

- Zero-prior: `Qwen3TemporalHierarchicalFixedBias` differs from FixedSoft only in prior initialization/scale configuration; its frozen zero table is equivalent to lambda=0. A FixedSoft copy with soft scale=0 is another option.
- Shared FFN: set `sparse_layers_decoder=[]` in a FixedSoft copy, retaining `mlp_type=Qwen3` and its behavior-injection layers. This shares experts across token positions while preserving other conditioning. Qwen3MultiDense is not this controlled ablation.
- Plain decoder: clear `temporal_hierarchical_attention_decoder`, `sparse_layers_decoder`, and `behavior_injection_decoder`, and disable `behavior_injection` in a TH-family configuration. This retains the same causal-mask and dropout implementation; increase FFN width to match parameters.

Actual model construction with current source settings, `vocab_size=23675`, three behaviors, and five tokens per item gives the following counts. These are not counts from the final submission checkpoint:

| Configuration | Trainable parameters |
| --- | ---: |
| FixedSoft full | 29,935,104 |
| Zero-prior | 29,935,104 |
| Shared FFN, retaining behavior conditioning | 12,240,384 |
| Plain decoder, original FFN width 512 | 11,571,200 |
| Plain decoder, FFN width 3501 | 29,935,616 |

The matched configuration differs by only 512 parameters. Matching total parameters does not match active parameters/FLOPs per token or runtime; the widened dense FFN requires its own measurement.

### P0 three seeds each for full and the matched decoder

`--seed` and the shell `suffix` already exist, but a cache dependence was reproduced: `SMBExplicitDatasetForDecoder._process_train_data()` calls `set_seed(42)`. With requested seed 7, cold-cache construction leaves the global torch seed at 42; a cache hit preserves 7. Model initialization occurs after data construction, so restore the requested seed before model initialization, or strictly prebuild and freeze the shared cache for all runs.

Keep augmentation seed/views fixed while varying model initialization and training randomness. Changing augmentation views is not merely changing the training seed. Add mean/std and paired differences for matching seeds. Allow approximately half an engineering day, overlapping with component work.

### P0 core-only / raw / consolidated

Only processed state sequences were located; no input could be established as genuine raw event logs. Filtering evaluation users by target behavior is not a core-only history experiment, which needs its own history view.

With genuine raw logs, implement stable event/session/target/candidate manifests, as-of rules, and a common split for all three history views. Report coverage under equal token budgets and token/FLOPs costs under equal historical coverage. Re-expanding deepest states is not a raw-log baseline.

Allow 2–4 engineering days after obtaining raw data, excluding acquisition delays. Sequence lengths and coverage must be measured before estimating the three training runtimes.

### P1 weak hierarchy interventions on Tmall

An independent identity-to-prior-rank map needs implementation. `_init_level_pair_bias()` uses `arange(num_behavior+1)`, lookup `action_indices` come from the router, and the training helper constructs `behavior_maps` by enumerating behavior tokens. The prior does not directly use potentially tied values in `behavior_level.json`.

For example, collect/cart may be tied in Tmall's data file but receive distinct default prior IDs. Editing JSON levels alone is not a reliable model-only hierarchy intervention and can affect augmentation, core selection, and cache semantics.

Change only the rank map used to initialize the frozen prior table, retaining behavior tokens, Q/K/V identity embeddings, FFN identities, core targets, augmented views, and candidates. This tests the direction/tie structure of the fixed prior, not every behavior-conditioned mechanism. Preserve special ID 0 separately. Allow 0.5–1 engineering day.

### P1 lambda grid `{0, 0.01, 0.05, 0.1}`

Model code already supports the grid; create independent configurations and preserve the final effective configuration. `--th_relation_bias_soft_scale` is not a registered CLI argument and does not automatically override model configuration. Point `--base_model` to a configuration directory containing the tokenizer resources.

Keep the table frozen, runtime multiplier at 1, data fixed, and seeds matched. Lambda=0 overlaps zero-prior and 0.05 overlaps full; if conditions match, only 0.01 and 0.1 require new runs. Allow 1–2 hours for configuration and checks, keeping validation selection distinct from test reporting.

### P1 offline sparse-user buckets

The main artifacts are available: FixedSoft and FixedBias cvr test metric files each contain 8784 entries and eight metrics. Every metric's array mean was verified against its corresponding test summary.

A small offline aggregation tool is needed. Existing files save arrays after lexicographically sorting UIDs, without preserving a UID column. Reconstruct the cvr user population from the same split and verify lengths/order; do not sort numeric UIDs or join arbitrary files by array position. If order cannot be established, re-export metrics with UIDs.

Existing `analyze_sparse_behavior` reloads both checkpoints and reruns inference, counts the entire pre-test history, and reconstructs NDCG from the best hit rank. That is not equivalent to the existing multi-target NDCG. The offline tool should aggregate stored metrics directly, distinguish core counts in the visible window (e.g. the last 100 items) from full-history counts, and report bucket sizes, differences, and paired intervals.

Allow 0.5–1 engineering day; execution should usually take CPU minutes and needs no GPU training. No formal bucket analysis was run and no bucket effect is claimed. Comparisons with the matched decoder or SASRecB/TIGER need their aligned per-user artifacts first.

## Runtime budget

Historical complete workflows provided by the user include periodic validation; no independently attributable test runtime is available:

| Historical model | Hardware | Workflow time |
| --- | --- | --- |
| FixedSoft, factor 4 | 4×A800-SXM4-80GB | 22:34:26 = 22.574 h |
| No soft prior | 4×A800-SXM4-80GB | 21:32:27 = 21.541 h |
| Standard Qwen3 decoder, factor 4 | 8×H800-SXM5-80GB | 08:37:50 = 8.631 h |
| MBGen | 8×A100-SXM4-80GB | 100:27:23 = 100.456 h |

Let `F=22.574 h`, `D` be the unmeasured matched plain decoder workflow, and `T_Tmall` be the unmeasured Tmall workflow. The following are scheduling formulas, not measurements or guarantees:

| Experiment | Estimated engineering effort | Training workflows/cost |
| --- | --- | --- |
| Six components, one seed each | 1–2 working days, including matching and option checks | 6 runs: one full, one matched decoder, and four ablations; variant times need measurement |
| Three seeds for full/matched | About 0.5 working day, partially overlapping above | 6 total; `3F+3D = 67.72h+3D`; four additional runs if each already has one seed |
| Core-only/raw/consolidated | 2–4 working days plus raw acquisition | Three views per seed; no credible total until their token lengths are measured |
| Tmall hierarchy maps | 0.5–1 working day | `maps×seeds×T_Tmall`; do not substitute ShortVideoAD timing |
| Four lambda values | 1–2 hours | Four fresh runs: about 89.26h assuming similar stopping lengths; reusing full/zero leaves two new runs, about 45.15h |
| Offline sparse buckets | 0.5–1 working day | Zero training runs; CPU minutes, pending actual timing |

Three seeds each for full/matched followed by one seed for each of the four remaining ablations requires **10 runs**; do not add the first two table rows as independent workloads. Adding the two missing lambda points makes 12 runs when full/zero are reused. Purely as equal-duration arithmetic, 10F=225.74h (about 9.4 days) and 12F=270.89h (about 11.3 days), assuming serial exclusive use of one 4×A800 allocation and identical workflow durations. Actual dense-FFN cost, early stopping, and standalone testing remain unmeasured. The 8×H800 8.63h reference cannot replace `D`.

## Lightweight verification completed

- Loader fix: eight Toy evaluation paths, default/True/False training settings, flake8, compileall, and diff check; committed as `38a90d2`.
- CPU verification of FixedSoft buffer values, freezing, and runtime multiplier=1.
- Actual model construction on the meta device for the parameter counts above.
- Tiny full, zero-prior, shared-FFN, and plain models passed finite-loss, backward-gradient, and cached-generation checks. These are not full-training results.
- Toy cold/warm-cache seed behavior was reproduced.
- FixedSoft/FixedBias per-user cvr test means match every summary metric.

The next reusable work is to recover final artifacts, add the two component options and seed restoration, freeze a run manifest, and run the key three-seed comparison. Raw-data semantics and upstream consolidation require separate evidence and cannot be established by those experiments.
