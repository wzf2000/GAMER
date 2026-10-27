## Scripts

### Basic Usage

We provide several shell scripts for training and evaluation under the `scripts/` directory. Please use them under the root directory of the project. For example:
```bash
bash scripts/train_SMB_decoder.sh
```

Each script contains some useful variables that can be modified by environment variables. For example, you can change the GPU device by setting the `gpu` variable:
```bash
gpu=0,1,2,3 bash scripts/train_SMB_decoder.sh
```

### Available Scripts

The available scripts are as follows:
- `generate_sem_emb.sh`: Script for generating semantic embeddings.
- `train_RQVAE.sh`: Script for training the `RQ-VAE` tokenizer.
- `tokenize.sh`: Script for tokenizing items into discrete tokens.
- `train_decoder.sh`: Script for training generative models on single-behavior datasets.
- `test_decoder.sh`: Script for evaluating generative models on single-behavior datasets.
- `train_MB_decoder.sh`: Script for training generative models on multi-behavior datasets
- `test_MB_decoder.sh`: Script for evaluating generative models on multi-behavior datasets.
- `train_SMB_decoder.sh`: Script for training generative models on session-wise multi-behavior datasets.
- `test_SMB_decoder.sh`: Script for evaluating generative models on session-wise multi-behavior datasets.
- `test_SMB_rule.sh`: Script for evaluating rule-based methods on session-wise multi-behavior datasets.
- `train_SMB_rec.sh`: Script for training and evaluating discriminative models on session-wise multi-behavior datasets.

### Script Arguments

Some general arguments for the above scripts are as follows:
- `port`: Port for distributed training. Only used when multiple GPUs are specified. Note you should ensure that the port is not occupied by other processes.
- `gpu`: GPU device(s) to use (e.g., `0`, `0,1`, etc.). Note for tasks that donot support multi-GPU training, only one GPU should be specified.
- `dataset`: Dataset name (e.g., `Beauty`, `Instruments`, etc.).
- `tasks`: Training task for the recommender. You should specify only one task. Note tasks for different datasets or models may be different (e.g., `mb_*` for multi-behavior datasets, `smb_*` for session-wise multi-behavior datasets, `smb_dis_*` for discriminative models on session-wise multi-behavior datasets, etc.). Please refer to the corresponding script for the available tasks.
- `test_task`: Test task for evaluation. You should specify only one test task. Note test tasks for different datasets or models may be different. Please refer to the corresponding script for the available test tasks.
- `backbone`: Backbone model for the discriminative recommender (e.g., `SASRec`, `BERT4Rec`, `GRU4Rec`, etc.) or generative model backbone (e.g., `TIGER`, `PBATransformer`, `Qwen3`, etc.).
- `batch_size`: Script-level total batch size. Training scripts convert it to the per-GPU batch size automatically. You can bypass this convenience layer by passing Hugging Face-style batch arguments after the script path, such as `--per_device_train_batch_size 64`.
- Training arguments: Pass training hyperparameters after the script path using the same argparse form as `main.py`. Prefer Hugging Face-style names for generative training, for example `bash scripts/train_SMB_decoder.sh --num_train_epochs 20 --learning_rate 3e-4 --warmup_ratio 0.04 --eval_strategy steps --save_strategy steps`.
- Model/task arguments: Pass additional task arguments after the script path, for example `bash scripts/train_SMB_decoder.sh --max_his_len 100 --debug`.
- Legacy extra task arguments: `extra_args="arg1=val1,arg2=val2"` and `extra_flags="flag1,flag2"` are still supported for old commands, but the trailing CLI form is preferred for new usage.

For other arguments, please refer to the corresponding script files in the `scripts/` directory and the corresponding task files in the `SeqRec/tasks/` directory.

### Rebuttal Experiment Tools

- `prepare_remaining_rebuttal.py`: Bootstrap machine-local `remaining-p0/plan.json` and `remaining-seeds/plan.json` without any existing `results/` directory. For example, `python scripts/prepare_remaining_rebuttal.py --gpu 2,5 --port 24567 --prepare_cache` generates plans and sequentially prepares CPU caches, never trains. Accumulation is derived from effective batch 2048 and the chosen device count/microbatch. Existing plan directories, missing inputs and LFS pointers are rejected. Optional `--groups tmall` requires its separately supplied CID index.
- `prepare_rebuttal_experiments.py`: Prepare isolated TH-FixedSoft component, seed, hierarchy, or prior-scale configurations, run manifests, and shell commands. It does not launch jobs. For example, `python scripts/prepare_rebuttal_experiments.py --suite components --seeds 42 43 44 --key_seeds_only --gpu 5,7 --output_dir ./results/rebuttal/new-plan` prepares ten runs; choose available GPUs and align hyperparameters before executing them.
- `summarize_rebuttal_results.py`: Aggregate a prepared plan's ranking results with an explicit `--split validation|test`, behavior, sample standard deviations, and paired seed differences. It checks execution records and data/configuration fingerprints.
- `prepare_rebuttal_history_views.py`: Export core-only/raw/consolidated SMB inputs from supplied real raw event JSONL, consolidated provenance, a shared session/target manifest, and a common item index. It cannot recover raw logs from consolidated states.
- `analyze_sparse_metrics.py`: Bucket existing per-user metrics on CPU without model inference. It requires explicit split/history length and UID sidecars, or an explicit legacy ordering assumption for older files.
- `audit_rebuttal_parameters.py`: Probe an independent model instance on CPU by default, with the final resolved vocabulary. Reports registered, forward-executed, loss-connected and nonzero-gradient tensor counts; accepts actual collated batches and an optional completed checkpoint. It never runs an optimizer or rewrites training caches/checkpoints.
- `manage_rebuttal_queue.py`: Refresh a handoff plan's process/artifact status, optionally bind parameter audits, and print/write the next single command. It never launches jobs; inaccessible process namespaces leave started jobs explicitly unconfirmed. GPU IDs are supplied by the user and are not treated as a claim of availability.
- `run_rebuttal_experiment.py`: Select exact variants/seeds from a local plan and run training, validation ranking and test ranking in order. Dry-run is the default; `--execute` starts the selected jobs. Existing outputs are refused and a failed phase stops subsequent phases/jobs. Use `--phases validation test` for completed weights. This explicit runner does not infer another machine's job state or schedule stage dependencies.

The prepare/queue/pipeline tools accept `--gpu 2,5` (alias `--gpus 2,5`) for nonconsecutive device IDs. In `run_rebuttal_experiment.py`, this overrides both training and ranking devices without rewriting the plan; `--port 24567` overrides the distributed port. Preserve the plan's training device count, or regenerate the plan with adjusted accumulation to retain the effective batch. For example: `python scripts/run_rebuttal_experiment.py --plan results/rebuttal/remaining-p0/plan.json --variants no_gate --seeds 42 --gpu 2,5 --port 24567 --execute`.

Run each script with `--help` for all arguments. Plan and report paths must be new; existing outputs are not overwritten. Training-time `--validation_behavior` is optional and defaults to the existing all-behavior validation loss. Plans enable `--record_run_manifest`; actual SMB ranking also writes split/result provenance, and per-user output adds UID sidecars while retaining the existing metric-array format.

`prepare_rebuttal_experiments.py --suite handoff --seeds 42` prepares the seven required configurations in P0-A/P0-B order, deduplicates Full/lambda=0.05, and enables optional `--export_user_targets` in ranking commands. Use `--reuse_plans <plan.json> ...` to reuse compatible existing run paths after configuration/data/recipe checks. Its stage scripts print the next pending command after a fresh status check; `manage_rebuttal_queue.py --output_script <new.sh>` writes that command for deliberate execution later. Other suites retain their existing launch-script behavior.

SMB ranking's `--export_user_targets` is off by default. When enabled, it attaches legacy validation UIDs only in memory after checking cached targets and history, and writes `user_level_metrics_<behavior>.targets.json` with original target item IDs, event order, repeated targets, SID collisions and metric/UID checksums. Existing metric arrays and caches keep their formats. Add `--csv_file <new.csv>` to `summarize_rebuttal_results.py` for actual per-run metrics and matching-seed differences; `--allow_partial` explicitly permits missing-only reports with blank CSV values.
