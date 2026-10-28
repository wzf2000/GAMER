# Full versus the recovered submission training recipe

Chinese counterpart: `2026-10-28-full-submission-recipe-comparison.zh.md`. This is a metadata/code audit; no training or ranking was launched.

## Evidence and principal difference

The user recovered the original invocation on 2026-10-28. It uses `train_SMB_decoder`, backbone `Qwen3TemporalHierarchical`, base config `Qwen3TemporalHierarchicalFixedSoft`, dataset ShortVideoAD, task `smb_explicit_decoder_4`, original `.index.json`, history 100, per-device batch 128, lr=5e-4, maximum 200 epochs, accumulation 8, warmup 0.04, patience 20, and temperature 0.7. The verbatim command is retained in the runtime audit under `results/rebuttal/2026-10-28-full-submission-comparison/`.

The original invocation does not include its distributed launcher. **Combining it with the user's previously supplied 4×A800-SXM4-80GB resource record for FixedSoft/factor4**, the original world size is 4. That hardware/world-size evidence is author supplied; it is not recovered from an old checkpoint. The new Full manifest independently establishes world size 2 on A100-SXM4-80GB.

| Setting | Recovered submission recipe | New Full |
| --- | --- | --- |
| Training hardware / world size | 4×A800 80GB, author resource record | 2×A100 80GB, manifest |
| Per-device training batch | 128 | 128 |
| Accumulation | 8 | 8 |
| **Nominal effective training batch** | **128×4×8 = 4096** | **128×2×8 = 2048** |
| Per-device validation batch | 128, legacy argument resolution | 128, effective HF args |
| Global validation batch | 512 under four ranks | 256 under two ranks |
| Learning rate | 5e-4 | 5e-4 |
| Warmup ratio | 0.04 | 0.04 |
| Epoch upper limit / patience | 200 / 20 | 200 / 20 |
| Temperature | 0.7 | 0.7 |
| History / task / index name | 100 / `smb_explicit_decoder_4` / `.index.json` | Same |

The main confirmed recipe mismatch is therefore **halving the effective optimizer batch**, alongside the different hardware/world size. It is not a change in the explicitly supplied learning rate, warmup ratio, accumulation, patience or temperature. The new manifest's legacy `per_device_batch_size=256` is superseded by its explicit HF `per_device_train_batch_size=128` and `per_device_eval_batch_size=128`; 128 is the actual value. The original legacy argument 128 resolves to both train/eval 128, so the spelling difference is not a microbatch mismatch.

With the current 238,270 training examples, the new run actually takes **117 optimizer updates per epoch** (step 8658 at epoch 74). A four-rank run with the same data/loader would take approximately **59**. The installed HF Trainer plans total scheduler steps using floor division of microbatches by accumulation: new `trainer_state.json` records **23,200** planned steps and warmup **928**; the corresponding four-rank calculation would be **11,600 / 464** if the old data and implementation matched. These old step counts are conditional calculations, not recovered training facts. Equal epoch/warmup-ratio settings do not make the optimizer trajectories identical when batch/world size changes.

## Model configuration and omitted defaults

The FixedSoft source config is unchanged from historical commits `0cad454` / `c682e1d` (June 2026). Resolving that template on the current ShortVideoAD index/tokenizer reproduces the prepared Full config exactly and gives **29,935,104 parameters**. It retains 8 decoder layers, TH layers 2–5, hidden size 256, Q/K/V behavior conditioning, output gate, position-routed FFN and behavior injection, frozen soft prior λ=0.05 with multiplier 1, and dropout/attention dropout 0.2. This establishes the intended template alignment; the original machine's final checkpoint config is still unavailable.

Historical and current parsers agree on omitted defaults: seed 42, AdamW (`adamw_torch`), cosine scheduler, weight decay 0.01, and FP32. The SMB task supplies model length **1024** despite the shared dataclass's generic 512 default. These are default-derived expectations, not independent inspection of the old `training_args.bin`.

The following are runtime resolution rather than evidence of a changed model: vocabulary 14 is a template placeholder expanded to 23,675; behavior maps/counts, positions and expert counts are filled from data/tokenizer; `model_type` serialization changes from `qwen3` to the shared `qwen3_moe` configuration class while the explicit backbone remains TH; training disables `use_cache` in both old and current code. The common parser's `sequence_augmentation=time_decay` does not select policy augmentation for `smb_explicit_decoder_4`; that task dispatches to the original random-augmentation dataset.

## Remaining limits and alignment

- No old checkpoint, trainer state, data/index/cache hashes or evaluation invocation was recovered. Actual old stopping/best epoch, exact dependency/backend/precision configuration and byte-level input identity remain unverified. Equal task/index names do not prove equal file contents.
- The original command's four-rank world size rests on the earlier author resource record. If that record belongs to another run, recompute batch from the actual launcher/world size.
- The old FixedSoft result files separately contain nominal test cvr HR@10/NDCG@10 **0.198087432 / 0.089980734**, and validation **0.204825680 / 0.091618456**. They lack run manifests; matching the manuscript row to its exact file remains a separate provenance check. Do not compare old validation against new test.
- All three new Full/plain/no-QKV runs use batch **2048**, so their internal recipe remains controlled. Treat comparison to an old batch-4096 Full as a different recipe.
- On two GPUs, matching nominal old training batch requires **accumulation 16**, keeping per-device batch 128. Using four GPUs with the recovered 128/8 settings also restores batch 4096. Any new component comparisons should use the same chosen recipe. Matching batch alone does not guarantee bitwise reproduction across different world sizes/hardware. No such rerun was launched by this audit.

Primary evidence: new `results/rebuttal/2026-10-26-p0/full/seed_42/checkpoint/run_manifest.start.json`, its `trainer_state.json`, `config/s2s-models/Qwen3TemporalHierarchicalFixedSoft/config.json`, historical `SeqRec/utils/args.py` and SMB task defaults, and the user-supplied command/resource record. The new Full best is epoch 54/step 6318, with early stopping at epoch 74.
