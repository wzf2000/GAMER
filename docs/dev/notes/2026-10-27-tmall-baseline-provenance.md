# Tmall baseline result provenance and behavior encoding

Chinese counterpart: `2026-10-27-tmall-baseline-provenance.zh.md`. Audit date: 2026-10-27; implementation checkout: `6013055`. Historical training revisions are not recorded in the supplied archives.

## Exact matches to the paper table

All three paper rows match **Behavior alipay**, not Merged Behavior, in the user-supplied `results_wzf.zip`. Paths below are relative to that archive's `results/Tmall-24-0.25/` directory.

| Paper row | Result path | HR@10 | NDCG@10 |
| --- | --- | ---: | ---: |
| SASRec | `smb_dis_sample/SASRec/result-smb_dis.json` | 0.07678938220887976 | 0.0533295220386152 |
| SASRec_B | `smb_dis_sample_diff/SASRec/result-smb_dis_diff.json` | 0.00647811660609891 | 0.004858649099901863 |
| TIGER | `smb/TIGER/results-smb-cid-shuffle-64.json` | 0.5809764575762364 | 0.4687695373957 |

Rounding to four decimals reproduces all six screenshot values. TIGER also has 6,329 per-user metric entries in this archive; their means reproduce every summary metric.

Two other TIGER files in `results_ygk.zip` must be kept separate:

- `smb_explicit/TIGER/results-cid-shuffle-64.json`: alipay HR@10 **0.5732785849652559**, NDCG@10 **0.4711377563057159**. This is the explicit-behavior variant, not the paper's 0.5810 / 0.4688 row.
- `smb/TIGER/results-smb-cid-shuffle-64.json`: alipay HR@10 **0.5809659090909091**, NDCG@10 **0.46870820571939775**. Its NDCG rounds to 0.4687, while the wzf artifact rounds to the screenshot's 0.4688. The archives contain different result bytes at the same relative path; they must not overwrite one another.

## Meaning of the identified tasks in current code

**SASRec, `smb_dis_sample`:** `SMBDisDataset(diff=False)` maps an event `(behavior, item)` to `item + 1`. Histories contain the mixed behavior sequence, but the model receives atomic item embeddings plus position embeddings without an explicit behavior distinction. The collator also supplies behavior metadata, which SASRec's forward/scoring methods do not consume. Training is not filtered to alipay-only events by this task.

**SASRec_B, `smb_dis_sample_diff`:** the same `SASRec` class and source config are used with `SMBDisDataset(diff=True)`. If `M = max_original_item_id + 1`, an event maps to `behavior_index * M + item + 1`. In Tmall the behavior enumeration is click, collect, cart, alipay. Thus each item has separate atomic IDs/embedding rows for those four identities; collect/cart remain distinct even though their hierarchy levels are tied. This is a joint behavior-item vocabulary, not a separate behavior embedding added to a shared item embedding. The output vocabulary also expands to four item-ID blocks. Current `smb_dis_diff` test filtering restricts scores to the requested behavior block through `item_range`.

**TIGER, `smb`:** `SMBDataset` represents each event only by its item-token sequence and does not add behavior tokens. It uses the mixed history, not an alipay-only history. The matching filename explicitly identifies **CID, shuffled, chunk size 64**; the script's corresponding index suffix is `.index.cid.shuffle.chunk64.json`. The historical index contents/hash are not provided, so the filename does not prove equality with a currently available index file. This row should not be described as a verified SID-tokenized TIGER run.

**Alternative TIGER, `smb_explicit`:** `SMBExplicitDataset` represents an event as `<behavior_click>` / `<behavior_collect>` / `<behavior_cart>` / `<behavior_alipay>` followed by the same item's token sequence. T5 learns those tokens through its ordinary shared token embedding. At encoder-decoder evaluation, the decoder prompt contains the start token followed by the requested behavior token. There is no TH prior or TH-specific Q/K/V behavior mechanism in TIGER.

Relevant implementation: `SeqRec/datasets/loaders/session_behavior_discriminative.py`, `SeqRec/datasets/discriminative/session_behavior.py:369`, `SeqRec/models/discriminative/SASRec/model.py:67`, `SeqRec/modules/model_base/seq_model.py:101`, `SeqRec/datasets/session_behavior/explicit.py`, and `SeqRec/tasks/evaluation/helpers.py:93`.

## Sampling, selection, and remaining limits

- In both `smb_dis_sample` tasks, training still uses `SMBDisDataset`; source config uses full-vocabulary CE. `sample` selects a sampled **validation** dataset, not a sampled training objective.
- The validation sampler defaults to 1,000 negatives, excludes pre-test observed items, and labels negative behavior as the target behavior. The collator prepends positives and truncates the combined candidate list to **1,000 total candidates**. Validation is filtered to alipay for this metadata; the trainer selects the last requested metric, default NDCG@10.
- Test loading is controlled independently by `test_task`. The recovered filenames are `result-smb_dis.json` and `result-smb_dis_diff.json`, matching the full-sort test paths. The training script does not automatically synchronize `test_task` with `tasks`; recover the original command to independently establish the exact historical execution. The historical test split is supported by task names/current loader semantics, not a retained `.run.json` manifest.
- Current source defaults: both SASRec variants use `config/dis-models/SASRec/config.json` (2 layers, 2 heads, hidden 128, FFN 256, dropout 0.5, CE); TIGER uses `config/s2s-models/TIGER/config.json` (4 encoder and 4 decoder layers, hidden 128, 6 heads, FFN 1024, dropout 0.1). The archives have no baseline checkpoints, configs, training arguments, tuning logs, or run manifests to establish these as final historical settings.
- Current discriminative SMB data loading requires exactly one maximum-level behavior. `Tmall-24-0.25` has alipay=2; its V2 metadata has three maximum-level behaviors and cannot be substituted directly into this loader.

Small CPU checks verified item-ID encoding, SASRec's independence from the unused behavior side channel, the diff behavior-block scoring mask, TIGER event-string encoding, and the 1,000-candidate validation collation. No source code changed and no training/evaluation jobs were launched for this audit.

## Preserved local evidence

Selected files were copied byte-for-byte into `results/rebuttal/2026-10-27-tmall-baseline-provenance/`, retaining separate `results_wzf/` and `results_ygk/` subdirectories. `audit.json` records archive/file SHA256 values, exact matches, and verification limits. Runtime evidence remains local and is not committed.
