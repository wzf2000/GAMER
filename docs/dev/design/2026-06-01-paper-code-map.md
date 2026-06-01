# Paper-to-Code Map for GAMER

## Source

- Paper: "Generative Sequential Recommendation via Hierarchical Behavior Modeling"
- arXiv PDF: https://arxiv.org/pdf/2511.03155
- arXiv version inspected: arXiv:2511.03155v1, dated 2025-11-05 in the PDF footer.

## Paper Summary

The paper presents GAMER for session-wise multi-behavior generative recommendation. The main method components are:

- Session-wise multi-behavior task formulation and evaluation.
- Multi-behavior sequential augmentation, using behavior-aware dropout of auxiliary behaviors to create additional training samples.
- A decoder-only Qwen3-based model with a Qwen3 MoE block.
- Cross-level behavior interaction, where higher-level behavior tokens attend to lower-level historical behavior tokens through a behavior-level mask.
- Position-and-behavior-aware MoE, where semantic ID token positions and behavior information control the feed-forward expert path.
- SID/CID/RID item tokenization for generative recommendation.

## High-Level Implementation Map

| Paper concept | Main implementation |
| --- | --- |
| GAMER main model | `SeqRec/models/generative/Qwen3Multi/model.py` |
| GAMER model config | `config/s2s-models/Qwen3Multi/config.json` |
| Qwen3 baseline/ablation | `SeqRec/models/generative/Qwen3/model.py`, `config/s2s-models/Qwen3-Light/` |
| Session-wise generative training | `SeqRec/tasks/train_SMB_decoder.py`, `scripts/train_SMB_decoder.sh` |
| Session-wise generative evaluation | `SeqRec/tasks/test_SMB_decoder.py`, `scripts/test_SMB_decoder.sh` |
| Session-wise generative dataset | `SeqRec/datasets/SMB_dataset.py`, `SeqRec/datasets/loading_SMB.py` |
| Discriminative baseline training/evaluation | `SeqRec/tasks/train_SMB_rec.py`, `scripts/train_SMB_rec.sh` |
| Discriminative baseline datasets | `SeqRec/datasets/SMB_dis_dataset.py`, `SeqRec/datasets/loading_SMB_dis.py` |
| Item tokenization | `SeqRec/tasks/tokenize.py`, `scripts/tokenize.sh` |
| RQ-VAE tokenizer model | `SeqRec/models/tokenizer/RQVAE/` |
| Constrained generation | `SeqRec/generation/trie.py` |
| HR/Recall/NDCG metrics | `SeqRec/evaluation/ranking.py` |
| Rule baseline | `SeqRec/tasks/test_SMB_rule.py`, `scripts/test_SMB_rule.sh` |

## Method Modules

### Session-Wise Protocol

The session-wise split is implemented in `BaseSMBDataset._load_data()`:

- Last session becomes test.
- Penultimate session becomes validation.
- Earlier sessions become training.
- Behavior levels are loaded from `<dataset>.behavior_level.json`; the max-level behavior is treated as the target behavior.

Relevant code:

- `SeqRec/datasets/SMB_dataset.py`: session parsing and split positions.
- `SeqRec/datasets/SMB_dis_dataset.py`: equivalent split for discriminative models.

This maps to the paper's session-wise leave-one-out protocol. During evaluation, all items of the target behavior inside the held-out session form the positive target set.

### Multi-Behavior Sequential Augmentation

The paper's sequential augmentation is mainly implemented by:

- `SMBExplicitDatasetForDecoder` in `SeqRec/datasets/SMB_dataset.py`.
- `load_SMB_datasets()` in `SeqRec/datasets/loading_SMB.py`.
- The script task name pattern `smb_explicit_decoder_<N>`.

When `tasks=smb_explicit_decoder_4`, `loading_SMB.py` parses `augment=4` and constructs `SMBExplicitDatasetForDecoder(..., augment=4)`. The dataset adds the original sequence plus augmented copies. For non-target behaviors, `_augment_interactions()` computes a per-behavior downsampling ratio based on the augmentation step and behavior level, then drops sampled auxiliary behavior interactions. Target behavior interactions are not dropped.

The README example:

```bash
dataset=ShortVideoAD original=1 batch_size=1024 tasks=smb_explicit_decoder_4 gpu=0,1,2,3,4,5,6,7 backbone=Qwen3Multi extra_args=max_his_len=100,gradient_accumulation_steps=4,warmup_ratio=0.04,patience=20 bash ./scripts/train_SMB_decoder.sh
```

### Qwen3 MoE Block and GAMER Architecture

The GAMER implementation is `Qwen3MultiWithTemperature`, which wraps `Qwen3MultiModel`.

Key implementation points:

- `Qwen3MultiAttention` implements normal self-attention and cross-level behavior attention. When `is_cross=True`, it adds behavior embeddings into Q/K/V and applies a SiLU-gated output.
- `Qwen3MultiDecoderLayer` runs self-attention, then optional cross-attention, then an MoE/FFN block.
- `Qwen3MultiModelBase` constructs decoder layers using config lists:
  - `sparse_layers_decoder`
  - `behavior_injection_decoder`
  - `cross_attention_decoder`
- `Qwen3MultiModel` builds the session/self mask and cross-level behavior mask.

Default `Qwen3Multi` config:

- 8 decoder layers.
- Hidden size 256.
- Intermediate size 512.
- 6 attention heads.
- Sparse layers: 0 through 7.
- Behavior injection layers: 0 through 3.
- Cross-attention layers: 4 through 7.
- Dropout and attention dropout: 0.20.
- `mlp_type: Qwen3`.

This means the current default implementation uses behavior-aware sparse MLP in all layers, injects behavior embeddings into the MLP in the first half, and applies cross-level behavior attention in the second half.

### Cross-Level Behavior Interaction

The paper describes cross-level behavior interaction as behavior-aware attention with a mask that allows higher-level behavior tokens to interact with lower-level historical behavior tokens.

Code mapping:

- Behavior embeddings for Q/K/V: `Qwen3MultiAttention.__init__()` and `Qwen3MultiAttention.forward()`.
- Gated cross-attention output: `Qwen3MultiAttention.forward()`.
- Behavior-level cross mask: `Qwen3MultiModel._compute_action_block_mask()`.
- Cross mask update during train/generation: `Qwen3MultiModel._update_session_multi_cross_mask()`.

The default `cross_mask_type` is `level`. In this setting, the code blocks keys whose level is greater than or equal to the query level, so a query token can attend only to lower-level behavior keys after also respecting the in-item causal block. The code also contains ablation variants:

- `causal`: no action-level gating.
- `reverse`: reverse behavior-level direction.
- `geq`: relaxed same-level access.
- `soft`: continuous behavior-level bias.

These variants correspond to likely ablation configs such as:

- `config/s2s-models/Qwen3MultiCausal/config.json`
- `config/s2s-models/Qwen3MultiReverse/config.json`
- `config/s2s-models/Qwen3MultiGeq/config.json`
- `config/s2s-models/Qwen3MultiSoft/config.json`

### Position-and-Behavior-Aware MoE

The position/behavior routing is implemented by:

- `SeqRec/models/generative/Qwen3Multi/router.py`
- `SeqRec/models/generative/Qwen3Moe/FFN.py`

`Qwen3MultiDecoderRouter` maps each token to:

- `position_indices`: token position within a behavior-item representation.
- `behavior_indices`: behavior id for behavior-injected MLP.
- `action_indices`: behavior level/id used by cross-attention.

`MyQwen3SparseMLP` and `PBATransformerSparseMLP` use `position_index` to choose fixed experts when `is_sparse=True`. If behavior injection is enabled, a learned behavior embedding is concatenated before the expert MLP. `RouterMoeBlock` is also available as a learned top-k routing ablation, but the default `Qwen3Multi` config uses `mlp_type: Qwen3`.

## Training Pipeline

Main script:

- `scripts/train_SMB_decoder.sh`

Task:

- `python main.py train_SMB_decoder ...`

Flow:

1. The shell script maps `backbone` to a base config directory.
2. It chooses the item index file:
   - original SID: `.index.json`
   - RQ-VAE SID: `.index.epoch<epoch>.alpha<alpha>-beta<beta>.json`
   - CID: `.index.cid[.shuffle].chunk<chunk_size>.json`
   - RID: `.index.rid.json`
   - RQ-Kmeans: `.index.rq-kmeans*.json`
3. It builds `output_dir` under `checkpoint/SMB-decoder/...`.
4. It converts `extra_args` and `extra_flags` into CLI args.
5. It launches `python main.py train_SMB_decoder` for one GPU or `torchrun` for multiple GPUs.
6. `TrainSMBDecoder` loads tokenizer/config, dataset, collator, and model.
7. `TrainSMBDecoder` adds item and behavior tokens to the tokenizer and saves config/tokenizer into the checkpoint directory.
8. Hugging Face `Trainer` runs training with early stopping and saves the best model.

Important implementation details:

- Decoder-only models (`Qwen3`, `Qwen3Session`, `Qwen3Multi`, `Qwen3SessionMulti`, `LlamaMulti`) use `DecoderOnlyCollator`.
- Encoder-decoder models (`TIGER`, `PBATransformer`) use `EncoderDecoderCollator`.
- `Qwen3Multi` passes `session_ids`, `extended_session_ids`, and `actions` as labels/input fields.
- The training loss is next-token causal LM loss, with temperature scaling in `Qwen3MultiWithTemperature.loss_function`.

## Evaluation Pipeline

Main script:

- `scripts/test_SMB_decoder.sh`

Task:

- `python main.py test_SMB_decoder ...`

Flow:

1. The shell script maps `backbone`, `ckpt_path`, `results_file`, and `index_file`.
2. `TestSMBDecoder` loads the checkpoint tokenizer/model.
3. It loads the session-wise test dataset and filters one dataset per behavior.
4. It builds candidate tries for constrained beam search:
   - all behavior-item candidates
   - one behavior-specific trie per behavior
5. It appends the behavior token for the requested behavior, generates item semantic ID tokens with beam search, and decodes generated IDs to item strings.
6. It computes HR, Recall, and NDCG, then writes JSON results.

The default script uses `num_beams=20`, matching the paper's constrained beam-search evaluation setting.

## Baseline Mapping

| Paper baseline/category | Current implementation status |
| --- | --- |
| Rule-Based | `SeqRec/tasks/test_SMB_rule.py`, `scripts/test_SMB_rule.sh` |
| GRU4Rec | `SeqRec/models/discriminative/GRU4Rec/`, trained through `train_SMB_rec` |
| SASRec | `SeqRec/models/discriminative/SASRec/`, trained through `train_SMB_rec` |
| BERT4Rec | `SeqRec/models/discriminative/BERT4Rec/`, trained through `train_SMB_rec` |
| SASRecB/BERT4RecB style behavior-item remapping | Likely handled by `SMBDisDataset` task variants in `loading_SMB_dis.py`; verify task naming before reproducing tables |
| PBAT | `SeqRec/models/discriminative/PBAT/`, trained through `train_SMB_rec` |
| MBHT | `SeqRec/models/discriminative/MBHT/`, trained through `train_SMB_rec`; code filters training/eval to target behavior for MBHT compatibility |
| MB-STR | `SeqRec/models/discriminative/MBSTR/`, trained through `train_SMB_rec` |
| TIGER | `SeqRec/models/generative/TIGER/`, trained through `train_SMB_decoder` or related decoder tasks |
| TIGERMB | Represented by using `backbone=TIGER` with explicit multi-behavior dataset/task variants |
| MBGen / PBATransformer | `SeqRec/models/generative/PBATransformer/`, trained through `train_SMB_decoder` with `backbone=PBATransformer` |
| Qwen3 architecture ablation | `SeqRec/models/generative/Qwen3/`, `backbone=Qwen3` |
| GAMER | `SeqRec/models/generative/Qwen3Multi/`, `backbone=Qwen3Multi` |
| MB-GMN, S-MBRec graph baselines | External baselines only. They are not implemented or maintained in this repository and can be ignored for in-repo development |

## Tokenization Mapping

The paper compares semantic IDs and chunked IDs. The implementation supports:

- RQ-VAE SID: `Tokenize.run_rq_vae()`, output `.index.epoch<epoch>.alpha<alpha>-beta<beta>.json`.
- RQ-Kmeans: `Tokenize.run_rq_kmeans()`, output `.index.rq-kmeans*.json`.
- CID: `Tokenize.run_CID()`, output `.index.cid[.shuffle].chunk<chunk_size>.json`.
- RID: `Tokenize.run_RID()`, output `.index.rid.json`.

Training and testing scripts select the same index naming convention, so tokenization settings must match between tokenizer generation, training, and evaluation.

## Reproduction Entry Points

Main GAMER example from the README:

```bash
dataset=ShortVideoAD original=1 batch_size=1024 tasks=smb_explicit_decoder_4 gpu=0,1,2,3,4,5,6,7 backbone=Qwen3Multi extra_args=max_his_len=100,gradient_accumulation_steps=4,warmup_ratio=0.04,patience=20 bash ./scripts/train_SMB_decoder.sh
```

```bash
dataset=ShortVideoAD original=1 batch_size=256 tasks=smb_explicit_decoder_4 gpu=0,1,2,3,4,5,6,7 backbone=Qwen3Multi extra_args=max_his_len=100 bash ./scripts/test_SMB_decoder.sh
```

Useful ablation switches:

- `backbone=Qwen3`: Qwen3 decoder-only architecture without GAMER cross-level behavior modeling.
- `backbone=PBATransformer`: MBGen-style encoder-decoder architecture.
- `tasks=smb_explicit_decoder`: no augmentation for the decoder dataset.
- `tasks=smb_explicit_decoder_4`: 4x sequential augmentation.
- `cid=1 chunk_size=64`: use CID tokenization instead of SID.
- `rid=1`: use random ID tokenization.

## Notes And Open Questions

- The current code has multiple Qwen3Multi config variants for cross-mask ablations. Keep future experiment notes tied to the exact config directory and commit hash.
- `session_ids` are passed through GAMER code paths, but the current Qwen3Multi self/cross masks mainly use token position and `actions`; session-aware variants live in separate `Qwen3Session*` model directories.
- Graph baselines mentioned in the paper appendix (`MB-GMN`, `S-MBRec`) are external to this repository. Do not spend in-repo development effort on them unless the user explicitly asks to import or reproduce them.
- Some paper table labels are conceptual; in this codebase, reproduction depends on the combination of `backbone`, `tasks`, `test_task`, and selected `index_file`.
