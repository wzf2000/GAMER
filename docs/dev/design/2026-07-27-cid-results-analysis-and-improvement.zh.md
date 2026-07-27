# GAMER CID 结果分析与效果改进台账

## 1. 文档目的与结论状态

本文档统一整理 GAMER 在 ShortVideoAD、Tmall 和 JData 上使用 chunked item ID（CID）的现有结果，核查训练、验证、测试、augmentation、checkpoint 和 baseline 口径，并给出投稿前后分层执行的效果改进建议。

本文档中的论文方法统一称为 **GAMER**。
结果目录中的 `Qwen3TemporalHierarchicalFixedSoft` 仅是内部实现配置名，表示当前冻结的 fixed-soft hierarchy-prior 配置，不作为论文方法名。
旧架构 CID 数值只用于来源追溯，不作为论文方法或当前可比结果。

截至 2026-07-27，本地能确认的最新三份 GAMER CID 汇总结果均位于：

```text
results/<dataset>/smb_explicit_decoder_4/
  Qwen3TemporalHierarchicalFixedSoft/
  results-smb_explicit-cid-shuffle-64.json
```

三份文件均包含 behavior-wise 和 micro-pooled 指标，并附有 behavior-wise user-level metric vectors。
数值及 user-level vector 内部一致性已经验证，但本地没有相应 checkpoint、`trainer_state.json`、训练命令或运行日志。
因此，本文将“结果数值已确认”和“训练/选点 provenance 已确认”严格区分：

- **数值状态：可用。** 汇总 JSON 与 user-level vectors 一致，差异仅为浮点累加误差。
- **测试口径：高度可信但仍需 manifest 固化。** 文件名与当前评测代码都指向 `smb_explicit` test mode、beam size 20、CID shuffle chunk 64。
- **训练配置与 checkpoint 选点：部分确认。** 目录、中央 config 和当前脚本能说明预期配置，但缺少该次运行自己的配置快照、日志和 checkpoint 元数据。
- **论文结论：暂不在本文中改写。** 本文仅列出由当前数值能够或不能够支持的表述边界。

用户已于 2026-07-27 确认：ShortVideoAD 上 CID 弱于 SID 符合预期，现有 ShortVideoAD CID 结果保留。
因此，CID/SID 差距只作为 item-code 语义差异的描述性结果，不再列为异常、修复目标或投稿前补跑触发条件。

## 2. 结果来源索引

### 2.1 最新 GAMER CID 结果

| 数据集 | 汇总结果文件 | user-level vectors | 文件时间 | SHA-256 |
| --- | --- | --- | --- | --- |
| ShortVideoAD | `results/ShortVideoAD/smb_explicit_decoder_4/Qwen3TemporalHierarchicalFixedSoft/results-smb_explicit-cid-shuffle-64.json` | 同名无 `.json` 后缀目录中的 `user_level_metrics_{p3s,click,cvr}.json` | 2026-07-27 15:11:11 | `46c8086581e07e137832708d67e75b91bb648af942f66846273a70ae7d74b20b` |
| Tmall | `results/Tmall-24-0.25-V2/smb_explicit_decoder_4/Qwen3TemporalHierarchicalFixedSoft/results-smb_explicit-cid-shuffle-64.json` | 同名无 `.json` 后缀目录中的 `user_level_metrics_{click,collect,cart,alipay}.json` | 2026-07-27 15:10:34 | `d7d2194ba8af2d7b1ffc04f059a39c210e36fb31912ef0ad8e9209b7c2ed3550` |
| JData | `results/JData-V2/smb_explicit_decoder_4/Qwen3TemporalHierarchicalFixedSoft/results-smb_explicit-cid-shuffle-64.json` | 同名无 `.json` 后缀目录中的 `user_level_metrics_{pv,click,cart,collect,buy}.json` | 2026-07-27 15:10:34 | `e8e6bd64461c10efd19f9ea4b5f00a3064de368a34bb799c12a3bf8d5e80dcec` |

以上三个结果目录都被 `GAMER/.gitignore` 中的 `results` 规则忽略，不在 GAMER Git 历史中。
文件时间只能证明这些文件在本地出现的时间，不能代替服务器训练和评测时间。

### 2.2 配置、代码与口径证据

| 证据 | 路径 | 能确认的内容 | 不能确认的内容 |
| --- | --- | --- | --- |
| 当前内部模型 config | `config/s2s-models/Qwen3TemporalHierarchicalFixedSoft/config.json` | 8 层、hidden 256、inner 512、6 heads、TH layers 2--5、fixed soft table、scale 0.05、dropout 0.2 | 结果生成时 checkpoint 内保存的 config 是否与当前文件逐字一致 |
| 训练入口 | `scripts/train_SMB_decoder.sh` | 结果目录对应的预期 checkpoint path 与 token tag 解析方式 | 该次运行的完整命令、CLI overrides、GPU 数、实际 batch |
| 评测入口 | `scripts/test_SMB_decoder.sh` | 默认 `test_task=smb_explicit`、`ckpt_num=best`、beam size 20、结果命名规则 | 该次运行是否显式传入非默认 `ckpt_num` 或其他 CLI override |
| Tokenization 解析 | `scripts/lib/tokenization.sh` | `cid=1 shuffle=1 chunk_size=64` 对应 `cid-shuffle-64` 和 `.index.cid.shuffle.chunk64.json` | ShortVideoAD 本地缺失的实际 CID index 内容和 checksum |
| Train/valid loader | `SeqRec/datasets/loaders/session_behavior.py` | `smb_explicit_decoder_4` 仅增强 train；valid 使用原始 `SMBExplicitDataset` | 实际缓存文件、训练样本数和每用户实际 view 数 |
| Augmentation 实现 | `SeqRec/datasets/session_behavior/decoder.py` | factor 4 的 ratio 为 `1/4,2/4,3/4,1`；保留原序列；只保护 target behavior；随机种子在数据处理中固定为 42 | 三个训练运行是否使用同一代码 commit |
| Trainer 构造 | `SeqRec/tasks/training/helpers.py` | `load_best_model_at_end=True`、默认 epoch evaluation、early stopping、默认未设置 `metric_for_best_model` | 实际 best epoch、best metric、是否 resume、是否覆盖 root checkpoint |
| Test split 解析 | `SeqRec/datasets/loaders/session_behavior.py` | `smb_explicit` 加载 `mode="test"`；只有 `smb_explicit_valid` 才加载 `valid_test` | 结果文件是否曾被另一次测试覆盖 |
| 评测实现 | `SeqRec/tasks/evaluation/test_SMB_decoder.py` | behavior-wise test、micro-pooled merged、beam 20、全 item trie、8 项指标、user-level vectors | baseline 是否使用同一 commit 和完全相同 candidate trie |

中央模型 config 的 SHA-256 为：

```text
d1c52b66211ce53f85eb0eae71999b518554b4f4f64990898a0c5dc701baaeb1
```

### 2.3 CID index 与数据映射

| 数据集 | 当前结果 token tag | 本地 index 文件 | 映射 item 数 | CID 长度 | 状态 |
| --- | --- | --- | ---: | ---: | --- |
| ShortVideoAD | `cid-shuffle-64` | **本地缺失**：预期为 `data/ShortVideoAD/ShortVideoAD.index.cid.shuffle.chunk64.json` | TODO | TODO；必须以实际 index 文件为准 | 结果可读，index provenance 不完整 |
| Tmall | `cid-shuffle-64` | `data/Tmall-24-0.25-V2/Tmall-24-0.25-V2.index.cid.shuffle.chunk64.json` | 361,932 | 4 | 文件存在 |
| JData | `cid-shuffle-64` | `data/JData-V2/JData-V2.index.cid.shuffle.chunk64.json` | 17,100 | 3 | 文件存在 |

Tmall 和 JData CID index 分别生成于 2026-04-07 21:36:45 和 2026-04-07 21:33:13。
二者都使用 shuffle 后的 balanced base-64 chunk code。
三份结果中的 `collision_info` 均为零 collision，说明本次评测看到的候选 item token 序列没有碰撞。
零碰撞不等于训练和 baseline 的 CID 映射完全相同；公平比较仍需 baseline index checksum。

### 2.4 Baseline 数值来源

当前本地没有 Tmall/JData baseline 的原始 result JSON、user-level vectors 或运行日志。
因此 baseline 可比分成两级：

1. **ShortVideoAD MBGen CID：可追溯到恢复 CSV。**
   `outputs/shortvideoad_table_restore/restored_exp-ShortVideoAD.csv` 第 101 个 source row（CSV 行中的 `source_row=100`）记录内部实现名 `PBATransformers(CID)`，论文将其统一展示为 MBGen (CID)。
2. **Tmall/JData baselines：只能追溯到当前论文表。**
   `Paper-Draft/Tex/Tables/4.2.1.main_exp.tex` 记录 TIGER、MB-STR 和 MBGen (CID) 的 HR@10/NDCG@10。
   缺少原始结果文件，当前只能作为论文台账参照，不能完成 user-level 配对检验或 index checksum 比对。

ShortVideoAD 的设计文档中还存在另一组较弱 MBGen test 数值。
当前论文明确使用 validation-selected session-wise MBGen variant，因此本文的当前主比较采用论文表中的 `0.1009/0.1531/0.1090/0.0635`，不混入设计文档中的另一训练构造。

## 3. 统一实验口径台账

### 3.1 三个数据集的共同配置

| 字段 | 当前值 | 证据强度 | 备注 |
| --- | --- | --- | --- |
| 论文方法名 | GAMER | 已冻结 | 内部配置为 `Qwen3TemporalHierarchicalFixedSoft` |
| 训练 task | `smb_explicit_decoder_4` | 目录强证据 | factor 4 full-sequence random-ratio augmentation |
| 训练视图预算 | 每用户最多 1 条原始序列 + 4 条增强序列 | 代码强证据 | 短序列增强后长度小于 2 时会跳过，因此不保证每个用户恰好 5 条 |
| Augmentation ratios | `1/4, 2/4, 3/4, 1` | 代码强证据 | 非 target behavior 的删除比例为 `ratio/(level+1)` |
| Augmentation RNG | 数据处理固定 seed 42 | 代码强证据 | 与 CLI 模型 seed 分开；实际 cache 未保留 |
| Validation dataset | 原始、未增强 validation prefix | 代码强证据 | `SMBExplicitDataset`，不是 augmented validation |
| Test task | `smb_explicit` | 文件名与代码强证据 | 代码映射到 `mode="test"` |
| Test split | 最后 session | 论文与 loader 口径一致 | result JSON 自身不写 split |
| CID | shuffled balanced CID, chunk size 64 | 文件名与脚本强证据 | baseline 是否同一 index 待 checksum |
| Beam size | 20 | 当前 test script 与论文 | result JSON 自身不写 beam |
| 指标 | HR@1/5/10、Recall@1/5/10、NDCG@5/10 | result JSON 强证据 | behavior-wise 与 micro-pooled merged 均齐全 |
| Test aggregation | per-user multi-positive；merged 为按 behavior sample 数 micro average | 评测代码强证据 | 不是 macro behavior average |
| Model seed | 预期 42 | 当前默认值 | 实际命令/`trainer_state.json` 缺失 |
| Checkpoint 规则 | 预期 validation loss 最低的 root best model | 当前代码推断 | HF 未显式设 `metric_for_best_model`，默认使用 loss；实际 best epoch 未确认 |
| Early stopping | 预期 patience 20 | 当前默认与论文历史命令 | 实际 CLI override 未确认 |
| Epoch 上限 | 预期 200 | 当前默认 | 实际 CLI override 未确认 |
| Learning rate | 论文写 `5e-4`，当前默认也为 `5e-4` | 论文与代码一致 | 缺运行日志 |
| Global batch | 论文写 4096 | 仅论文证据 | 缺 GPU 数、per-device batch 和 accumulation 日志 |
| Max history length | 论文/历史命令通常为 100 | 间接证据 | result JSON 不记录，必须从 run manifest 确认 |
| 测试次数 | 当前每 dataset 仅见一份同名最终 CID 文件 | 文件系统观察 | 同名文件可覆盖，不能证明 test 只运行一次 |

### 3.2 数据集特定口径

| 数据集 | 运行 dataset key | Behavior 顺序 | 代码数值层级 | Target behavior | Target test 用户数 |
| --- | --- | --- | --- | --- | ---: |
| ShortVideoAD | `ShortVideoAD` | p3s, click, cvr | `0,1,2` | cvr | 8,784 |
| Tmall | `Tmall-24-0.25-V2` | click, collect, cart, alipay | `0,1,1,1` | alipay | 6,329 |
| JData | `JData-V2` | pv, click, cart, collect, buy | `0,1,2,2,2` | buy | 230 |

Target behavior 由当前 `BaseSMBDataset` 在最大数值层级中选择 JSON insertion order 的最后一个 behavior。
因此，Tmall 的 alipay 和 JData 的 buy 虽然与若干其他 behavior 共享最大数值层级，仍分别成为唯一 target behavior，并在 random-ratio augmentation 中受到保护。
上表“代码数值层级”仅指 `behavior_level.json`。
当前 FixedSoft 的 relation table、behavior-aware Q/K/V 和 behavior-aware MoE 实际使用 behavior token 的 insertion-order identity index，而不读取这组数值层级；详见 5.7。

## 4. 统一结果表

### 4.1 Target-behavior 完整结果

以下是论文 RQ1 最直接相关的 target behavior test 结果。
表中数值由原始 JSON 四舍五入到四位小数；括号内为 test 用户数。

| 数据集 / target | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShortVideoAD / cvr (8,784) | 0.0348 | 0.1193 | 0.1803 | 0.0256 | 0.0879 | 0.1330 | 0.0620 | 0.0771 |
| Tmall / alipay (6,329) | 0.3678 | 0.5611 | 0.5772 | 0.3320 | 0.5396 | 0.5576 | 0.4655 | 0.4719 |
| JData / buy (230) | 0.4174 | 0.6391 | 0.7000 | 0.4071 | 0.6200 | 0.6816 | 0.5244 | 0.5449 |

### 4.2 Behavior-wise 与 pooled 结果

| 数据集 | Behavior | N | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShortVideoAD | p3s | 43,618 | 0.0456 | 0.1416 | 0.2030 | 0.0177 | 0.0595 | 0.0919 | 0.0552 | 0.0651 |
| ShortVideoAD | click | 11,444 | 0.0315 | 0.0912 | 0.1325 | 0.0193 | 0.0588 | 0.0889 | 0.0462 | 0.0562 |
| ShortVideoAD | cvr | 8,784 | 0.0348 | 0.1193 | 0.1803 | 0.0256 | 0.0879 | 0.1330 | 0.0620 | 0.0771 |
| ShortVideoAD | pooled | 63,846 | 0.0416 | 0.1295 | 0.1873 | 0.0191 | 0.0633 | 0.0970 | 0.0546 | 0.0652 |
| Tmall | click | 215,673 | 0.1294 | 0.2462 | 0.2719 | 0.0751 | 0.1488 | 0.1654 | 0.1397 | 0.1433 |
| Tmall | collect | 15,816 | 0.0456 | 0.1031 | 0.1183 | 0.0382 | 0.0903 | 0.1033 | 0.0697 | 0.0743 |
| Tmall | cart | 15,747 | 0.1013 | 0.2072 | 0.2314 | 0.0853 | 0.1823 | 0.2050 | 0.1450 | 0.1531 |
| Tmall | alipay | 6,329 | 0.3678 | 0.5611 | 0.5772 | 0.3320 | 0.5396 | 0.5576 | 0.4655 | 0.4719 |
| Tmall | pooled | 253,565 | 0.1284 | 0.2427 | 0.2674 | 0.0798 | 0.1570 | 0.1738 | 0.1438 | 0.1478 |
| JData | pv | 4,361 | 0.2502 | 0.4375 | 0.4898 | 0.1979 | 0.3589 | 0.4115 | 0.3065 | 0.3240 |
| JData | click | 9,087 | 0.2570 | 0.4454 | 0.5006 | 0.1733 | 0.3224 | 0.3711 | 0.2892 | 0.3040 |
| JData | cart | 1,137 | 0.2269 | 0.4538 | 0.5092 | 0.1748 | 0.3778 | 0.4391 | 0.3061 | 0.3267 |
| JData | collect | 219 | 0.1461 | 0.3105 | 0.3607 | 0.1208 | 0.2681 | 0.3038 | 0.2096 | 0.2216 |
| JData | buy | 230 | 0.4174 | 0.6391 | 0.7000 | 0.4071 | 0.6200 | 0.6816 | 0.5244 | 0.5449 |
| JData | pooled | 15,034 | 0.2536 | 0.4447 | 0.4991 | 0.1833 | 0.3409 | 0.3917 | 0.2979 | 0.3140 |

### 4.3 与当前可比 external baselines 的 target 结果

| 数据集 | 方法 | ID | HR@5 | HR@10 | R@10 | N@10 | 来源与限制 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| ShortVideoAD | MBGen | CID | 0.1009 | 0.1531 | 0.1090 | 0.0635 | 恢复 CSV；同论文 test protocol，缺 user vectors |
| ShortVideoAD | MBGen | SID | 0.1012 | 0.1622 | 0.1205 | 0.0673 | 当前论文表；不同 item ID，不是 CID-only 公平隔离 |
| ShortVideoAD | GAMER | CID | **0.1193** | **0.1803** | **0.1330** | **0.0771** | 最新 CID JSON |
| ShortVideoAD | GAMER | SID | 0.1349 | 0.1981 | 0.1513 | 0.0900 | 当前最终 SID JSON；用于 ID 差异诊断 |
| Tmall | TIGER | 论文表未标 ID 后缀 | -- | 0.5810 | -- | 0.4687 | 当前论文表；原始 result 缺失 |
| Tmall | MBGen | CID | -- | 0.5721 | -- | 0.4672 | 当前论文表；原始 result 缺失 |
| Tmall | GAMER | CID | -- | 0.5772 | -- | **0.4719** | 最新 CID JSON |
| JData | TIGER | 论文表未标 ID 后缀 | -- | 0.6509 | -- | 0.4824 | 当前论文表；原始 result 缺失 |
| JData | MB-STR | 非生成式 | -- | 0.6522 | -- | 0.4646 | 当前论文表；metric-specific HR 最强 baseline |
| JData | MBGen | CID | -- | 0.6164 | -- | 0.4520 | 当前论文表；原始 result 缺失 |
| JData | GAMER | CID | -- | **0.7000** | -- | **0.5449** | 最新 CID JSON |

表中的粗体只用于标出该小组当前数值最高项，不代表已完成显著性检验。

### 4.4 相对差异

| 数据集 | 对比 | HR@5 | HR@10 | R@10 | N@10 | 解释 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ShortVideoAD | GAMER CID vs MBGen CID | +18.24% | +17.78% | +22.06% | +21.43% | CID-to-CID 的最直接 external baseline 证据 |
| ShortVideoAD | GAMER CID vs 当前 GAMER SID | -11.56% | -8.97% | -12.06% | -14.32% | 用户已确认符合预期；保留为 ID 语义差异，不作为异常 |
| ShortVideoAD | 当前 CID vs legacy CID 数值 | +3.03% | +2.52% | +2.98% | +0.67% | 仅说明最终架构 CID 高于旧架构 CID，不进入论文方法命名 |
| Tmall | GAMER CID vs MBGen CID | -- | +0.89% | -- | +1.00% | 对 MBGen 为小幅正增益 |
| Tmall | GAMER CID vs metric-specific strongest baseline | -- | -0.66% | -- | +0.68% | HR@10 不是 overall best；NDCG@10 小幅最好 |
| JData | GAMER CID vs MBGen CID | -- | +13.56% | -- | +20.56% | 对 MBGen 的增益较大 |
| JData | GAMER CID vs metric-specific strongest baseline | -- | +7.33% | -- | +12.96% | 两项均为当前表中最好 |

Tmall 和 JData 最新结果分别低于活动论文中的 legacy GAMER CID 占位数值：

- Tmall HR@10/NDCG@10：`-1.18%/-1.55%`。
- JData HR@10/NDCG@10：`-6.13%/-2.19%`。

这不是应被隐藏的负面结果。
使用与最终冻结架构对齐的新结果，即使低于旧占位数值，也比从不同架构结果中挑最高 test 值更符合投稿口径。

## 5. 数值与口径核查

### 5.1 已通过检查

1. **User-level vector 一致性。**
   三个数据集、所有 behaviors、全部八项指标的 vector length 均等于汇总 JSON 中对应 `collision_info.total`。
   vector mean 与汇总值的最大绝对差分别为：
   ShortVideoAD `1.60e-15`、Tmall `2.41e-14`、JData `1.67e-15`。
2. **Merged 指标一致性。**
   评测代码按 behavior test 样本数加权，汇总表中的 pooled 数值与该 micro-average 定义一致。
3. **指标完整。**
   三份结果均有 HR@1/5/10、Recall@1/5/10、NDCG@5/10，没有缺失指标。
4. **Test 样本数一致。**
   ShortVideoAD CID 与同一最终配置 SID 的 p3s/click/cvr test 数均为 `43,618/11,444/8,784`，说明 ID 变化没有改变 test 用户集合。
5. **CID token collision。**
   三个数据集所有 behavior 均报告 zero collision。
6. **评测 split 命名。**
   三份 CID 文件名都是 `results-smb_explicit-*`，不是曾经造成混淆的 `results-smb_explicit_valid-*`。
7. **最新对齐结果不是按最高 test 值挑选。**
   Tmall/JData 新结果低于旧占位结果，仍被识别为当前最终架构结果，符合“架构先冻结、再报告结果”的方向。

### 5.2 尚不能确认

1. **Checkpoint 的实际 best epoch 和选择指标。**
   当前代码预期用 validation loss 选择 root best model，但结果目录没有 `trainer_state.json`，也没有 `best_model_checkpoint`、`best_metric`、epoch 或 global step。
2. **是否显式测试过多个 checkpoint。**
   `test_SMB_decoder.sh` 允许用 `ckpt_num` 测指定 checkpoint，但结果文件名不包含 checkpoint number。
   因此多个 checkpoint 的测试可能覆盖同一个 JSON，文件系统无法排除该情况。
3. **完整训练命令和 overrides。**
   模型 seed、global batch、gradient accumulation、warmup、max history、epoch、resume、precision、GPU 数和代码 commit 均未随结果保存。
4. **ShortVideoAD CID index。**
   本地只有 SID `.index.json`，没有生成这些 CID 结果所需的 shuffled CID index。
5. **Baseline index 和评测实现版本。**
   当前 baseline 表值没有 index checksum、commit 或 user vectors，不能证明 MBGen CID 与 GAMER CID 使用逐字相同的 item-to-code mapping。
6. **Validation-selection 证据。**
   三个 CID run 没有 validation retrieval JSON，也没有 validation loss trajectory。
   不能验证超参数和 checkpoint 在 test 前已经冻结。
7. **训练视图的实际倍率。**
   代码会跳过长度小于 2 的增强视图，因此 factor 4 是最大新增 view 数，不保证每个用户严格产生 4 个有效增强 view。
   本地没有 CID cache 或 dataset summary log 来统计真实倍率。
8. **重复训练。**
   每个最终目录当前只看到一份结果，但结果被 gitignore 且可能覆盖；不能判断是否存在服务器端重复 run 或未同步 run。

### 5.3 配置命名歧义

- `smb_explicit_decoder_4` 表示新增 4 个 random-ratio full-sequence views，而不是总共 4 条序列。
- 代码里的 `augment=4` 最多形成 1 original + 4 augmented，即最多 5 条。
- 结果 token tag `cid-shuffle-64` 同时编码了 CID、shuffle 和 chunk size。
- `Qwen3TemporalHierarchicalFixedSoft` 只应出现在内部路径与实现说明中；论文统一称 GAMER。
- `Merged Behavior` 是按 test instance 数 micro-pooled，不是各 behavior 等权 macro average。
- Tmall/JData 目录中的 `V2` 是实际运行 dataset key；论文只写 Tmall/JData。
  需要在 supplement 的数据处理说明中明确 V2 与论文数据版本的对应关系。

### 5.4 数据版本、软链接与生成来源

本地实际存在三个 Tmall-family 目录和两个 JData-family 目录；不存在 `data/Tmall-V2/` 或其他第四个 Tmall 版本。

| 目录 | 交互格式 | 交互/行为/session/time/item 来源 | `behavior_level.json` | CID index | 结论 |
| --- | --- | --- | --- | --- | --- |
| `data/Tmall/` | legacy MB；无 session/time | 独立 regular files | `click:0, collect:1, cart:1, alipay:2` | 无当前 CID index | 与 Tmall-24 不是同一数据版本 |
| `data/Tmall-24-0.25/` | SMB | 独立 regular files | `click:0, collect:1, cart:1, alipay:2` | 无当前 CID index | 原始 3-level sessionized 版本 |
| `data/Tmall-24-0.25-V2/` | SMB | 六个文件均为指向 `../Tmall-24-0.25/` 的符号链接 | `click:0, collect:1, cart:1, alipay:1` | 独立 V2 CID index | V2 只改 level JSON 并增加 index |
| `data/JData/` | SMB | 独立 regular files | `pv:0, click:1, cart:2, collect:2, buy:3` | 无当前 CID index | 原始 4-level 版本 |
| `data/JData-V2/` | SMB | 五个文件均为指向 `../JData/` 的符号链接 | `pv:0, click:1, cart:2, collect:2, buy:2` | 独立 V2 CID index | V2 只改 level JSON 并增加 index |

符号链接解析后的 inode 与原文件一致。
例如 Tmall-24 的 inter/behavior 分别解析到 inode `103698122/103698081`，JData 的 inter/behavior 分别解析到 `103698190/103698191`。
关键 checksum 为：

| 文件对 | SHA-256 |
| --- | --- |
| Tmall-24 base/V2 `SMB.inter.json` | `6c1b1967d82314642b140399b25b6bb4e538fb4993f62c1948a1f4e1786a2d87` |
| Tmall-24 base/V2 `SMB.behavior.json` | `e18fa12fedd680b9f6703dd75fd2991a6a3e998a82d3bac92dbffcb24ca982b6` |
| JData base/V2 `SMB.inter.json` | `9f694dc8becb7d4d481137647e12c736e22909d9a0a25b80ce5a6bd58030744b` |
| JData base/V2 `SMB.behavior.json` | `87b426fb7e17c2b9de8f079ddcc5e616d2930386fc32c1be55da298a73066765` |
| Tmall base level / V2 level | `05c76d80...` / `fec91330...` |
| JData base level / V2 level | `1faa2d9c...` / `11f3db90...` |

Git 历史显示 Tmall-24 SMB 数据由 `f2820a3` 引入，JData 更新见 `2c492b6`；V2 由 `cb987b2` 引入并在 `ea90279`/`c108048` 中完成 LFS materialization 与文件名清理。
这些 commit 没有解释把 alipay/buy 与 collect/cart 合并为最大 level 的业务依据。
仓库也没有生成这些数据文件的预处理脚本或可复现 manifest。
因此可确认“V2 没改 interactions、session、time、item、filtering 或已保存序列”，但 Tmall-24 名称中的 `24-0.25` 分别对应什么生成参数仍为 **TODO：需要原始预处理命令或服务器日志**。

Tmall legacy 与 Tmall-24 明显不同：前者为 46,077 用户、3,816,709 事件、352,463 个实际 item，后者为 217,374 用户、3,818,122 事件、361,932 个实际 item；文件 checksum、用户切分和序列结构均不同。
因此不能把 `Tmall/`、`Tmall-24-0.25/` 和 `Tmall-24-0.25-V2/` 视为仅改名或仅改 index 的三个副本。

### 5.5 全量统计方法

本轮统计对本地 pretty-printed JSON 做逐用户流式解析，没有抽样，也没有把分析产物写入仓库。
临时脚本与 JSON 输出位于 `/tmp/cid_hierarchy_stats.py`、`/tmp/{tmall_legacy,tmall24,jdata}_hierarchy_stats.json`。

- 对 inter、behavior、session、time 四个文件按 uid 同步迭代，并逐用户校验数组长度。
- 时间判断使用已按 ISO datetime 存储的数组顺序；Tmall-24 与 JData 均为零相邻时间逆序。
- user-item 共现以同一用户内 item ID 分组，并同时统计任意先后、首次出现先后、同 session 与跨 session successive transition。
- counterfactual collapse 同时报告两种定义：全用户范围每个 user-item 保留一次，或每个 user-session-item 保留一次。
- collapse 选择最大映射 level；同 level tie 取时间上最后一条，再按被选事件的原位置序列化。
  这是明确记录的分析规则，不代表当前 loader 已实现该规则。
- Legacy Tmall 没有 session/time 文件，只能使用文件声明的存储顺序，不能做真实 session 内/跨 session 对比。

### 5.6 原始事件、覆盖与序列分布

| 数据版本 | 用户 | 事件 | 实际 item | 每用户事件 min/median/p90/p99/max | 平均 |
| --- | ---: | ---: | ---: | --- | ---: |
| Tmall legacy MB | 46,077 | 3,816,709 | 352,463 | 5 / 44 / 187 / 594.24 / 4,462 | 82.83 |
| Tmall-24 SMB（V2 同源） | 217,374 | 3,818,122 | 361,932 | 5 / 11 / 34 / 96 / 25,663 | 17.56 |
| JData SMB（V2 同源） | 10,010 | 1,643,212 | 17,100 | 3 / 91 / 381 / 1,106.73 / 3,558 | 164.16 |

Tmall-24 的 max `25,663` 是显著长尾，但该用户数据本身保持时间单调；训练时仍会被 `max_his_len` 截断。

| 数据 | Behavior | 事件数（占比） | 用户覆盖 | Item 覆盖 |
| --- | --- | ---: | ---: | ---: |
| Tmall-24 | click | 3,559,974（93.239%） | 217,373 | 361,878 |
| Tmall-24 | collect | 108,300（2.836%） | 47,209 | 76,156 |
| Tmall-24 | cart | 118,061（3.092%） | 53,644 | 88,223 |
| Tmall-24 | alipay | 31,787（0.833%） | 22,944 | 30,127 |
| JData | pv | 290,696（17.691%） | 9,760 | 12,078 |
| JData | click | 1,281,055（77.960%） | 9,809 | 16,615 |
| JData | cart | 56,661（3.448%） | 6,984 | 5,056 |
| JData | collect | 10,077（0.613%） | 2,566 | 2,936 |
| JData | buy | 4,723（0.287%） | 2,872 | 1,501 |

每用户深行为高度稀疏。
Tmall-24 的 collect/cart/alipay 中位数均为 0；JData 的 collect/buy 中位数均为 0，cart 中位数为 2。

| 数据 | Behavior | 每用户计数 median / p90 / p99 / max |
| --- | --- | --- |
| Tmall legacy | click / collect / cart / alipay | `40/173/551.24/4462`; `0/6/30/358`; `1/7/32/689`; `0/3/9/236` |
| Tmall-24 | click / collect / cart / alipay | `11/31/90/25663`; `0/2/7/76`; `0/2/7/82`; `0/1/2/34` |
| JData | pv / click / cart / collect / buy | `14/69/216.91/559`; `69/301/873.82/2926`; `2/15/49/178`; `0/3/17/69`; `0/1/4/23` |

### 5.7 当前实现对 mapping 的真实使用

源码核查修正了此前“V2 level JSON 直接决定全部 TH 计算”的假设。

| 组件 | 当前真实规则 | Base 与 V2 的差异 |
| --- | --- | --- |
| `BaseSMBDataset.target_behavior` | 取最大 numeric level 的 behaviors，并选择 JSON insertion order 最后一个 | Base 唯一最大；V2 多个最大但最后仍为 alipay/buy，伴随 warning |
| Deepest-state collapse | `BaseSMBDataset` 不执行；直接读取并序列化所有 `SMB.inter/behavior` 事件 | 无差异；V2 symlink 到完全相同的未 collapse 序列 |
| 同 level tie | loader 中不存在 collapse tie；本轮只在 counterfactual 统计中用“最后事件” | 若以后真正 collapse，V2 tie 会改变保留状态 |
| Behavior-aware Q/K/V | `Qwen3MultiDecoderRouter` 按 behavior token identity 建 embedding index | 不读取 `behavior_level`；Base/V2 无差异 |
| Behavior-aware MoE/FFN | 同样按 token identity index | 不读取 `behavior_level`；Base/V2 无差异 |
| FixedSoft relation prior | `training/helpers.py` 按 behavior token 列表顺序生成 `behavior_maps`；soft table 使用该 identity index 的差值 | Tmall 实际全序为 click < collect < cart < alipay；JData 为 pv < click < cart < collect < buy；Base/V2 无差异 |
| `actions` | `_generate_actions` 使用 numeric `behavior_level` | Base/V2 数值不同，但当前 replacement-style TH model 的 relation/QKV 路径重新从 token identity 生成 action index，未使用 collator 传入的 `actions` 做 relation bias |
| Factor-4 augmentation | 只保护 `target_behavior`；其他行为 drop 为 `ratio/(level+1)` | Base/V2 非 target 的 level 完全相同，target 又被保护，所以本次训练增强逐行为 drop rate 相同 |
| Test target | `filter_by_behavior(target_behavior)` | 两者仍评 alipay/buy；V2 的“max-level”集合却额外包括 collect/cart |

因此，对当前 `Qwen3TemporalHierarchicalFixedSoft + smb_explicit_decoder_4` 路径而言，base/V2 level JSON 的变化不会让 FixedSoft 或 behavior-aware Q/K/V 获得 2/3-level 结构，也不会改变本次 augmentation rate。
模型实际上按 behavior vocabulary insertion order 使用 4/5 个 identity ranks，并把本应并列的 collect/cart 强制排成相反的跨数据集全序。
这比“论文写 3/4 levels、实现只用了 2/3 levels”更准确，也更需要在后续实现中分离 behavior identity 与 behavior level。

### 5.8 user-item 共现、方向与缺失前驱

Tmall-24 有 2,593,623 个 user-item groups，单组事件数 median 1、p95 4；JData 有 382,250 个 groups，median 2、p95 14。
下表以“包含右侧行为的 user-item groups”为分母；“前驱在先”要求左侧行为在时间上早于至少一次右侧行为。

| 数据 | 候选关系 | 共现 | 前驱在先 | 反向也出现 | 完全缺前驱 |
| --- | --- | ---: | ---: | ---: | ---: |
| Tmall-24 | click → collect | 97.06% | 92.21% | 40.60% | 2.94% |
| Tmall-24 | click → cart | 98.55% | 95.45% | 61.96% | 1.45% |
| Tmall-24 | collect → cart | 10.19% | 4.26% | 6.54% | 89.81% |
| Tmall-24 | cart → collect | 9.77% | 6.27% | 4.08% | 90.23% |
| Tmall-24 | collect → alipay | 10.78% | 10.38% | 0.43% | 89.22% |
| Tmall-24 | cart → alipay | 69.81% | 69.57% | 1.94% | 30.19% |
| Tmall-24 | click → alipay | 99.40% | 98.24% | 50.67% | 0.60% |
| JData | pv → click | 35.89% | 28.65% | 24.06% | 64.11% |
| JData | click → cart | 90.84% | 77.04% | 84.65% | 9.16% |
| JData | click → collect | 95.70% | 76.52% | 89.82% | 4.30% |
| JData | cart → collect | 29.34% | 15.31% | 18.96% | 70.66% |
| JData | collect → cart | 8.01% | 5.17% | 4.18% | 91.99% |
| JData | cart → buy | 87.73% | 85.70% | 20.00% | 12.27% |
| JData | collect → buy | 11.12% | 9.76% | 1.57% | 88.88% |
| JData | click → buy | 95.52% | 91.46% | 80.67% | 4.48% |
| JData | pv → buy | 89.10% | 77.40% | 67.59% | 10.90% |

“反向也出现”并不等价于错误标注：同一 item 可重复访问，深行为后继续 click/pv 很常见。
它说明这些数据不是每个步骤只出现一次的严格 DAG path，不能用“所有购买都必须经历且只经历一次完整前驱链”解释。

### 5.9 Session 内与跨 session 转移

Tmall-24 有 2,961,366 个 session 内相邻转移和 639,382 个跨 session 相邻转移。
Session 内 click→click 占 86.16%，click→cart/collect 分别占 3.42%/3.19%；跨 session click→click 占 91.25%，collect/cart/alipay→click 合计约 7.19%。
同一 user-item 的 successive transitions 中，session 内 click→cart/collect/alipay 分别占 10.96%/9.90%/2.00%，而跨 session 深行为回到 click 更常见。

JData 有 1,447,352 个 session 内相邻转移和 185,850 个跨 session 相邻转移。
Session 内 click→click、pv→click、click→pv 分别占 62.84%/12.83%/12.02%；跨 session 分别为 59.23%/11.68%/16.40%。
同一 user-item 的 session 内 pv→click 与 click→pv 分别占 14.88%/11.26%，跨 session分别为 10.85%/15.64%。
pv/click 的双向频繁转移与大量同 timestamp 事件（JData 相邻等时 188,467 次）共同说明 `pv < click` 不是数据强制的必经全序。

### 5.10 Candidate mapping、collapse 与 target sparsity

#### Tmall

业务语义与数据共同支持：

```text
click < {collect, cart} < alipay
```

- collect 是收藏/愿望清单，cart 是加入购物车；二者都是较强购买意图，但属于替代分支，不是稳定必经顺序。
- cart 对 alipay 的经验前驱覆盖显著高于 collect（69.57% vs 10.38%），说明预测强度不同，但不构成 `collect < cart` 的必经偏序。
- alipay 是实际支付，必须单独作为最深业务目标；直接购买或缺 cart/collect 不否定它的终态语义。

原始 base/paper mapping `0,1,1,2` 的置信度为 **高**。
V2 `0,1,1,1` 在业务上不合理；collect/cart 分层的两个全序候选也缺少共现方向支持。

在 base mapping 下，全局 user-item collapse 后最高状态为 click/collect/cart/alipay：
`2,384,051/102,288/76,225/31,059`，level 占比 `91.920%/6.883%/1.198%`。
每 session collapse 将总事件从 3,818,122 降至 2,864,348（-24.98%），每用户长度 median 从 11 降至 9；全局 collapse 降至 2,593,623（-32.07%），median 为 8。
Base test 最深 alipay 为 `7,906/874,645=0.904%`，覆盖 6,329 个用户。
V2 把 collect/cart 也算 max level 后，max-level test 占比虚增至 6.395%，但 target 仍只评 alipay。

若未来按 V2 真正 collapse 并用“同 level 取最后事件”，全局最高 alipay 会从 31,059 降至 30,641；test collapsed target 用户会从 6,329 降至 6,298。
这表明 V2 不适合作为 deepest-state collapse mapping。

#### JData

首选偏序为：

```text
pv < click < {collect, cart} < buy
```

其中 `pv/click` 同层是有实证依据的备选：

```text
{pv, click} < {collect, cart} < buy
```

- pv 是页面曝光/浏览，click 通常表示更主动选择，因此业务上 `pv < click` 合理。
- 但只有 28.65% 的 click-item groups 能观察到更早 pv，且 pv↔click 双向转移接近，因此这一级的统计置信度仅为 **中等**；若日志埋点使 click 自动伴随 pv，二者同层更稳健。
- collect/cart 仍是并列意图分支；强行分层的两个候选都缺少方向覆盖。
- buy 是实际交易，必须单独最深；即使 12.27% 的 buy-item groups 没有 cart、88.88% 没有 collect，也只说明这些前驱不是必经步骤。

原始 base/paper mapping `0,1,2,2,3` 的综合置信度为 **中高**；`pv/click` 同层为 **中等置信备选**；V2 `0,1,2,2,2` 不推荐。

Base mapping 全局 collapse 后最高状态为 pv/click/cart/collect/buy：
`22,817/315,883/31,076/7,895/4,579`，level 占比 `5.969%/82.638%/10.195%/1.198%`。
每 session collapse 将 1,643,212 条事件降至 632,971（-61.48%），每用户 median 从 91 降至 37；全局 collapse 降至 382,250（-76.74%），median 为 25。
Base test buy 为 `259/58,495=0.443%`，覆盖 230 用户。
V2 把 cart/collect 也算 max level 后，max-level test 占比变为 4.599%，但 target 仍只评 buy。

若按 V2 真正 collapse并以最后事件打破 tie，全局最高 buy 从 4,579 降至 3,689；test collapsed target 用户从 230 降至 214。
这会实质改变 target universe，不能与 base mapping 混用。

#### 候选 mapping 汇总

| 数据 | Mapping | 全局 collapse 后 level 占比 | 语义/统计判断 |
| --- | --- | --- | --- |
| Tmall | base/paper `0,1,1,2` | 91.920% / 6.883% / 1.198% | 推荐，高置信 |
| Tmall | V2 `0,1,1,1` | 91.920% / 8.080% | 不推荐；丢失支付终态 |
| Tmall | collect<cart `0,1,2,3` | 91.920% / 3.715% / 3.168% / 1.198% | 不推荐；collect→cart 前驱覆盖仅 4.26% |
| Tmall | cart<collect `0,2,1,3` | 91.920% / 2.836% / 4.047% / 1.198% | 不推荐；cart→collect 前驱覆盖仅 6.27% |
| JData | base/paper `0,1,2,2,3` | 5.969% / 82.638% / 10.195% / 1.198% | 推荐，中高置信 |
| JData | V2 `0,1,2,2,2` | 5.969% / 82.638% / 11.393% | 不推荐；丢失购买终态 |
| JData | pv/click tied `0,0,1,1,2` | 88.607% / 10.195% / 1.198% | 合理备选，中等置信 |
| JData | collect<cart `0,1,3,2,4` | 5.969% / 82.638% / 1.810% / 8.385% / 1.198% | 不推荐为严格链 |
| JData | cart<collect `0,1,2,3,4` | 5.969% / 82.638% / 7.754% / 2.442% / 1.198% | 不推荐为严格链 |

如果这些 alternative 真正接入 level-aware augmentation，Tmall collect<cart 会令 cart drop 从 `ratio/2` 变成 `ratio/3`；反向分层则改变 collect。
JData pv/click 同层会令 click drop 从 `ratio/2` 增至 `ratio`，collect/cart 分层会令其中一个从 `ratio/3` 变成 `ratio/4`。
alipay/buy 始终由 target identity 保护，因此只改变它们的 numeric level 不改变当前 factor-4 drop。

按 loader 的“倒数第二 session 为 validation、最后 session 为 test”规则，base/paper mapping 的 target 稀疏度如下。
Train input 指 decoder 训练 label 之前的 prefix；validation/test 列按对应 held-out session 的 event pool 统计。

| 数据 | Train input target events | Train target labels | Validation target events | Test target events |
| --- | ---: | ---: | ---: | ---: |
| Tmall-24 | 11,706 / 1,782,306（0.657%） | 3,340 / 217,374（1.537%） | 8,835 / 943,797（0.936%） | 7,906 / 874,645（0.904%） |
| JData | 4,049 / 1,509,483（0.268%） | 116 / 10,010（1.159%） | 299 / 65,224（0.458%） | 259 / 58,495（0.443%） |

若在每个 session 内执行 base mapping collapse，train/valid/test 三个 event pools 的 target 占比分别变为：
Tmall `1.003%/1.233%/1.169%`，JData `0.710%/1.129%/1.045%`。
V2 的 target-identity 占比在未 collapse 数据中不变，但所谓 max-level 占比会分别虚增到 Tmall `6.827%/6.960%/6.395%`、JData `4.335%/4.454%/4.599%`。

### 5.11 结论等级与最小重跑边界

| 问题 | 结论等级 | 是否必须重跑 | 最小范围 |
| --- | --- | --- | --- |
| ShortVideoAD CID < SID | 当前可继续使用 | 否 | 无；用户已确认符合预期 |
| Tmall/JData base vs V2 原始事件版本 | 当前可继续使用，但须修正版本术语 | 否 | 在数据说明中明确 V2 仅改 level JSON/index |
| Tmall hierarchy | base/paper `click < collect/cart < alipay` 可继续使用 | 仅把 JSON 恢复为 base mapping时，当前 FixedSoft 路径理论上无需重训；需做 dataset/cache parity 与一次固定 checkpoint 重评 | GAMER 两数据集各一次 deterministic parity/eval；不动 baselines |
| JData hierarchy | base/paper `pv < click < collect/cart < buy` 可继续使用；pv/click 同层仅作备选 | 同上 | 同上 |
| FixedSoft 实际按 behavior identity 全序，而非 level mapping | 需要方法实现修复后重跑 GAMER | 若论文声称 relation prior 严格按并列层级计算，则必须 | 先修复 identity/level 双索引，至少重跑 Tmall、JData GAMER；baseline 输入/评测未变时无需重跑 baselines |
| 当前 SMB 文件与 loader 未执行 deepest-state collapse | 与当前论文主线不一致 | 若坚持“实际输入已 collapse”，必须重建数据 | Tmall/JData 重建 train/valid/test/index/cache，并重跑 GAMER 及所有使用该序列输入的 baselines |
| 仅将论文描述改为实际 flattened repeated-event 输入 | 仅修正文统计/术语但会改变主线 | 无训练 | 必须由用户决定；本轮不改论文 |

AAAI 截稿最现实方案是先冻结现有结果，不因 V2 名称立即启动训练；优先完成两个零/低成本检查：

1. 用 base mapping 加载同一 symlink 数据，验证 augmentation 后序列、target 样本与当前 checkpoint 输出逐项一致。
2. 明确论文究竟坚持 deepest-state collapse，还是承认当前 public datasets 使用 repeated event stream。

只有第 2 项选择“坚持 collapse”时，才需要 baselines 一起重跑。
若只修正 FixedSoft 的 level-index 使用，最小必要重跑是 Tmall/JData 的 GAMER，不需要因方法内部改动重跑数据不变的 baselines。

## 6. 数据集逐项分析

### 6.1 ShortVideoAD

#### 优势

- GAMER CID 在四个主 target 指标上均明显超过 MBGen CID，提升为 `+17.78%` 至 `+22.06%`。
- Test 用户数与最终 GAMER SID 完全一致，CID/SID 结果可用于诊断 tokenization，而不是 split 变化。
- 当前最终架构 CID 相对旧架构 CID 的四项主指标均为正增益，说明 TH redesign 的收益并不完全依赖语义 ID。
- cvr 的 NDCG@10 `0.0771` 高于 MBGen CID `0.0635`，是 CID 条件下最有论文价值的 target ranking 证据。

#### 弱项

- CID 相对 SID 有系统性下降：HR@10 `-8.97%`、Recall@10 `-12.06%`、NDCG@10 `-14.32%`。
  用户已确认这一差距符合预期，不再视为需要修复的异常。
- click 是 CID 下降最明显的 behavior。
  CID/SID 的 click HR@10 分别为 `0.1325/0.1654`，约下降 `19.9%`；NDCG@10 为 `0.0562/0.0705`，约下降 `20.3%`。
- pooled CID HR@10/NDCG@10 为 `0.1873/0.0652`，明显低于 SID 的 `0.2121/0.0756`。
- 本地缺失实际 ShortVideoAD CID index，当前无法检查 code 长度、映射 checksum 或是否与 MBGen CID 共用 mapping。

#### 解释

ShortVideoAD 的 SID 来自预训练 semantic IDs，而 shuffled CID 只保证 balanced code，不携带 item semantic proximity。
CID 仍超过 MBGen CID，支持 GAMER 架构在无语义 item code 时也有效；CID 显著弱于 SID，则说明 item semantics 与 TH behavioral modeling 是互补收益，而不是互相替代。
该结论只在 split、mapping 和 baseline 训练口径完全对齐后才能作为正式 ID ablation。
本轮不再为缩小 CID/SID 差距安排优化或补跑。

### 6.2 Tmall

#### 优势

- GAMER CID 相对 MBGen CID 的 HR@10/NDCG@10 分别提升 `0.89%/1.00%`。
- NDCG@10 `0.4719` 略高于当前表中 TIGER 的 `0.4687`，说明靠前排序质量仍有小幅优势。
- Target alipay 有 6,329 个 test 用户，远多于 JData buy，结果的单次运行 sample-level 方差风险相对较低。

#### 弱项

- HR@10 `0.5772` 低于 TIGER `0.5810`，不能声称在 Tmall 两个主指标上都超过所有 baselines。
- 相对 strongest baseline 的差异很小：HR@10 `-0.66%`，NDCG@10 `+0.68%`。
  在没有 user-level baseline vectors 和显著性检验时，不能把该差异表述为稳定优势。
- click 占 pooled test instances 的约 85%，因此 pooled 结果主要反映 click，不应代替 alipay target 结论。
- V2 level JSON 把 collect、cart 和 alipay 都放在 level 1，但 FixedSoft 实际又按 token insertion order 把 collect < cart < alipay 强制成全序。
  两种层级来源互相不一致，且 collect/cart 的全序缺少数据支持。
- 当前 SMB 文件含同一 user-item 重复事件，loader 不做 collapse；这与当前论文的 collapsed-input 叙述不一致。
- 当前 item 数为 361,932，而活动论文 dataset table 写 379,450。
  需要确认论文统计是过滤前还是 V2 过滤后口径。

#### 解释

Tmall 是当前最弱的跨域证据。
其结论应是 GAMER 与强生成式/序列 baseline 竞争，并在 NDCG@10 上小幅领先，而不是全面最好。
效果上限可能同时受层级索引实现错位、CID 无语义性和 public baseline provenance 不完整影响。
数据本身支持 `click < {collect, cart} < alipay`，不支持把 alipay 与中层行为并列，也不支持 collect/cart 的严格全序。

### 6.3 JData

#### 优势

- GAMER CID 的 buy HR@10/NDCG@10 为 `0.7000/0.5449`，相对 metric-specific strongest baselines 提升 `7.33%/12.96%`。
- 相对 MBGen CID 的提升更大，为 `13.56%/20.56%`。
- 所有 behavior 都有完整 8 项指标，且 zero collision。
- 与 Tmall 的弱/混合结果相比，JData 提供了更明确的 public-dataset 正向证据。

#### 弱项

- buy test 用户只有 230，collect 只有 219。
  单次运行的 HR 指标粒度约为 `1/230=0.00435`，少量用户即可产生明显相对变化。
- 缺少 baseline user-level vectors，无法做 paired permutation test 或置信区间。
- V2 level JSON 把 cart、collect 和 buy 都放在 level 2，但 FixedSoft 实际按 token insertion order使用 `pv < click < cart < collect < buy`。
  数据支持 buy 单独最深与 collect/cart 并列，不支持 cart<collect 的强制全序。
- pv<click 业务语义合理，但数据方向证据只有中等强度；pv/click 同层应保留为备选而非当前必改项。
- 当前 SMB 文件含 repeated user-item events，loader 不做 collapse。
- 当前新结果低于旧占位值，说明跨 run 或跨架构变化不小；必须保留最终冻结配置的 provenance。

#### 解释

JData 是目前最强的跨域 CID 证据，但也是 target sample 最少、最需要不确定性说明的数据集。
其较大相对增益支持 GAMER 并非只对 ShortVideoAD 有效；不能据此声称跨随机种子稳定，也不能忽略 target test set 较小。
原始 base mapping `pv < click < {cart, collect} < buy` 比 V2 mapping 更符合业务目标与实证覆盖。

## 7. 跨数据集结论

### 7.1 一致性

- GAMER CID 在三个数据集上都超过 MBGen CID 的 NDCG@10。
- 相对 MBGen CID 的 HR@10 也全部为正：ShortVideoAD `+17.78%`、Tmall `+0.89%`、JData `+13.56%`。
- 增益大小与数据集并不一致。
  ShortVideoAD 和 JData 较强，Tmall 接近持平，说明方法收益受 behavior hierarchy、target sparsity、ID semantics 和数据规模共同影响。
- ShortVideoAD CID 仍显著弱于 SID；用户已确认这符合预期，仅作为 item semantics 的描述性证据。

### 7.2 对论文主线的支撑

当前 CID 结果能支撑的核心方向是：

1. GAMER 的 Temporal-Hierarchical backbone 在不依赖 pretrained SID 的 CID 条件下仍能优于同为 CID 的主要生成式 baseline。
2. 收益主要体现在稀疏 target behavior，而非简单依赖 pooled behavior。
3. 跨域收益是正但不均匀的；Tmall/JData 数据支持偏序而非严格全序，但当前 FixedSoft 按 behavior identity insertion order 形成全序，不能把现有结果写成“并列层级已被正确实现”的验证。
4. SID 对 ShortVideoAD 仍有额外明显贡献，适合解释为 item semantics 与 temporal-hierarchical behavior modeling 的互补。

### 7.3 当前不能写死的结论

- GAMER CID 在三个数据集的所有指标上都超过所有 baselines。
- Tmall 上存在明确且显著的 overall improvement。
- public datasets 的 FixedSoft 计算严格使用了论文概念上的 3/4-level partial order。
- Tmall/JData 当前输入确实已经执行 deepest-state collapse。
- 三个结果具有跨 seed 稳定性。
- 当前 best checkpoint 一定由预先冻结的 validation metric 选出。
- GAMER 与 MBGen 使用逐字相同的 shuffled CID mapping。
- CID/SID 差异完全由 item semantics 导致；code length、词表、优化难度也可能参与。

## 8. 异常与风险清单

| 优先级 | 风险 | 影响 | 当前证据 | 处理 |
| --- | --- | --- | --- | --- |
| P0 | 缺 checkpoint/config/trainer-state/run command | 不能证明 best epoch、seed、超参和代码版本 | 本地无 checkpoint/log | 从训练服务器同步最小 provenance bundle |
| P0 | ShortVideoAD CID index 缺失 | 不能复核 mapping 与 baseline 公平性 | 结果存在，index 不存在 | 同步 index 并记录 SHA-256 |
| P0 | 当前 SMB 输入未 collapse | 与论文方法主线直接冲突，且 counterfactual 序列长度差异达 25%--77% | loader 无 collapse；原始 user-item 重复事件 | 用户决定改叙事还是重建数据；坚持 collapse 则所有序列 baselines 一起重跑 |
| P0 | FixedSoft 按 identity insertion order 而非 level mapping | collect/cart 被强制全序，论文并列层级机制与实现不一致 | router/helpers/TH attention 源码 | 先做双索引设计；若论文坚持 level prior，重跑 Tmall/JData GAMER |
| P1 | V2 level 数与论文表不同 | dataset 术语与 target-level 统计冲突 | V2 JSON 为 2/3 levels；base JSON恰为 3/4 | 当前结果可保留；明确 V2 只改 JSON/index并优先采用 base 业务 mapping |
| P0 | Tmall 新 HR@10 低于 TIGER | 旧 cross-domain “全部提升”句子不成立 | `0.5772 < 0.5810` | 论文更新时必须使用 metric-specific 比较 |
| P0 | Baseline 原始结果缺失 | 无法做 paired significance 和 mapping checksum | 只有恢复 CSV/TeX | 找回 baseline JSON、vectors、index |
| P1 | 结果文件可被不同 ckpt 覆盖 | 不能排除 test checkpoint selection | 结果名不含 ckpt | 版本化结果名并写 manifest |
| P1 | JData buy 仅 230 users | 相对提升方差和量化粒度较大 | result total | 报 user-level interval/paired test；不作跨 seed claim |
| P1 | Factor 4 实际 view 数不恒定 | 论文“每用户 5 条”可能过强 | 长度小于 2 的 view 会跳过 | 从 cache/log统计实际 view histogram |
| P1 | CID baseline mapping 未确认 | CID-to-CID 可能不完全公平 | 仅 token tag 文本 | 比较 index checksum |
| P1 | 当前 checkpoint 以 eval loss 而非 target retrieval 选点 | 预训练 loss 与主目标可能错位 | 当前 Trainer 默认 | 仅在 validation 上做 checkpoint-selection audit |
| P2 | Prediction duplicate ratio 约 26%--34% | 候选中含历史 item，可能影响业务解释 | 三份 result JSON | 与 baselines/SID 同口径比较后再决定是否做 filtered eval |
| P2 | Tmall item 数 361,932 vs 论文 379,450 | 数据版本口径不清 | V2 item/index length | 确认过滤前后统计 |

## 9. 效果改进实验优先级

成本不虚构 GPU 小时。
由于本地缺训练日志，以下以“训练 run 等价成本”表示；实际 wall-clock/GPU-hour 需从服务器日志补充。

### 9.1 低成本：配置与评估修复

#### L0. 建立不可覆盖的 run manifest（P0，零训练成本）

每个数据集保存：

```text
dataset key
data/index SHA-256
git commit
full train command
full test command
resolved model config
trainer_state.json
best_model_checkpoint / best_metric
result SHA-256
user-vector directory
```

预期成本：文件同步与一次结构化整理，无训练。
价值：消除当前最大可信度风险，也是后续任何效果实验的前置条件。

#### L1. 精确 checkpoint 重评（P0，3 次 test evaluation，零训练成本）

对每个数据集从记录的 root best checkpoint 重新运行一次固定 `smb_explicit` test，并输出到带 run ID、checkpoint step 和 timestamp 的新文件，不覆盖当前结果。

验证假设：当前 JSON 确实来自 validation-selected root best checkpoint，而不是显式 `ckpt_num`。

- 成功：数值逐项复现，当前结果升级为可直接进入论文的 frozen result。
- 失败：保留两份结果，定位 checkpoint、代码 commit、index 或 CLI 差异；不能挑较高一份。

#### L2. CID mapping 公平性核查（P0，零训练成本）

同步 ShortVideoAD CID index，并比较 GAMER、MBGen 的 dataset key、item universe、shuffle seed、chunk size、mapping SHA-256 和 code length。

验证假设：CID-to-CID baseline 真正共享同一 item code。

- 成功：ShortVideoAD 的 `+17.78%/+21.43%` 等比较可视为 matched CID evidence。
- 失败：必须用同一 index 重新训练/评测至少一方，现有 CID comparison 降级。

#### L3. Validation-only checkpoint-selection audit（P1，若 checkpoint 尚在则只需 validation evaluation）

用保存的最后 1--2 个 checkpoint 和 root best checkpoint，在 validation split 上比较：

- full-sequence validation loss；
- target-behavior HR@10/NDCG@10；
- pooled HR@10/NDCG@10。

在看 test 之前固定一种统一规则，例如：

```text
primary: validation target NDCG@10
tie-break: validation target HR@10
```

验证假设：默认 validation loss 选点可能不是 CID target retrieval 的最佳代理。

- 成功：若三数据集 validation retrieval 都支持同一替代选点规则，可在**预先冻结规则后**各做一次 test。
- 失败：继续使用 loss-selected checkpoint，不增加论文复杂度。

注意：不能在已经观察各 checkpoint test 结果后选择规则。

#### L4. User-level uncertainty（P1，零训练成本）

先对 GAMER 自身 vectors 做 bootstrap confidence interval；正式 paired test 必须找回 strongest baseline 的同一用户 vectors。

验证假设：Tmall 的小幅 NDCG 优势和 JData 的大幅优势在 user-level uncertainty 下结论不同。

- 成功：为论文提供正确的不确定性边界。
- 失败或 baseline vectors 缺失：只报告单次 point estimate，不加显著性符号。

#### L5. Seen-item/duplicate evaluation audit（P2，3 次 evaluation 或离线重算）

当前生成候选允许历史 item，behavior-wise average duplicate ratio 约为：

- ShortVideoAD：p3s 30.82%、click 33.67%、cvr 26.37%。
- Tmall：28.78%--33.05%。
- JData：26.08%--29.93%。

先确认 baseline 完全同口径。
若论文主协议不应过滤历史 item，则保留当前主结果，只在 supplement 报诊断；不要只对 GAMER 改 evaluation。

#### L6. Base/V2 deterministic parity audit（P0，零训练 + 2 次 evaluation）

在不改仓库文件的临时数据目录中，让 Tmall/JData 分别使用原始 base level JSON，固定 RNG 后比较：

- factor-4 augmentation 每个 view 的保留 event indices；
- train/valid/test target uid 与 item；
- tokenized input；
- 当前 checkpoint 的逐样本 logits 和最终 metrics。

验证假设：对当前 FixedSoft decoder 路径，V2 把 terminal behavior 降到 tied level 不改变 augmentation、target 或模型 hierarchy index。

- 成功：现有 V2 结果可保留，并将数据业务 mapping 记录为 base/paper partial order。
- 失败：定位 `actions` 的隐藏消费者或 cache 差异；在未解释前不把结果改挂到 base mapping。

### 9.2 中成本：训练调整

#### M1. CID validation-selected optimization audit（P1，每数据集最多 2--3 个短列表配置；约 6--9 个完整 run）

只搜索少量、可跨数据集统一的配置：

```text
learning rate: 3e-4 vs 5e-4
warmup ratio: 0.04 vs 当前实际值
checkpoint selection: validation target NDCG vs validation loss
```

先在 validation 上确定一套统一 CID recipe，再对每个数据集只测试一次。

验证假设：CID code 的离散、无语义特性使当前 SID-oriented optimization recipe 不是最优。

- 成功：形成统一 CID recipe，增强三个数据集可比性。
- 失败：保留当前冻结配置，论文不承担额外超参叙事。

AAAI 截稿前不建议启动完整 6--9 run 矩阵；可在投稿后补充。

#### M2. Fixed-soft prior vs no-prior 的 public-dataset matched audit（P1，4 个完整 run 或先做 2 个 validation run）

在 Tmall/JData CID 上比较同一 factor 4 recipe 下的：

```text
GAMER
GAMER w/o soft prior
```

验证假设：当前 identity-order fixed prior 的任意 collect/cart 全序可能降低 public-dataset 收益。

- 成功且 GAMER 更强：支持 soft hierarchy control 可迁移。
- no-prior 更强：说明 public hierarchy definition 需要修正，论文主方法仍保持统一 GAMER，但 public result 不应被归因于 fixed prior。

不得按 dataset 事后选择不同方法作为同一个 GAMER 主结果。

#### M3. Factor 2 vs factor 4 的 CID validation 对照（P2，3 个额外完整 run）

只比较 factor 2/4，不做 factor 0/2/4/8 的大 sweep。

验证假设：CID 已增加 token-level optimization 难度，过多随机 dropout views 可能对 public partial-order 数据集收益有限。

- 成功：若 factor 2 在三个 validation set 上统一更优，可冻结新统一 recipe 后测试。
- 失败：保留 factor 4，与 SID 主配置一致，避免论文复杂度。

#### M4. 多随机种子（P2，最终配置额外 2 seeds/dataset，共 6 个完整 run）

用途是稳定性，不是快速提点。
AAAI 当前七页正文与截稿时间下优先级低于 provenance、映射公平性和 checkpoint 选点修复。

### 9.3 高成本：方法或数据改动

#### H1. Behavior identity / level 双索引修复（高风险、中高成本）

推荐语义 mapping：

```text
Tmall: click < {collect, cart} < alipay
JData: pv < click < {collect, cart} < buy
```

当前仅修改 `behavior_level.json` 不会改变 FixedSoft relation prior 或 behavior-aware Q/K/V，因为它们使用 behavior identity insertion order。
真正修复需要显式分离：

```text
behavior_identity_index: 区分 collect 与 cart 的可学习 embedding / expert
behavior_level_index: 为 relation prior、level auxiliary 与 level-aware policy 提供并列层级
```

验证假设：保留行为身份差异、同时让 relation prior 遵循 partial order，优于当前任意全序。

- 成功：可形成更严格的 public temporal-hierarchical benchmark。
- 失败：保留现有 GAMER 数值，但不能把 identity-order prior 解释成严格的 partial-order prior。

若原始 event stream、split 和 evaluation 不变，只需重跑 Tmall/JData GAMER 及直接方法消融，不需要重跑输入无变化的 external baselines。
该项仍不适合在没有 validation-only protocol 的情况下于 AAAI 截稿前仓促执行。

#### H1b. 真正 deepest-state collapse 数据重建（最高风险、最高成本）

若论文坚持“每个 repeated user-item chain 只保留 deepest state”，必须先明确 collapse scope（全历史、session 内或时间窗内）、tie、selected timestamp、跨 session 重访与 split-before/after-collapse 规则。
随后重建 Tmall/JData inter/behavior/session/time、CID index 与 caches。

该改动会改变所有方法看到的训练与测试数据，因此 GAMER 和所有序列 baselines 必须一起重跑。
本轮统计显示 event 数会下降约 25%--77%，不能把它当作只改 metadata 的小修。

#### H2. CID-aware token objective（已撤销当前优先级）

可能方向包括 code-position loss reweighting、prefix-balanced decoding 或轻量 item-level contrastive regularization。
任何改动都必须保持同一 CID mapping、候选空间和 decoding trie，并补 matched ablation。

用户已确认 ShortVideoAD CID 弱于 SID 符合预期，因此不再以缩小 CID/SID 差距为目标验证这一方向。
该项会扩展方法主线并占用七页正文，当前不安排实验。

#### H3. 语义增强或复杂 policy（高成本）

现有证据显示 random-ratio augmentation 强于当前 semantic policy。
CID 提效不应重新打开 Hybrid/Gated/动态 policy 主线，除非固定 view budget、统一 backbone 和 validation-only selection 下稳定超过 random baseline。

## 10. 下一批最值得补跑的实验

### 截稿前最小批次

| 顺序 | 实验 | 假设 | 成本 | 成功后 | 失败后 |
| --- | --- | --- | --- | --- | --- |
| 1 | L6 base/V2 parity audit | V2 level JSON 对当前 FixedSoft run 为计算等价 | 零训练 + Tmall/JData 各一次 test | 保留现有结果并采用正确业务 mapping | 查隐藏 `actions` 依赖，必要时只重跑 GAMER |
| 2 | L0 provenance bundle | 当前结果来自声明的 final run | 零训练 | 结果可冻结 | 阻止论文直接采用，先修来源 |
| 3 | L2 mapping checksum | GAMER/MBGen CID 完全同 mapping | 零训练 | CID baseline comparison 可信 | 同 mapping 重跑一方 |
| 4 | L1 exact-best checkpoint 重评 | 当前 JSON 可复现 | 3 次 test | 锁定三数据集最终值 | 找差异，不挑高值 |
| 5 | L4 baseline vectors / uncertainty | Tmall/JData point estimate 边界可量化 | 零训练 | 可加严谨 uncertainty | 保守表述 single-run |

### 截稿后第一批

| 顺序 | 实验 | 假设 | 成本 | 对论文的影响 |
| --- | --- | --- | --- | --- |
| 1 | H1 双索引 prototype + validation | partial-order level prior 应与 behavior identity 分离 | 2 个数据集各 1 个 validation run；成功后再 full run | 直接核验论文层级机制 |
| 2 | L3 validation checkpoint audit | eval loss 与 target retrieval 选点错位 | validation evaluations；可能 3 次最终 test | 若统一有效，可提升 CID 且强化 selection protocol |
| 3 | M2 no-prior vs GAMER on Tmall/JData | identity-order prior 是否有净收益 | 先 2 validation run，必要时扩至 4 full runs | 澄清跨域收益来自 base TH 还是 fixed prior |
| 4 | M1 小型统一优化审计 | CID 需要不同优化 recipe | 6--9 full runs | 建立统一 CID recipe，不做 dataset-specific cherry-pick |
| 5 | M4 multi-seed | 当前结论跨 seed 稳定 | 6 full runs | supplement 稳定性，非当前主提点手段 |

## 11. 论文可用结论

在完成 P0 provenance 与 mapping 核查后，以下结论可考虑用于论文或 supplement：

- GAMER 在 CID 条件下对三个数据集的 MBGen CID 都取得正向 HR@10/NDCG@10 差异，但幅度高度不均匀。
- ShortVideoAD CID 相对 MBGen CID 的主要 target 指标提升约为 18%--22%，说明 GAMER 的优势不依赖 semantic IDs 才存在。
- ShortVideoAD 上 SID 仍明显优于 CID，支持 item semantics 与 temporal-hierarchical behavior modeling 互补。
- JData 提供明确跨域正向证据，但 buy test users 只有 230，需要谨慎说明单次运行和不确定性。
- Tmall 只支持“具有竞争力并在 NDCG@10 小幅领先”：当前 HR@10 低于 TIGER，不能写成所有指标全面最好。
- 业务语义与全量统计支持 Tmall `click < {collect, cart} < alipay`；支持 JData `pv < click < {collect, cart} < buy`，但 JData 的 pv/click 同层仍是合理备选。
- Public-dataset 的现有结果可以说明跨域适用性，但在 FixedSoft identity/level 索引错位和输入未 collapse 两项问题解决前，不应声称已经严格验证 partial-order deepest-state 建模。

七页正文约束下，主表只需要保留 target HR@10/NDCG@10 和最强 baselines。
完整 behavior-wise、Recall、CID/SID、provenance 和 uncertainty 应进入 technical supplement。

## 12. 待用户确认事项

1. **训练服务器 provenance：**
   是否可以同步三个最终 CID run 的 checkpoint config、`trainer_state.json`、完整 train/test command、Git commit 和日志？
2. **ShortVideoAD CID index：**
   训练实际使用的 `.index.cid.shuffle.chunk64.json` 在哪里，SHA-256 是什么？
3. **Baseline CID mapping：**
   MBGen CID 是否与 GAMER CID 逐字复用同一 index 文件，而不只是同为 chunk size 64？
4. **Checkpoint selection：**
   三个结果是否都来自 `ckpt_num=best` 的 root model？
   是否曾测试其他 checkpoint 并覆盖同名结果？
5. **Deepest-state 输入定义：**
   当前 Tmall/JData SMB 文件与 loader 都保留 repeated user-item events。
   论文是否坚持真实执行 collapse？
   若坚持，需要用户确认 collapse 是跨全历史、session 内还是固定时间窗，以及 split 在 collapse 前还是后。
6. **Identity/level 双索引：**
   是否接受把 collect/cart 保留为不同 behavior identities、但共享 relation level，并在 AAAI 截稿后先做两数据集 validation prototype？
7. **V2 设计依据：**
   `cb987b2/ea90279` 把 alipay/buy 与中层行为 tied 的原始动机是什么？
   若无额外证据，本文建议业务 mapping 回到 base JSON 的 3/4 levels。
8. **Tmall item count：**
   论文的 379,450 与 V2 运行的 361,932 分别对应过滤前还是过滤后数据？
9. **Baseline raw outputs：**
   Tmall/JData 的 TIGER、MB-STR、MBGen 原始 result JSON 和 user-level vectors 是否仍在服务器？
10. **实际训练超参：**
   global batch、GPU 数、per-device batch、gradient accumulation、warmup、max history、epoch、precision 和 seed 是否三个数据集完全一致？
11. **实际训练倍率：**
   是否有 Dataset log 可给出 factor 4 下每个数据集的有效 view histogram 和总训练样本数？
12. **投稿前范围：**
   是否将当前工作严格限制为结果 provenance 与论文数值替换，不再启动新训练？

## 13. 维护规则

- 新结果只能新增 versioned 记录，不覆盖本表中已有来源。
- 每个数值必须保留 result path、run ID、checkpoint、index checksum 和 selection rule。
- Validation 与 test 必须分表；`smb_explicit_valid` 绝不进入 test 主表。
- 多 checkpoint、多 seed 和多配置结果必须先写预先冻结的 selection rule，再读取 test。
- 论文方法统一称 GAMER；内部配置名只放 provenance。
- 如果新证据与本文冲突，以当前源码、实际 run artifact 和最新 versioned result 为准，并在本文记录变更原因。

## 14. 2026-07-27 V3 双索引实现

用户已确认 Tmall/JData 可以重跑，并接受保留 base、V2 的同时新增 V3。
本轮实现采用“数据版本显式启用、旧版本完全兼容”的方式，不全局改变旧 checkpoint 的计算语义。

### 14.1 V3 数据定义

| 数据集 | behavior identity | hierarchy level | target | 复用内容 |
| --- | --- | --- | --- | --- |
| `Tmall-24-0.25-V3` | click, collect, cart, alipay 四个独立 identity | `click:0, collect:1, cart:1, alipay:2` | alipay | 与 V2 相同的 SMB event stream、session/time/item 文件；CID index 独立生成 |
| `JData-V3` | pv, click, cart, collect, buy 五个独立 identity | `pv:0, click:1, cart:2, collect:2, buy:3` | buy | 与 V2 相同的 SMB event stream、session/time/item 文件；CID index 独立生成 |

V3 通过 `<dataset>.behavior_schema.json` 中的
`"separate_behavior_identity_and_level": true` 显式启用双索引。
base 与 V2 没有该文件，因而仍使用 legacy identity-order relation semantics，可继续加载和复现既有 checkpoint。

本轮没有执行 deepest-state collapse。
这是有意的控制变量设计：V3 与 V2 的输入 event stream 完全一致，并使用相同的 CID 生成参数与固定随机种子，仅比较业务层级修正与双索引 TH 实现。
若后续重建 collapsed event stream，应建立独立数据版本并重跑所有看到输入变化的方法。

### 14.2 模型计算分工

```text
behavior_identity_index
  -> behavior-aware Q/K/V embeddings
  -> behavior-aware MoE/FFN routing

behavior_level_index
  -> fixed-soft relation bias
  -> multi-view up/down/same relation
  -> behavior-level auxiliary target
```

对于 Tmall，collect 与 cart 保留不同 identity，但共享 level 1。
对于 JData，cart 与 collect 保留不同 identity，但共享 level 2。
特殊 token 继续使用索引 0，实际 behavior identity/level 均在模型内部加 1。

relation table 的维度从 legacy 的 `num_behavior + 1` 改为 V3 下的
`num_behavior_levels + 1`；behavior-aware Q/K/V embedding 的维度仍为
`num_behavior + 1`。
旧 checkpoint 未保存 `num_behavior_levels` 时自动回退到 `num_behavior`，参数形状与旧实现一致。

### 14.3 数据一致性核查

- Tmall V3：217,374 users，3,818,122 events；四类行为均出现在 mapping 中，无 unknown/missing behavior。
- JData V3：10,010 users，1,643,212 events；五类行为均出现在 mapping 中，无 unknown/missing behavior。
- V3 的交互、behavior、session、time 与 item 文件均为指向既有数据的有效符号链接。
- `*.index.cid.shuffle.chunk64.json` 属于可再生成产物，继续由 `.gitignore` 排除，不建立 V2/V3 符号链接。
- `Tokenize` 在 CID 生成前调用 `set_seed(42)`；相同 item universe、shuffle 与 chunk size 应产生相同 mapping，生成后仍需以 SHA-256 复核。

### 14.4 推荐重跑顺序

先为两个 V3 数据集独立生成 CID index：

```bash
dataset=Tmall-24-0.25-V3 cid=1 shuffle=1 chunk_size=64 \
bash scripts/tokenize.sh

dataset=JData-V3 cid=1 shuffle=1 chunk_size=64 \
bash scripts/tokenize.sh
```

生成后应先比较 V3/V2 CID 文件的 SHA-256；只有 mapping 一致时，V2 external baselines 才能直接作为严格控制变量。

随后只训练两个 V3 GAMER run，不重跑 external baselines：

```bash
dataset=Tmall-24-0.25-V3 cid=1 shuffle=1 chunk_size=64 \
tasks=smb_explicit_decoder_4 backbone=Qwen3TemporalHierarchicalFixedSoft \
bash scripts/train_SMB_decoder.sh <沿用已冻结的Tmall训练参数>

dataset=JData-V3 cid=1 shuffle=1 chunk_size=64 \
tasks=smb_explicit_decoder_4 backbone=Qwen3TemporalHierarchicalFixedSoft \
bash scripts/train_SMB_decoder.sh <沿用已冻结的JData训练参数>
```

checkpoint 仍只依据 validation 选择。
冻结 checkpoint 后再运行：

```bash
dataset=Tmall-24-0.25-V3 cid=1 shuffle=1 chunk_size=64 \
tasks=smb_explicit_decoder_4 test_task=smb_explicit \
backbone=Qwen3TemporalHierarchicalFixedSoft \
bash scripts/test_SMB_decoder.sh <沿用已冻结的Tmall测试参数>

dataset=JData-V3 cid=1 shuffle=1 chunk_size=64 \
tasks=smb_explicit_decoder_4 test_task=smb_explicit \
backbone=Qwen3TemporalHierarchicalFixedSoft \
bash scripts/test_SMB_decoder.sh <沿用已冻结的JData测试参数>
```

建议结果对比至少包含：

1. V2 GAMER 的现有冻结结果；
2. V3 GAMER 的新结果；
3. 同一 CID mapping 下的 external baselines；
4. behavior-wise target 与 merged HR@10/NDCG@10；
5. V3 checkpoint 中保存的 `behavior_maps`、`behavior_level_maps`、
   `num_behavior` 和 `num_behavior_levels`。

### 14.5 验证状态

- dataset schema/loader 与 augmentation regression：7 tests passed；
- identity/level config、router、relation table、auxiliary head 与 TH forward：
  11 tests passed；
- Python syntax、Git whitespace、V3 source symlink 与 behavior coverage：
  passed。

本机 `gamer` 环境直接导入模型时会触发 deepspeed 对 MPS
`current_allocated_memory` 的崩溃。
测试时只在 runner 中 stub deepspeed 以绕过该环境初始化问题；模型、Torch 和
Transformers 代码路径仍按真实实现执行。
