# ShortVideoAD 的 P0 最终 test ranking

英文对应：`2026-10-28-p0-final-test.md`。本次按用户要求评测已完成的训练，并复用相同最佳 checkpoint 的既有 test。未启动训练，也未修改模型、数据集或打分实现。

## 选模与可比性

三组均使用 ShortVideoAD、original SID `.index.json`、seed 42、`smb_explicit_decoder_4`，记录的有效训练 batch 为 2048。训练 manifest 标明训练 checkout 及其工作区修改标志。按**全行为 validation loss** 选模，不按 test ranking 选模。

| 变体 | 参数量 | 最佳 epoch／step | 最佳 validation loss | 最终 epoch |
| --- | ---: | --- | ---: | ---: |
| Full／TH-FixedSoft | 29,935,104 | 54／6318 | 2.0361568927764893 | 74 |
| 等容量普通 decoder（`plain_matched`） | 29,935,616 | 61／7137 | 2.01444149017334 | 81 |
| no-QKV（`no_qkv`） | 29,922,816 | 53／6201 | 2.029874086380005 | 73 |

最终根目录的 `model.safetensors` 与所选最佳 checkpoint 权重逐字节一致。SHA256 为：

- Full：`90da57d7e077dbbce5d3654ab2bffc06504b4fa3a4ead096a84cb3eef0433a77`。
- 普通 decoder：`5faaade154a43edf21070fbcf51f51538a19669a7bb3e47bc643475da67e9287`。
- no-QKV：`f6071ceeedbf699e814b7c8520f8bcf0633c657b6d4ddf610f62ace8221b054f`。

已对照共同 plan 核验训练数据／validation 缓存指纹和当前数据／索引快照。将 data_path 的相对／绝对写法归一化后，三组保存的 tokenizer／generation 文件及主要有效训练参数也一致；核查保存在 `verified-training-comparability.json`。SID 索引 SHA256 为 `0e71e2064bf2c0f95e639e511f4c81a00ef839ec12cd1e2e7ad43e485ab6e62d`。

## 评测与复用

任务为 `test_SMB_decoder --test_task smb_explicit`，实际 mode 为 **test**；历史上限 100、beam 20、FP32、SDPA，保留现有行为候选前缀构造。每组新评测使用两张 A100-SXM4-80GB、每卡 batch 64。`--export_user_targets` 保留 UID 对齐的指标和原始 held-out target ID。Test 用户数分别为 **cvr 8,784**、**click 11,444**、**p3s 43,618**。Merged 按 63,846 个行为–用户实例加权，不是去重用户数。Cvr 指标表示行为条件下的 next-item ranking。

Full 使用 GPU 2／3，no-QKV 使用 6／7，均先评测 cvr。Full 没有现成 test；no-QKV 的旧中间评测使用 step 4914，不能替代最终 step 6201。

普通 decoder 最终最佳仍为 step 7137，与旧中间评测完全相同。已核验模型权重、完整 config、tokenizer 文件、generation config、训练 manifest、数据／索引及相关打分代码均未变化。因此复用 `results/rebuttal/interim-ranking/plain_matched-seed42-step7137-20261027-192017/` 中的 **test／cvr** 结果；该旧推理使用两张 GPU、每卡 batch **8**。保留原始 cvr 结果行及全部逐用户／UID／target 文件。旧 cvr-only merged 行不能作为全行为 merged 复用。

用户明确要求复用后，已通过 torchrun 父进程 PID 3321407 安全停止刚启动的普通 decoder 重复评测，保留日志和非零取消记录。该任务因复用而被替代，不是训练失败。现仅在 GPU 4／5 补测 click／p3s，运行记录位于 `plain_matched/remaining/`，补测汇总为 `test-remaining-ranking.json`。最终普通 decoder 结果合并已核验的 cvr 行及两项新行为结果，重新计算 merged；来源记录明确区分每个行为的文件和 batch（复用 cvr 为 8，新 click／p3s 为 64），不会声称 batch 完全统一。

## 结果

三组最终 test 结果已全部完成。Full 于 **2026-10-28 15:19:41** 完成，no-QKV 于 **2026-10-28 15:18:59** 完成，普通 decoder 的新 click／p3s 阶段于 **2026-10-28 15:22:23** 完成，均为 Asia/Shanghai。成功的最终评测进程退出码均为 0；较早的普通 cvr 重复任务按用户要求取消并复用，详见上文。已核验三组使用相同的 **8,784 个 UID、16,812 个 held-out target 事件**，指标／UID／target 校验和一致绑定，逐用户指标有限且均值正确。普通 decoder 按上文说明复用旧评测。

| 变体 | HR@1 | HR@5 | HR@10 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 0.039845173 | 0.128756831 | 0.198542805 | 0.152507869 | 0.069265608 | 0.087807118 |
| 等容量普通 decoder | 0.040983607 | 0.135587432 | 0.197973588 | 0.149450960 | 0.072071882 | 0.087893460 |
| no-QKV | 0.041666667 | 0.131944444 | 0.203551913 | 0.154267208 | 0.071500847 | 0.089708056 |

本 seed 下 Full 与普通 decoder 的 HR@10／NDCG@10 接近，两项差值方向不同；no-QKV 的 HR@10、NDCG@10 更高。没有按 test 重新选 checkpoint。本次仅一个 seed，不能据此给出多 seed 波动或显著性结论。`verified-cvr-summary.json` 将这些数值绑定到原始导出和所选权重。完整辅助行为及 merged 结果如下：

| 行为 | Full：HR@10／NDCG@10 | 普通 decoder：HR@10／NDCG@10 | no-QKV：HR@10／NDCG@10 |
| --- | --- | --- | --- |
| click | 0.160608179 / 0.068894657 | 0.162617966 / 0.069915867 | 0.165152045 / 0.070905045 |
| p3s | 0.223783759 / 0.072082603 | 0.227887569 / 0.073614020 | 0.226947590 / 0.074014237 |
| Merged | 0.208987251 / 0.073674579 | 0.212072800 / 0.074915730 | 0.212652320 / 0.075616105 |

Full 的 cvr HR@10 略高于普通 decoder（+0.000569217），cvr NDCG@10 略低（-0.000086342）。no-QKV 的这两项 cvr 指标高于 Full（+0.005009107／+0.001900938），且在本次三组中 merged HR@10／NDCG@10 最高。p3s HR@10 最高的是普通 decoder。上述为单 seed 的描述性结果，不能由此认定组件收益或多 seed 显著性。

`verified-summary.json` 已核验最终阶段成功退出、所选权重／config 未改变、训练／数据／索引来源、相同训练／validation 缓存、实际 test split、各行为完全相同的 UID／target、八项有限的逐用户指标及均值，以及 merged 加权均值。现有 `summarize_rebuttal_results.py` 也已生成完整 cvr 和 merged 汇总（`partial=False`）。完整 `test-metrics.csv` 有 96 行：3 个变体 × 4 行行为／merged × 8 项指标。复用的普通 cvr 数值与旧结果完全一致。没有为选择更有利的 test 分数追加评测或训练。

## 本地产物

运行根目录为 `results/rebuttal/2026-10-28-final-test/`，保留原始 test plan、普通 decoder 剩余行为 plan、进程监督记录／日志及复用决策。最终结果路径为：

- `results/rebuttal/2026-10-26-p0/full/seed_42/test-ranking.json`。
- `results/rebuttal/2026-10-26-p0/plain_matched/seed_42/test-ranking.json`。
- `results/rebuttal/2026-10-26-p0/no_qkv/seed_42/test-ranking.json`。

各结果配套 `.run.json` 来源记录及同名目录下的逐用户指标、UID sidecar 和 target 导出。运行准备／监督／合并／核验辅助脚本作为忽略产物保留，并通过规定的 flake8／compileall 检查。权重、数据、日志和结果不纳入 Git。墙钟时间含加载、候选构建和导出，普通模型的 cvr 也来自旧运行，不能视为推理效率比较。
