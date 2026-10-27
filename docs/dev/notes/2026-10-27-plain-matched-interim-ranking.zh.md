# 等容量普通 decoder 的中间 test ranking 评测

英文对应：`2026-10-27-plain-matched-interim-ranking.md`。

## checkpoint 与范围

2026-10-27 19:20:17（Asia/Shanghai），等容量普通 decoder 最新已保存训练状态及按**全行为 validation loss** 选择的最佳 checkpoint 均为 epoch 61、step 7137，loss 为 **2.01444149017334**。从 `results/rebuttal/2026-10-26-p0/plain_matched/seed_42/checkpoint/checkpoint-7137/` 复制独立推理快照，并确认权重 SHA256 与源文件一致：

`5faaade154a43edf21070fbcf51f51538a19669a7bb3e47bc643475da67e9287`。

训练仍在继续。约 19:33 的评测后检查中，最佳已保存 checkpoint 仍是这一份。本次是按 validation loss 选模的中间 test ranking 结果，未按 test 指标选模。未修改训练文件或模型源码。

## 已核验配置

- 普通 decoder：29,935,616 个可训练参数；8 层、hidden size 256、共享 FFN 宽度 3501；TH、稀疏 FFN、FFN 注入层列表均为空，行为注入关闭。序列化张量中没有 TH 专用行为 embedding、relation table、attention 输出 gate 或 FFN 专家组；保留标准 FFN 的 gate/up/down 投影。
- ShortVideoAD、original SID `.index.json`、历史上限 100、seed 42。索引 SHA256 为 `0e71e2064bf2c0f95e639e511f4c81a00ef839ec12cd1e2e7ad43e485ab6e62d`。
- 使用启动前空闲的 GPU 1、5，2×A100-SXM4-80GB；FP32、SDPA、每卡 ranking batch 8、beam 20。本模型与此前 no-QKV 快照的生成缓存均开启。
- `test_SMB_decoder --test_task smb_explicit --behaviors cvr --export_user_targets`，实际数据 mode 为 `test`。候选采用与 no-QKV 评测相同的 cvr 前缀 SID 索引。
- 评测 checkout：`7c0c6fb`；与前一轮 no-QKV 评测的 `e7e94f3` 相比，仅增加了 no-QKV 实验文档。

## Test 结果

两次评测使用**完全相同的 8,784 个用户、16,812 个 held-out cvr target 事件**，原始 target ID 和 SID 表示也逐项一致。以下是 cvr 条件下的 next-item ranking 指标。

| 快照 | HR@1 | HR@5 | HR@10 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 等容量普通 decoder，epoch 61 / step 7137 | 0.040983607 | 0.135587432 | 0.197973588 | 0.149450960 | 0.072071882 | 0.087893460 |
| 此前 no-QKV，epoch 42 / step 4914 | 0.042008197 | 0.132058288 | 0.198315118 | 0.151055865 | 0.070861593 | 0.088230915 |

普通 decoder 减去 no-QKV 的差值：HR@10 为 **-0.000341530**，NDCG@10 为 **-0.000337454**。两者为不同训练进度的中间快照；该描述性比较不是最终 Full/plain 消融结论或多 seed 结论。此前 no-QKV 评测详见 `2026-10-27-no-qkv-interim-ranking.zh.md`。

进程正常结束。已独立核验结果／模型／索引／数据校验和、实际 split、逐用户指标均值、UID 唯一性、target 对齐，以及与 no-QKV 一致的评测设置。按现有 collision 检查，target 中有歧义的 SID 碰撞样本数为零。

墙钟耗时 **776.889 秒**，2026-10-27 19:20:26–19:33:23，Asia/Shanghai，包含加载、候选构建、推理及输出导出，不是隔离的推理效率基准。评测后 GPU 1、5 已释放。

## 本地产物

根目录：`results/rebuttal/interim-ranking/plain_matched-seed42-step7137-20261027-192017/`。

- `evaluation.json`：checkpoint 选择、快照 hash、实际命令、时间戳和退出码。
- `verified-summary.json`：完整指标、已核验数量／校验和，以及相对此前 no-QKV 快照的描述性差值。
- `checkpoint-snapshot/`：独立推理权重／config／tokenizer 及复制的训练来源记录，不包含 optimizer／scheduler／RNG 状态。
- `test-ranking.json`、`test-ranking.run.json`、`test.log`：结果、实际评测来源记录及完整进程输出。
- `test-ranking/`：逐用户指标、UID sidecar 和原始 target 导出。

生成的运行产物保留本地，不纳入 Git。
