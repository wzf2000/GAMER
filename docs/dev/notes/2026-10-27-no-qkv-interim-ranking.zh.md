# no-QKV 固定 checkpoint 的中间 ranking 评测

英文对应：`2026-10-27-no-qkv-interim-ranking.md`。

## 范围与 checkpoint 选择

2026-10-27 18:52:51（Asia/Shanghai），no-QKV 最新已保存训练状态为 epoch 44、step 5148。按**全行为 validation loss** 选择的最佳 checkpoint 是 `checkpoint-4914`，epoch 42，loss 为 **2.034342050552368**。训练仍在继续；本次是所选权重的中间评测，不是 no-QKV 最终结果。test ranking 不参与 checkpoint 选择。

来源：`results/rebuttal/2026-10-26-p0/no_qkv/seed_42/checkpoint/checkpoint-4914/`。

独立推理快照复制权重、config、tokenizer、trainer state、训练参数及训练启动 manifest，不复制 optimizer／scheduler／RNG 状态。该快照避免 checkpoint 轮换影响推理，未修改训练文件。复制后权重 SHA256 与源文件一致：

`bfff940b2ea499a8cc1d99843296dea790b7cefa1fc2a0a2406bed2007588f3a`。

## 评测设置

- 数据：ShortVideoAD；original SID `.index.json`，SHA256 为 `0e71e2064bf2c0f95e639e511f4c81a00ef839ec12cd1e2e7ad43e485ab6e62d`。
- 模型：29,922,816 个可训练参数；`th_use_behavior_qkv=false`；冻结 soft prior 的 lambda 为 0.05；保留 attention gate、位置专属 FFN 和 FFN 行为注入。
- GPU：启动前空闲的 1、5 号卡，2×A100-SXM4-80GB；FP32、SDPA、每卡 ranking batch 8、beam 20、seed 42、历史长度上限 100。
- validation ranking：`test_SMB_decoder --test_task smb_explicit_valid --behaviors cvr`，实际数据 mode 为 `valid_test`。
- test ranking：`test_SMB_decoder --test_task smb_explicit --behaviors cvr`，实际数据 mode 为 `test`。
- 候选：索引含 168,530 个 item ID、151,925 个唯一 SID 表示，使用 cvr 行为前缀。测量的是 cvr 条件下的 next-item ranking，不是转化概率预测。
- 使用 `--export_user_targets` 导出逐用户指标、UID sidecar 及原始 target ID。
- 评测代码版本：`e7e94f3e75dff8e7bfbd4c7264b31b9dc87aff3f`；实际命令和选模来源保存在下述 `evaluation.json`。复制的训练启动 manifest 单独记录训练 checkout 和当时工作区是否有修改。

## 结果

| Split | 用户数 | HR@1 | HR@5 | HR@10 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 9,408 | 0.043579932 | 0.137223639 | 0.205888605 | 0.158593993 | 0.075447820 | 0.093055472 |
| Test | 8,784 | 0.042008197 | 0.132058288 | 0.198315118 | 0.151055865 | 0.070861593 | 0.088230915 |

validation 包含 17,788 个 held-out cvr target 事件，test 包含 16,812 个。两阶段进程均正常结束。已分别独立核验 JSON 校验和、真实 split、模型开关、权重／数据／索引指纹保持一致、逐用户指标均值、UID 唯一性及 target 对齐。按现有 collision 检查，两组 target 均未命中有歧义的 SID 碰撞表示。

validation 于 2026-10-27 18:53:03–19:06:05 运行，耗时 781.385 秒；test 于 19:06:05–19:18:11 运行，耗时 726.654 秒，均为 Asia/Shanghai。墙钟时间包含加载、候选构建、推理及输出导出，不是隔离的模型推理基准。评测结束后 GPU 1、5 已释放。

约 19:18 的评测后检查发现，训练已保存**新的**最佳 checkpoint：epoch 45、step 5265，全行为 validation loss 为 **2.0321006774902344**。上表仍严格对应启动评测时选定的 epoch 42／step 4914 快照，未声称已获得 step 5265 的 ranking 结果。

## 本地产物

根目录：`results/rebuttal/interim-ranking/no_qkv-seed42-step4914-20261027-185251/`。

- `evaluation.json`：源 checkpoint、选择时训练状态、快照 hash、命令及各阶段时间／退出码。
- `verified-summary.json`：独立核验后的指标、用户／target 事件数量、结果校验和及墙钟耗时。
- `checkpoint-snapshot/`：独立推理权重／config／tokenizer 和训练来源记录。
- `validation-ranking.json`、`test-ranking.json`：各 split 结果，分别配套 `.run.json` 来源记录。
- `validation-ranking/`、`test-ranking/`：逐用户指标、UID sidecar 和原始 target 导出。
- `validation.log`、`test.log`：完整进程输出。

生成产物保留本地，不纳入 Git。本次未修改训练／模型源码。判断 Full/no-QKV 推荐效果差异仍需受控 Full ranking 结果，不能用历史投稿分数代替。
