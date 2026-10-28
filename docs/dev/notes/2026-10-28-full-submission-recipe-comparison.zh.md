# 本次 Full 与找回的投稿训练配方对照

英文对应：`2026-10-28-full-submission-recipe-comparison.md`。本次只核查元数据与代码，未启动训练或 ranking。

## 证据与主要差异

用户于 2026-10-28 找回原始命令：使用 `train_SMB_decoder`、backbone `Qwen3TemporalHierarchical`、基配置 `Qwen3TemporalHierarchicalFixedSoft`，数据集 ShortVideoAD，任务 `smb_explicit_decoder_4`，original `.index.json`，历史 100，每卡 batch 128，lr=5e-4，最多 200 epoch，梯度累积 8，warmup 0.04，patience 20，temperature 0.7。完整原始命令保存在 `results/rebuttal/2026-10-28-full-submission-comparison/` 的运行核查记录中。

原命令没有包含分布式 launcher。**结合用户此前提供的 FixedSoft／factor4 使用 4×A800-SXM4-80GB 的资源记录**，原 world size 为 4。旧硬件／卡数是作者提供的证据，不是从旧 checkpoint 恢复的信息。本次 Full 的 manifest 可独立确认 world size 为 2、硬件为 A100-SXM4-80GB。

| 设置 | 找回的投稿配方 | 本次 Full |
| --- | --- | --- |
| 训练硬件／world size | 4×A800 80GB，作者资源记录 | 2×A100 80GB，manifest |
| 每卡训练 batch | 128 | 128 |
| 梯度累积 | 8 | 8 |
| **名义有效训练 batch** | **128×4×8 = 4096** | **128×2×8 = 2048** |
| 每卡 validation batch | 128，由旧参数解析 | 128，HF 有效参数 |
| 全局 validation batch | 四卡为 512 | 双卡为 256 |
| 学习率 | 5e-4 | 5e-4 |
| Warmup ratio | 0.04 | 0.04 |
| Epoch 上限／patience | 200／20 | 200／20 |
| Temperature | 0.7 | 0.7 |
| 历史／任务／索引名 | 100／`smb_explicit_decoder_4`／`.index.json` | 相同 |

所以主要可确认的配方不一致是**有效 optimizer batch 减半**，以及硬件／world size 不同。不能说显式学习率、warmup、梯度累积、patience 或 temperature 改变了。本次 manifest 中旧接口 `per_device_batch_size=256` 已被显式 HF 参数 `per_device_train_batch_size=128`、`per_device_eval_batch_size=128` 覆盖；实际用的是 128。原命令的旧接口 128 也解析为 train／eval 各 128，参数写法不同不等于 microbatch 不同。

当前训练样本为 238,270 条，本次实际每 epoch **117 次 optimizer 更新**（epoch 74、step 8658）。同数据／loader 下，四卡约为 **59 次**。当前 HF Trainer 规划 scheduler 总步数时按 microbatch／累积向下取整：新 `trainer_state.json` 记录计划 **23,200** 步、warmup **928** 步；若旧数据与实现也相同，对应四卡推算为 **11,600／464**。旧步数仅为条件推算，不是找回的训练事实。Batch／卡数变化时，相同 epoch 上限和 warmup 比率不能保证相同优化轨迹。

## 模型配置与未显式参数

FixedSoft 源配置与历史提交 `0cad454`／`c682e1d`（2026 年 6 月）一致。使用当前 ShortVideoAD 索引／tokenizer 解析该模板后，与本次 Full 准备配置完全相同，参数量均为 **29,935,104**。保留 8 层 decoder、TH 层 2–5、hidden size 256、Q/K/V 行为条件化、output gate、position-routed FFN 与 behavior injection，冻结 soft prior λ=0.05、额外倍率 1，dropout／attention dropout 0.2。这证明预期模板对齐，原机器的最终 checkpoint config 仍未取得。

历史与当前 parser 的省略默认值一致：seed 42、AdamW（`adamw_torch`）、cosine scheduler、weight decay 0.01、FP32。SMB 任务提供模型长度 **1024**，覆盖共享 dataclass 的通用默认 512。这些属于依据默认值推定，不是独立读取旧 `training_args.bin` 的结论。

以下属于运行时解析，不能当作换模型的证据：词表 14 是模板占位，扩展后为 23,675；behavior map／数量、位置和专家数由数据／tokenizer 填入；`model_type` 从 `qwen3` 序列化为共用 `qwen3_moe` 配置类，实际显式 backbone 仍是 TH；旧代码和当前代码训练时都关闭 `use_cache`。公共 parser 中的 `sequence_augmentation=time_decay` 不会把 `smb_explicit_decoder_4` 切成 policy augmentation，该任务实际走原 random augmentation 数据集。

## 尚未核实及对齐方式

- 尚未找回旧 checkpoint、trainer state、数据／索引／缓存 hash 或测试命令。旧停止／最佳 epoch、依赖／backend／精度的最终状态，以及输入文件逐字节一致性仍未核实。同任务和索引名称不能证明文件内容相同。
- 原命令对应四卡依赖此前作者资源记录；若该记录对应另一个运行，应按真实 launcher／world size 重算 batch。
- 旧 FixedSoft 结果文件分别包含名义 test cvr HR@10／NDCG@10 **0.198087432／0.089980734**，以及 validation **0.204825680／0.091618456**。旧文件缺少 run manifest，论文表格与哪份文件对应仍需单独核查，不能用旧 validation 与新 test 比较。
- 本轮 Full／plain／no-QKV 三组均为 batch **2048**，内部配方仍受控。与旧 batch-4096 Full 比较时，应标为不同配方。
- 双卡匹配旧名义训练 batch 需要将**梯度累积改为 16**，每卡 batch 保持 128；或使用四卡及原来的 128／8。新的组件对照应统一采用所选配方。仅对齐 batch 仍不能保证跨卡数／硬件逐步数值复现。本次核查没有启动这些重训。

主要证据为本次 `results/rebuttal/2026-10-26-p0/full/seed_42/checkpoint/run_manifest.start.json`、`trainer_state.json`、`config/s2s-models/Qwen3TemporalHierarchicalFixedSoft/config.json`、历史 `SeqRec/utils/args.py` 与 SMB 任务默认值，以及用户提供的命令／资源记录。本次 Full 最佳为 epoch 54／step 6318，epoch 74 早停。
