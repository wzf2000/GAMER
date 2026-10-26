# P0 seed42 两组实验启动核查

检查时间：2026-10-26 12:00（Asia/Shanghai）。本记录确认启动配置和早期运行状态，不代表已收敛或完成 validation。

## 结论

两组参数、数据和输出目录符合准备的计划，训练持续推进。所有实际 HF training arguments 在排除 output/logging/run-name 路径后相同；未发现 unknown/unused CLI 参数或 OOM、NaN/Inf、NCCL 错误。

| 项目 | FixedSoft full | 等容量 decoder |
| --- | --- | --- |
| GPU / DDP 端口 | 1/5，2314 | 2/6，2315 |
| DDP world size | 2 | 2 |
| 可训练参数 | 29,935,104 | 29,935,616 |
| hidden / 层数 / vocabulary | 256 / 8 / 23675 | 256 / 8 / 23675 |
| FFN intermediate size | 512 | 3501 |
| TH attention 层 | [2,3,4,5] | [] |
| position-expert 层 | [0,1,2,3,4,5,6,7] | [] |
| FFN behavior injection 层 | [0,1,2,3,4,5] | []，且 injection=false |
| dropout / attention dropout | 0.2 / 0.2 | 0.2 / 0.2 |
| 当前进度条 | 49/23200 optimizer steps | 33/23200 optimizer steps |
| 首条记录，step 30 | loss=10.1334，grad_norm=0.46155 | loss=10.1313，grad_norm=0.37762 |
| 首条记录 LR / epoch | 1.5625e-5 / 0.26 | 1.5625e-5 / 0.26 |

参数仅差 512，约 0.00171%。full 使用 frozen soft relation table，lambda=0.05，默认运行时倍率 1，Q/K/V 和 output gate 均保留。等容量配置虽继承部分 prior 字段，但 TH 层为空，实际不实例化这些层级 attention 组件。

## 共同数据与训练设置

- `ShortVideoAD`、`.index.json`、`smb_explicit_decoder_4`、历史窗口 100、model_max_length=1024、seed=42。
- 两组源配置 SHA256 分别匹配其计划配置；数据文件快照与计划一致，两组数据指纹完全相同。
- train cache：`data/ShortVideoAD/ShortVideoAD.SMBExplicitDatasetForDecoder.100.SMB.aug4.train.pkl`，238270 样本；validation cache：`data/ShortVideoAD/ShortVideoAD.SMBExplicitDataset.100.SMB.valid.pkl`，193316 样本。路径、大小、mtime、样本数均相同。
- 实际增强来自 random4 缓存。启动参数中的 `sequence_augmentation=time_decay`、`augmentation_views=1` 是未走 policy 分支的 parser 默认项，不表示当前用了 time-decay 增强。
- 实际每卡 train/eval batch=128，梯度累积=8，有效训练 batch=2048。旧参数 `per_device_batch_size=256` 未覆盖显式 HF batch 参数。
- FP32、SDPA、AdamW、LR=5e-4、warmup=0.04、cosine、weight_decay=0.01、loss temperature=0.7、最多 200 epochs、patience=20。
- 每 epoch validation/save，按全行为 validation loss 选择 best；首轮 validation 尚未执行。当前 checkpoint 目录已有 config/tokenizer/start manifest，尚未出现 epoch checkpoint 属于正常阶段。

## 证据与输出

- Full 日志：`logs/train_SMB_decoder/20261026_115245.log`，首条 loss 在第 305 行。
- Decoder 日志：`logs/train_SMB_decoder/20261026_115328.log`，首条 loss 在第 122 行。
- Full 输出：`results/rebuttal/2026-10-26-p0/full/seed_42/checkpoint/`。
- Decoder 输出：`results/rebuttal/2026-10-26-p0/plain_matched/seed_42/checkpoint/`。
- 各输出目录下的 `run_manifest.start.json` 记录实际模型、HF training arguments、数据和缓存指纹；没有写入旧 checkpoint 路径。
- 核对了四个 worker 的 CUDA_VISIBLE_DEVICES、rank、world size 和 master port，确认两个独立的双卡 DDP 组。
- 当前代码为 `38a90d2` 加未提交工作区修改，已将源码差异与 23 个变化/新增源文件保存到 `results/rebuttal/2026-10-26-p0/code_snapshot/`，共约 190KB；运行任务未被修改或重启。

进度条 ETA 不含完整周期性 validation 开销，本记录不据此推算最终用时。后续效果应依据 validation 与固定设置下的 test cvr 指标，不由初始 loss 判断。
