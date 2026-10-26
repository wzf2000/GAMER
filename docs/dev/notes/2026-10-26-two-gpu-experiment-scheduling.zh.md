# 两张 A100 的补实验排期依据

日期：2026-10-26。代码为 `38a90d2` 之后尚未提交的补实验支持工作区。本记录仅用于排期，不替代正式论文效率实验。

## 当前资源与优先任务

首次检查时 GPU 1/5/6/7 均无计算进程；本次仅使用原先讨论的 GPU 5/7，两张均为 A100-SXM4-80GB。

优先使用双卡 DDP，按同一 seed 成对运行：

```text
full42 → plain_matched42 → full43 → plain_matched43 → full44 → plain_matched44
```

该顺序优先得到关键的参数容量对照，避免先完成三个 full 才开始等容量 decoder。之后再运行 no_qkv、no_gate、shared_ffn、zero_prior 各一次。raw 对照等待真实输入；Tmall 与 lambda 属于后续优先级；离线稀疏分桶不占用 GPU。

生成的 `results/rebuttal/2026-10-26-p0/run_train_paired.sh` 直接复用既有计划中的六条命令，仅改变队列顺序，不改变配置或启动任何任务。通过 bash -n，且保留已有 checkpoint 的拒绝覆盖检查。

## 实测方法

- 配置来自 `results/rebuttal/2026-10-26-p0/configs/{full,plain_matched}/config.json`。
- FP32，TF32 关闭，SDPA，训练模式，源配置 dropout；共 101 个五 token item 块加 EOS，长度 506。
- 使用合成 token 序列和 masked causal LM labels，屏蔽行为 token。没有读取实际训练样本、更新参数或创建 optimizer。
- 单卡测试 batch=32/64/128；各一次 warmup、三次计时，CUDA synchronize，报告中位数。两种模型分别在 GPU 5 和 GPU 7 测量。
- 双卡 DDP 每卡 batch=128，梯度每步 all-reduce，采用较慢 rank 的耗时；同样一次 warmup、三次计时。真实梯度累积会减少同步频率。
- 时间包含 forward、LM loss、backward；DDP 另含梯度通信。不含 optimizer step、数据加载、周期性 validation、test 或保存。

## 结果

| 模型 | 单卡 batch | 单卡前反向中位数 | 单卡 peak allocated |
| --- | ---: | ---: | ---: |
| full | 32 | 0.1881 s | 11.06 GiB |
| full | 64 | 0.3370 s | 21.97 GiB |
| full | 128 | 0.6387 s | 43.79 GiB |
| 等容量 decoder | 32 | 0.2536 s | 15.43 GiB |
| 等容量 decoder | 64 | 0.4968 s | 30.74 GiB |
| 等容量 decoder | 128 | 0.9724 s | 61.32 GiB |

单卡 batch=128 时 allocator peak reserved 分别为 49.96/68.37 GiB。两者均能容纳；实际 optimizer、数据和完整训练仍需另计。

| 模型 | 双卡每卡 batch | 双卡一次前反向 | 吞吐相对单卡 | rank0 peak allocated |
| --- | ---: | ---: | ---: | ---: |
| full | 128 | 0.6439 s | 1.984× | 43.89 GiB |
| 等容量 decoder | 128 | 0.9736 s | 1.998× | 61.66 GiB |

在两组处理相同样本数的假设下，每组样本数/128 归一化后的成对纯计算时间为：

- 每模型各占一张卡并行：`max(0.6387, 0.9724) = 0.9724`。
- 每模型使用两张卡、依次完成：`(0.6439 + 0.9736)/2 = 0.8088`。

后一方案约少 16.8% 的成对纯计算等待时间，同时保持现有计划的双卡训练设置。该比例不保证等于完整流程的提速。

建议沿用每卡 batch=128、梯度累积=8、双卡有效 batch=2048、FP32、SID `.index.json`、历史 100、seed 42/43/44。不能直接把旧 4×A800 的 22.57h 或 8×H800 的 8.63h 换算成此处整次训练时间。

临时原始记录：`/tmp/gamer-full-capacity.json`、`/tmp/gamer-plain-capacity.json`、`/tmp/gamer-ddp-capacity.json`；测量脚本在 `/tmp/gamer_rebuttal_capacity_probe.py` 与 `/tmp/gamer_rebuttal_ddp_capacity_probe.py`。正式效率复现应将脚本和完整运行条件另行固化。
