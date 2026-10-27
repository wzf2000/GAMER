# FFN 修复与 Tmall 两种 SASRec 重训

英文对应：`2026-10-27-sasrec-ffn-fix-reruns.md`。

## 实现与验证

提交 `d8d06c3` 将共享 `FeedForward.forward` 的条件由 `if not self.residual` 改为 `if self.residual`。启用残差时正确执行 dropout、输入相加和 LayerNorm；关闭残差时返回投影结果，不再访问不存在的模块。构造参数、参数形状及 state-dict key 不变。按旧错误前向训练的权重，仅换到新前向加载不能视为已修复训练。

4 项 CPU 回归测试通过：零投影下残差保留和模块调用、关闭残差路径、原子／扩展词表 SASRec 的归一化梯度及优化器更新、PBAT／MB-STR FFN 调用方的前／反向。规定的 flake8（`--max-line-length=500 --ignore=F401,E203,W503,F841`）、`main.py`／`SeqRec` compileall，以及 `bash -n scripts/train_SMB_rec.sh` 均通过。

在空闲 GPU 5 上，使用 Tmall 实际扩展词表规模（1,447,728 个 item 类别）、batch 4096、历史 20、FP32、AdamW 完成合成单步训练验证。loss 有限（**14.21317482**），所有 FFN 归一化梯度有限且非零，峰值 allocated 显存为 **69,731.25 MiB**。这是容量／功能检查，不是训练后的推荐效果或效率基准。

首次正式 SASRec_B 启动随后在初始 validation 后因 CUDA 分配器碎片 OOM；单步预检未覆盖这一分配顺序。第二次 GPU 预检在 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 下覆盖 validation batch 4096／2821、训练 batch 4096／881、再次 validation 和再次完整训练 batch。全部通过，loss／梯度有限，峰值 allocated 为 **71,154.40 MiB**，未更改 batch、模型或精度。当前 SASRec_B 重试使用该分配器设置，首次失败记录保留。

## 已授权重训

用户要求优先重跑 Tmall 两种 SASRec，其他流程差异暂缓。本次保留已找回的任务身份、当前源配置、采样 validation 选模和全量排序 test 流程。历史最终超参仍缺失，因此这是显式记录的新运行，不能证明与旧分数只相差一个变量。

| 变体 | 物理 GPU | 训练任务 | Test 任务 | Item 类别数 |
| --- | ---: | --- | --- | ---: |
| SASRec | 1 | `smb_dis_sample` | `smb_dis` | 361,932 |
| SASRec_B | 5 | `smb_dis_sample_diff` | `smb_dis_diff` | 1,447,728 |

共同设置：`Tmall-24-0.25`（非 V2）、seed 42、历史 20、每组单张 A100-SXM4-80GB、batch 4096、FP32、最多 200 epoch、AdamW lr=0.001、weight decay=0.01、patience=20；2 层／2 heads／hidden 128／FFN 256／dropout 0.5／CE。判别式模型不使用 SID／CID 索引。`--add_uid` 仅保留逐用户导出元数据，SASRec loss／forward 不读取 UID。validation 按 alipay ranking、默认 NDCG@10 选模。最终 test 评测四种行为并分别输出行为行及 merged 结果。

已确认数据规模：1,057,649 个训练样本、6,917 个 validation alipay 用户、6,329 个 test alipay 用户。原子／diff 缓存和预期排序后的 test UID 列表分别保留本地。

2026-10-27（Asia/Shanghai），独立 supervisor 分别于 **20:22:31** 启动 SASRec（worker PID 2592796）、**20:25:55** 启动首次 SASRec_B（worker PID 2595620）。SASRec_B 失败后于 **20:29:46** 在 `sasrec_b/seed_42/retry-expandable/` 重试（worker PID 2599878）。SASRec 已在 epoch 3 后保存过最佳采样 validation NDCG@10 checkpoint；目前不声称已获得新的 full-test 结果。

## 不改变抽样的缓存准备

原始冷 validation builder 为每个用户重建 361,932 个 item 的候选池，预计需要数小时。已停止本轮新启动的慢速缓存准备进程，未停止任何训练进程。运行产物中保留的进程内准备工具，将抽中的 rank 映射回同一个升序稠密整数候选池；未修改生产 dataset 代码，超出限定条件会回退原 sampler。

在 CPython 3.12 下，105 个真实历史／边界样例与原方法相比，抽样 item ID 及抽样后 RNG 状态逐项一致。所有用户历史均满足稠密整数、小排除集合的限定条件。抽样 seed、负例、完整 validation 用户集合、target 编码和 collator 保持一致；工具不改变模型代码，也不会将 monkeypatch 带入训练进程。正式任务读取生成的完整缓存。

## 命令与本地产物

运行根目录：`results/rebuttal/2026-10-27-sasrec-ffn-fixed/`。配置快照为各变体的 `seed_42/config/config.json`。当前 SASRec 位于 `sasrec/seed_42/`，SASRec_B 位于 `sasrec_b/seed_42/retry-expandable/`；各自包含 `run_manifest.start.json`、`train.log` 及 `checkpoint/`、`results/` 输出。`active-jobs.json` 指向当前有效 manifest。完整训练／测试进程退出后，supervisor 写入 `run_manifest.finish.json`。训练与聊天进程分离，失败会记录退出状态。

实际命令保存在各 start manifest。以下为从仓库根目录运行的等价命令，使用已准备的配置快照及独立输出路径：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 WANDB_MODE=offline python main.py train_SMB_rec \
  --backbone SASRec --dataset Tmall-24-0.25 --data_path ./data \
  --base_model ./results/rebuttal/2026-10-27-sasrec-ffn-fixed/sasrec/seed_42/config \
  --output_dir ./results/rebuttal/2026-10-27-sasrec-ffn-fixed/sasrec/seed_42/checkpoint \
  --result_dir ./results/rebuttal/2026-10-27-sasrec-ffn-fixed/sasrec/seed_42/results \
  --tasks smb_dis_sample --test_task smb_dis --seed 42 --max_his_len 20 --add_uid \
  --epochs 200 --batch_size 4096 --learning_rate 0.001 --weight_decay 0.01 --optim adamw --patience 20

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py train_SMB_rec \
  --backbone SASRec --dataset Tmall-24-0.25 --data_path ./data \
  --base_model ./results/rebuttal/2026-10-27-sasrec-ffn-fixed/sasrec_b/seed_42/config \
  --output_dir ./results/rebuttal/2026-10-27-sasrec-ffn-fixed/sasrec_b/seed_42/retry-expandable/checkpoint \
  --result_dir ./results/rebuttal/2026-10-27-sasrec-ffn-fixed/sasrec_b/seed_42/retry-expandable/results \
  --tasks smb_dis_sample_diff --test_task smb_dis_diff --seed 42 --max_his_len 20 --add_uid \
  --epochs 200 --batch_size 4096 --learning_rate 0.001 --weight_decay 0.01 --optim adamw --patience 20
```

这些路径已分配给当前任务，不要再次执行命令覆盖。最终 JSON 为运行根目录下的 `sasrec/seed_42/results/result-smb_dis.json` 和 `sasrec_b/seed_42/retry-expandable/results/result-smb_dis_diff.json`。与论文比较时取 **Behavior alipay** 行。运行数据、权重、日志、缓存工具及 manifest 不纳入 Git。

## 普通 SASRec 已完成结果

普通 SASRec 于 **2026-10-27 21:25:10，Asia/Shanghai** 以退出码 0 完成，包括最终全量排序 test。训练在 epoch **37** 后早停，保存的最佳 checkpoint 在 epoch **17** 按采样 validation NDCG@10 选出（日志保留四位为 **0.3514**，不是 test 分数）。完整训练／validation／test 流程耗时 3,759.075 秒，GPU 1 已释放。后续检查时 SASRec_B 仍在约第 37 轮训练，尚无最终 test JSON。

全候选 **test / Behavior alipay**，6,329 个用户：

| 指标 | 论文归档 SASRec | 修复后新运行 | 相对增加 |
| --- | ---: | ---: | ---: |
| HR@1 | 0.036182651 | 0.159424870 | +340.61% |
| HR@5 | 0.065887186 | 0.205719703 | +212.23% |
| HR@10 | 0.076789382 | 0.216621899 | +182.10% |
| Recall@10 | 0.072005407 | 0.202203450 | +180.82% |
| NDCG@5 | 0.049573397 | 0.174461469 | +251.93% |
| NDCG@10 | 0.053329522 | 0.178387853 | +234.50% |

HR@10 达到归档值的 **2.82099 倍**，绝对增加 0.139832517；NDCG@10 达到 **3.34501 倍**，绝对增加 0.125058330。两者仍低于归档 TIGER 的 0.580976458／0.468769537。click、collect、cart 的 HR@10／NDCG@10 也均提高，辅助结果保留在 JSON 中。

已核验：退出成功、权重已保存、源码／config／原始数据／缓存指纹未改变、6,329 条逐用户指标有限且均值与汇总一致、四个 FFN LayerNorm weight/bias 张量均偏离初始化值并实际学到参数。最佳权重 SHA256 为 `f2c79abe19f577f7fd5220457d8c26ae1cfb5dbd8c6ca8e5542ed0ed59e2810b`；结果 SHA256 为 `9c42b397f085f0baa180931e9863f31e9d221e3b0d4dd46f5d169997e81ca1f6`。

证据为运行根目录下的 `sasrec/seed_42/results/result-smb_dis.json`、其 `result-smb_dis/user_level_metrics_behavior_alipay.json`、`run_manifest.finish.json` 和 `verified-test-comparison.json`。旧对照为 `2026-10-27-tmall-baseline-provenance.zh.md` 中已匹配的 `results_wzf.zip` 文件。

这是修复后新运行与论文归档结果的数值比较。历史超参／训练版本缺失，不能严格将所有提升归因于 FFN 单一因素；旧结果缺少 UID 对齐的逐用户指标，不能据此给出配对显著性结论。

## 首轮重训后补齐的历史命令

用户随后提供了普通 SASRec 原始命令，明确使用历史长度 **50**、patience **10**；本次已完成修复重训使用 **20／20**。数据集、backbone、训练／test 任务、batch 4096、lr=0.001、最多 200 epoch 一致。当前 SASRec_B 重训也使用 20／20，但所给历史命令只直接对应普通 SASRec。

| 设置 | 用户提供的历史命令 | 已完成修复重训 |
| --- | --- | --- |
| `max_his_len` | 50 | 20 |
| `patience` | 10 | 20 |
| `add_uid` | 未传，false | true，仅元数据 |
| FFN 计算 | 历史训练 checkout 仍未核验 | `d8d06c3` 修正后的实现 |

本次 config 快照与 `1c941cc`（2025-10-04）中的受版本管理 SASRec config 逐字节一致，SHA256 均为 `5d113d2c49cb06f47cdc72377f6c0b09ac6905af5e56901d363a920b87ee667a`。这支持用户“config 未改”的说明，但不是从原机器直接取回的文件。该历史 parser 也确认，原命令省略的默认值为 seed 42、AdamW、weight decay 0.01、logging step 30，以及相同指标列表／默认 NDCG@10 选模。

即使 JSON config 一样，历史长度仍会改变模型可见输入及可学习 position embedding 的长度。按本次已记录的 best 更新事件回放，patience 10 会在 epoch **27** 停止、仍保留 epoch 17 的 checkpoint；这是针对本次历史 20 轨迹的推算，不是对历史 50 的预测。

因此，**0.216621899／0.178387853 必须继续标为修复后 history-20／patience-20 的新运行**。与所给历史配方匹配的比较，需要重新训练 history-50／patience-10；本次命令核查没有启动这项额外训练。原数据／缓存 hash、运行环境、准确训练版本仍缺失。运行证据：`sasrec/seed_42/historical-command-comparison.json`。

## 新增 history-50／patience-10 运行

用户随后明确要求重训，普通 SASRec 已于 **2026-10-27 22:08:46 Asia/Shanghai** 在物理 **GPU 1（A100-SXM4-80GB）** 启动。supervisor PID：**2669289**；训练 PID：**2669290**。FFN 实现为 `d8d06c3`；准备时仓库 HEAD 为 `d65d7d6`。模型从头初始化，seed 42，SASRec JSON config 和数据指纹与上文一致。该运行随后于 **2026-10-27 23:05:39 Asia/Shanghai** 以退出码 0 完成训练及全候选 test；最佳 checkpoint 为 epoch **11**，epoch **21** 早停。

独立运行根目录：`results/rebuttal/2026-10-27-sasrec-ffn-fixed-history50-patience10/`；普通模型产物位于其 `sasrec/seed_42/`。保留已有历史 20 产物及 SASRec_B 进程。历史 50 的 train／validation／test 缓存使用独立 `.50.` 文件名。缓存准备在 105 组样例上验证进程内采样加速路径的候选和 RNG 状态与原实现相同，保留 1,057,649 条训练样本、6,917 个 alipay validation 用户和 6,329 个 alipay test 用户。本次未修改生产 Python 源码；运行辅助脚本通过 compileall 及规定的 flake8 检查。

从仓库根目录、激活 GAMER 环境后的等价命令如下（这些输出路径已被当前运行使用）：

```bash
run=./results/rebuttal/2026-10-27-sasrec-ffn-fixed-history50-patience10/sasrec/seed_42
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
WANDB_MODE=offline PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python main.py train_SMB_rec \
  --backbone SASRec --dataset Tmall-24-0.25 --data_path ./data \
  --base_model "$run/config" --output_dir "$run/checkpoint" --result_dir "$run/results" \
  --tasks smb_dis_sample --test_task smb_dis --seed 42 --max_his_len 50 --add_uid \
  --epochs 200 --batch_size 4096 --learning_rate 0.001 --weight_decay 0.01 \
  --optim adamw --patience 10 --logging_step 30 \
  --metrics hit@1,hit@5,hit@10,recall@1,recall@5,recall@10,ndcg@5,ndcg@10 \
  --wandb_run_name rebuttal/tmall-ffn-fixed-history50-patience10/sasrec/seed_42
```

`run_manifest.start.json` 保留准确命令、config／数据／源码／缓存指纹及 GPU 身份；`train.log` 记录进度。选模仍使用采样 **validation／alipay／NDCG@10**。早停或达到 200 轮后，任务自动加载 `checkpoint/best_model.pth`，对所有行为执行全候选 **test**。预期汇总文件：`results/result-smb_dis.json`，目标行 **Behavior alipay**。逐用户指标位于 `results/result-smb_dis/`；完整进程结束后 supervisor 写入 `run_manifest.finish.json`。UID 仅用于元数据导出，不使用 SID／CID 索引。匹配已找回的超参仍不代表找回历史训练 checkout、数据／缓存指纹或环境，历史比较的这些限制仍然存在。

## 2026-10-28 完成核查

上述三组修复后的 Tmall 实验均成功退出，保留最佳权重及五行汇总结果（四种行为加 merged），已完成全候选 test。**test／alipay** 结果如下：

| 模型 | 历史长度／patience | HR@10 | NDCG@10 |
| --- | --- | ---: | ---: |
| SASRec | 20／20 | 0.2166218991941855 | 0.17838785251618217 |
| SASRec_B | 20／20 | 0.012166218991941855 | 0.009992610300676929 |
| SASRec | 50／10 | 0.21535787644177595 | 0.17751063055090913 |

三组每项指标均有 6,329 个有限的 alipay 逐用户数值，均值与汇总结果的绝对差小于 1e-12。SASRec_B 的显存分配器重试运行于 **2026-10-27 23:02:29 Asia/Shanghai** 完成，最佳 epoch **19**，epoch **39** 早停。证据为各运行的 `run_manifest.finish.json`、普通模型的 `results/result-smb_dis.json` 或 SASRec_B 的 `results/result-smb_dis_diff.json`，以及对应的 `user_level_metrics_behavior_alipay.json`。本 seed 下普通模型历史 50 与历史 20 的分数接近，不能由此认定跨 seed 等价或隔离历史 FFN 单因素影响。SASRec_B 的 history-50／patience-10 后续实验作为独立新运行处理。

## SASRec_B history-50／patience-10 后续运行

用户要求 SASRec_B 使用相同的新历史长度／patience 设置。该从头训练于 **2026-10-28 00:37:36 Asia/Shanghai** 在物理 **GPU 5（A100-SXM4-80GB）** 启动，supervisor PID **2772948**，训练 PID **2772949**。独立根目录为 `results/rebuttal/2026-10-28-sasrec-b-ffn-fixed-history50-patience10/`，模型运行位于 `sasrec_b/seed_42/`。FFN 代码为 `d8d06c3`，准备时 HEAD 为 `9110dcf`；未修改生产源码。已有历史 20 及普通模型历史 50 产物均保留。

与上一轮修复后的 SASRec_B 相比，仅改变统计设置 `max_his_len: 20 -> 50`、`patience: 20 -> 10`。config 校验和、数据指纹、行为–item 联合表示、seed 42、batch 4096、FP32、AdamW lr=0.001、weight decay 0.01、dropout 0.5、最多 200 轮，以及采样 alipay validation NDCG@10 选模均一致。这是本次要求的新配方，不表示已找回历史 SASRec_B 命令。不使用 SID／CID 索引，UID 元数据导出不参与模型 loss。

历史 50 的 `.diff.adduid` 缓存包含 1,057,649 条训练样本、6,917 个 alipay validation 用户和 6,329 个 alipay test 用户；联合词表有 1,447,728 个 ID，不含 padding。进程内缓存采样加速再次通过 105 组候选／RNG 一致性检查。独立合成模型 GPU 预检覆盖 validation batch 4096／2821、训练 batch 4096／881、再次 validation 及训练 batch 4096，loss／梯度有限，优化器更新正常。使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 时，峰值分配／保留显存为 **73,090.4／73,532 MiB**。运行辅助脚本通过 compileall 和规定的 flake8 检查。

激活 GAMER 环境后、从仓库根目录运行的等价命令如下（这些路径已分配给正在运行的任务）：

```bash
run=./results/rebuttal/2026-10-28-sasrec-b-ffn-fixed-history50-patience10/sasrec_b/seed_42
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
WANDB_MODE=offline PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python main.py train_SMB_rec \
  --backbone SASRec --dataset Tmall-24-0.25 --data_path ./data \
  --base_model "$run/config" --output_dir "$run/checkpoint" --result_dir "$run/results" \
  --tasks smb_dis_sample_diff --test_task smb_dis_diff --seed 42 --max_his_len 50 --add_uid \
  --epochs 200 --batch_size 4096 --learning_rate 0.001 --weight_decay 0.01 \
  --optim adamw --patience 10 --logging_step 30 \
  --metrics hit@1,hit@5,hit@10,recall@1,recall@5,recall@10,ndcg@5,ndcg@10 \
  --wandb_run_name rebuttal/tmall-ffn-fixed-history50-patience10/sasrec_b/seed_42
```

独立 supervisor 保留完整退出状态。训练后，任务会自动加载 `checkpoint/best_model.pth`，对四种行为执行全候选 test。预期汇总为 `results/result-smb_dis_diff.json`，目标行 **Behavior alipay**；逐用户文件为 `results/result-smb_dis_diff/user_level_metrics_behavior_alipay.json`。最终指标待完成。`run_manifest.start.json` 包含准确命令、config／数据／源码／缓存／GPU 身份及上一轮引用；进度保存在 `train.log`，完整退出状态保存在 `run_manifest.finish.json`。
