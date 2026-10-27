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
