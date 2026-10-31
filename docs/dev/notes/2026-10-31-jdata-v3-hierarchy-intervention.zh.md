# JData-V3 指定层级置换实验

英文对应：`2026-10-31-jdata-v3-hierarchy-intervention.md`。时间均为 Asia/Shanghai。实现提交：`389c99a`。

## 范围与对照纠正

作者确认论文 Tmall 主方法使用弱辅助层级，而不是此前 Tmall plan 生成的 legacy identity-order `full`。不能把该生成对照当作论文那次运行。本次暂缓 Tmall 层级干预，改做 JData。

用户提供的 `JData-V3.zip` 包含 TH-FixedSoft／`smb_explicit_decoder_4`／CID-shuffle-chunk64 的汇总及逐用户指标，没有 checkpoint/config/training_args、UID sidecar、索引文件／hash 或评测 manifest。名义 test 的 buy 行为 **HR@10 0.726086957、NDCG@10 0.550338765**，共 **230** 个用户；merged 为 **0.516895038／0.322759241**。五种行为的逐用户均值已与汇总核对。test split 依据 `results-smb_explicit-cid-shuffle-64.json` 命名推断，尚无旧执行记录独立佐证。

## 预先固定的干预

| 行为身份 | V3 自然层级 | 本次 prior 层级 |
| --- | --- | --- |
| pv | 0 | 2 |
| click | 1 | 1 |
| cart | 2 | 0 |
| collect | 2 | 0 |
| buy | 3 | 3 |

变体为 `random_auxiliary_reversed`，训练 seed **42**。该强干预在训练前由用户明确指定，是一次辅助层级组的固定置换，不是多个随机映射的分布，也不是用 seed42 抽出的映射。cart/collect 仍并列，buy 仍最高。各数值层级对应的身份数量发生变化，因此检验的是这个特定 prior 分配及其偏置距离，不能声称隔离了“是否并列”的影响。

只改变 TH attention 使用的层级 lookup。行为 token、Q/K/V 身份 embedding、gate、位置专家／FFN 路由、输入、增强、目标、候选、数据切分及因果 mask 均保持一致。λ 仍为 **0.05**，relation table 冻结，运行时倍率为 1，辅助目标关闭。数据层级仍为 `0,1,2,2,3`，不会被实验 ranks 覆盖。

## V3 支持与验证

当前 checkout 原先支持 legacy identity-order prior，但缺少 V3 schema 处理。本次仅通过显式数据 schema 启用身份／层级分离：仍由原 router 产生身份编号，在 TH attention 内用非持久化 lookup 转成层级编号，保留 router 返回接口及其他 backbone。没有切换分支或 cherry-pick。

V3 的五种身份对应四个层级：Q/K/V embedding 含特殊编号 0，共六行；冻结 relation table 形状为 `5 × 5 × 6`。自然 lookup 是 `[0,1,2,3,3,4]`，本次为 `[0,3,2,1,1,4]`，特殊编号 0 仍映射到 0。加载 checkpoint 时由 config 重建 lookup。Legacy `th_prior_behavior_ranks` 仍沿用原有行为间 table 初始化语义。

已将自然和置换两个 tiny V3 模型与历史 V3 提交 `eadc431` 的只读导出逐项比较：全部 state tensor、logits、loss、Q 投影梯度及缓存 beam 输出完全一致，缓存／非缓存生成也一致。新增测试覆盖 planner／trainer 配置一致、schema 指纹、causal shifted 监督、冻结 bias、梯度、重载及非法 ranks。相关套件共 **53 项测试通过**：模型 18、plan／queue／seed／pipeline 17、session-behavior 数据集 18；四张 A100 上也通过两步 DDP 反向和缓存／非缓存 beam 验证。规定 flake8、compileall 和 whitespace 检查均通过。

JData-V3 元数据及输入链接沿用历史 V3 定义，复用本机 JData 的事件／session／time／item 文件。CID 已在 CPU 按 chunk64／shuffle／seed42 独立生成，与本机 V2 索引逐字节相同，SHA256 为 `7270735f9a5109d7797b1c5bafe63dca73144291cd417a24d6f38d4d88fe5222`；这不证明与历史训练机器的 V3 索引 hash 相同。已独立生成 V3 缓存，没有覆盖 V2 缓存或 checkpoint。

| 数据 | 样本数 |
| --- | ---: |
| factor4 训练 | 49,448 |
| 训练期间全行为 validation loss | 65,224 |
| Validation ranking：pv／click／cart／collect／buy | 4,651／9,098／1,255／233／263 |
| Test ranking：pv／click／cart／collect／buy | 4,361／9,087／1,137／219／230 |

五种 test 行为计数全部与用户归档匹配；计数相同仍不等于历史 target／UID／索引逐项一致。数据来源、缓存、压缩包成员及索引的 SHA256 记录在 `results/rebuttal/2026-10-31-jdata-v3-preflight/inputs-and-historical-results.json`。

## 实际配方与执行

作者补充了主方法命令：`dataset=JData-V3 cid=1 shuffle=1 chunk_size=64 batch_size=512 tasks=smb_explicit_decoder_4 backbone=Qwen3TemporalHierarchicalFixedSoft gpu=0,1,2,3 port=2314 bash scripts/train_SMB_decoder.sh --max_his_len 100 --learning_rate 5e-4 --num_train_epochs 200 --gradient_accumulation_steps 8 --warmup_ratio 0.04 --patience 20 --temperature 0.7`。

脚本将 batch512 除以四张卡，因此 **每卡128 × 四卡 × 累积8 = 有效 batch4096**。本次按这些显式参数对齐。Weight decay0.01、cosine、AdamW、seed42、FP32、最大长度1024 来自解析后的默认值，原命令本身不能证明旧 checkout 上这些默认值没有变过。保留训练 shell 的 `CUDA_LAUNCH_BLOCKING=1`，本次 W&B 使用 offline。Ranking 使用四卡、每卡 batch32、beam20，评测全部五种行为并导出 UID／原始 target。

本次唯一启动的变体于 **2026-10-31 19:59:13** 在 GPU **0,1,2,3**（四张 **A100-SXM4-80GB**）启动，端口 **24631**，启动时 supervisor PID 为 **2440452**。实际 start manifest 确认 **20,381,184 个可训练参数**、SDPA、FP32、五身份／四层级、ranks `[2,1,0,0,3]`、两种缓存样本数及有效 batch。本文记录启动与初步核验，不表示训练已经完成。

运行根目录：`results/rebuttal/2026-10-31-jdata-v3-hierarchy/`。

- `plan.json`：可复现的三配置清单；其中 `full` 已采用自然 V3 层级，`weak_auxiliary` 在这里与其数值相同，两个对照均未启动。
- `random-only-plan.json`：实际执行的单项清单。
- `pipeline.log`、`job.start.json`、`launch.json`：实时日志与进程／来源记录；supervisor 退出后生成 `job.finish.json`。
- `random_auxiliary_reversed/seed_42/checkpoint/`：最终选定权重、trainer state、config 与训练 manifest。
- `random_auxiliary_reversed/seed_42/validation-ranking.json`：倒数第二个 session 的 ranking（`smb_explicit_valid`）。
- `random_auxiliary_reversed/seed_42/test-ranking.json`：最后一个 session 的 ranking（`smb_explicit`）。
- 结果同名目录：逐用户指标、UID sidecar 和原始 target 导出。
- 成功完成后生成 `verified-results.json` 与 `metrics.csv`：核验 root／best 权重、split 来源、逐用户均值、计数、target 导出绑定、merged 加权值及缓存 hash 未变。

从仓库根目录执行的准确 pipeline 命令为：

```bash
python scripts/run_rebuttal_experiment.py \
  --plan ./results/rebuttal/2026-10-31-jdata-v3-hierarchy/random-only-plan.json \
  --variants random_auxiliary_reversed --seeds 42 \
  --gpu 0,1,2,3 --port 24631 --execute
```

该任务已经启动，请勿重复执行。它按 **全行为 validation loss** 选择最终权重，再自动进行独立 validation ranking 和 test ranking；任一阶段失败即停止后续阶段。Supervisor 会导出最终核验 CSV；在相关阶段结束前不报告最终结果。此前的 V2 草稿 `results/rebuttal/2026-10-31-jdata-random/` 从未启动。

## 尚需核实的可比性

现已对齐历史命令和模型／结果系列，且本机 V3 数值语义与历史实现一致。压缩包仍不能确认历史数据／索引 hash、准确 checkpoint/config、环境、beam／评测 batch 或实际 split 执行；在补齐前，不能把与历史分数的差异宣称为严格 prior 单因素效应。没有旧 UID 配对或显著性检验依据。本实验只有一个训练 seed 和一个明确指定的层级置换；buy ranking 不等于购买概率估计。
