# GAMER CVR 排序任务设计与实现

## 1. 任务概述

该任务面向 session 内的候选 item 进行 CVR 倾向评分。对于每个候选，模型联合使用用户历史
行为和候选 item 信息，输出一个标量 `logit`。候选按 `logit` 从高到低排列后，即得到最终的
排序结果。

在 ShortVideoAD 数据集上，最高层级行为为 `cvr`，因此训练目标定义为二分类：

| 候选交互的 behavior | `ranking_label` |
| --- | ---: |
| `cvr` | 1 |
| `p3s` 或 `click` | 0 |

行为层级由
[`ShortVideoAD.behavior_level.json`](../data/ShortVideoAD/ShortVideoAD.behavior_level.json)
定义，其中 `p3s=0`、`click=1`、`cvr=2`。

模型在架构上复用生成式 backbone `Qwen3TemporalHierarchical`，但当前任务不执行 token 生成，
而是通过判别式 ranking head 直接优化 CVR 二分类目标。完整计算链路如下：

```text
用户历史 + 候选 item
    -> 拼接为单条 token 序列
    -> Qwen3TemporalHierarchical 编码
    -> 提取历史表示与候选表示
    -> ranking head 输出 CVR logit
    -> 加权二分类损失
```

## 2. 数据划分与样本构造

### 2.1 目标 session 的选择

每个用户至少需要包含 3 个 session。数据集按时间顺序选择目标 session，并将目标 session
之前的交互作为用户历史：

| 数据划分 | 目标 session | 历史范围 |
| --- | --- | --- |
| train | 倒数第 3 个 session | 该 session 之前的交互 |
| valid | 倒数第 2 个 session | 该 session 之前的交互 |
| test | 最后 1 个 session | 该 session 之前的交互 |

训练脚本默认设置 `max_his_len=100`，因此最多保留最近 100 条历史交互。

### 2.2 样本粒度与标签

训练和验证阶段以“用户、目标 session 内的一条交互记录”为基本样本粒度。假设某个目标
session 包含以下三条交互：

| 交互记录 | candidate item | behavior | label |
| --- | --- | --- | ---: |
| 1 | item_A | `p3s` | 0 |
| 2 | item_B | `click` | 0 |
| 3 | item_C | `cvr` | 1 |

数据构造后得到：

```text
(同一份用户历史, item_A) -> 0
(同一份用户历史, item_B) -> 0
(同一份用户历史, item_C) -> 1
```

标签由当前交互记录的 behavior 直接确定，不会先对 item 进行 session 内聚合。因此，同一 item
如果以不同 behavior 多次出现，将形成多条独立样本。

测试集在数据文件中以 session 为单位保存，评估时再将目标 session 内的交互逐条展开并打分。
这一实现避免在数据缓存中为每个候选重复保存相同历史，同时保持与训练阶段一致的候选级
标签定义。

### 2.3 正负样本来源

当前实现不使用随机负采样，也不设置人工采样比例。目标 session 内的真实交互均参与训练：

- `cvr` 交互构成正例；
- `p3s`、`click` 等非 CVR 交互构成负例。

因此，正负样本比例由原始数据分布决定。类别不平衡通过损失函数中的正例权重处理，而不是
通过重采样改变数据分布。

## 3. 输入表示

### 3.1 历史交互

每条历史交互由一个显式 behavior token 和一组 item semantic-ID token 组成。例如，一个 item
由 4 个 semantic-ID token 表示时，历史序列可写为：

```text
[<behavior_p3s>   item_A_token_1 ... item_A_token_4]
[<behavior_click> item_B_token_1 ... item_B_token_4]
...
```

这种表示同时保留了用户访问过的 item 以及每次访问对应的行为层级。

### 3.2 候选 item

候选不包含其真实 behavior token，而是统一在 item semantic-ID token 前加入最低层级行为作为
已知输入条件。ShortVideoAD 的最低层级行为是 `p3s`：

```text
[历史 behavior-item 序列] [<behavior_p3s> candidate_token_1 ... candidate_token_4]
```

CTR 和 CVR 都使用相同的 `<behavior_p3s>` candidate 条件。候选真实 behavior 仅用于生成
二分类标签；如果将真实 behavior token 加入候选输入，模型会在打分前直接获得标签信息。

历史与候选被拼接到同一条自回归序列中，而不是分别进入两个独立编码器。因果注意力保证：

- 候选 token 可以访问其之前的完整用户历史；
- 历史 token 无法访问位于其后的候选。

因此，候选的 hidden state 已经融合用户历史。同一个 item 面对不同用户时，可以形成不同的
上下文表示和 CVR 分数。

## 4. 行为层级信息的注入

### 4.1 action index 映射

`behavior_level` 与模型实际接收的 action index 并不相同。行为编号统一加 1，将 0 保留给
padding、未知或不适用的情况：

| token 类型 | action index |
| --- | ---: |
| padding、特殊 token、behavior 未知 | 0 |
| 历史 `p3s`、candidate 条件及其 item tokens | 1 |
| 历史 `click` | 2 |
| 历史 `cvr` | 3 |

Candidate 的 query token 和全部 semantic-ID tokens 都使用 `p3s` 对应的 action index 1。
CTR/CVR 的目标行为只决定标签阈值，不改变 candidate 输入及 relation action。

### 4.2 Temporal-Hierarchical Attention

`Qwen3TemporalHierarchical` 在部分注意力层中使用 action index 注入行为层级信息。当前训练
脚本默认选择 `Qwen3TemporalHierarchicalFactorized`，其 `relation_bias` 采用可学习的低秩
参数化形式。

Candidate 以 `p3s` 层级参与 relation bias 或 `multi_view` 行为 mask；所有模式继续使用
因果 mask。

## 5. 候选打分模型

训练任务将 `ranking_score_type` 设置为 `llm_pair`。该模式从 backbone 最后一层 hidden state
中提取以下特征：

1. `history_state`：候选起始位置前一个 token 的 hidden state；
2. `candidate_state`：`<behavior_p3s>` 和候选全部 semantic-ID token 的 hidden state 均值；
3. `interaction_state`：`history_state * candidate_state`，即两者的逐元素乘积。

最终特征为：

```text
pair_feature = concat(
    history_state,
    candidate_state,
    history_state * candidate_state,
)
```

`pair_feature` 经过两层 MLP 得到单个候选分数：

```text
logit = Linear -> PReLU -> Linear(pair_feature)
```

`logit` 是未经 sigmoid 的原始分数。计算概率时可使用 `sigmoid(logit)`；执行候选排序时可以
直接比较 logit，因为 sigmoid 不改变候选之间的相对顺序。

当前配置设置 `ranking_use_user_embedding=False`，不额外引入独立的 user embedding。用户信息
完全由历史序列提供。训练过程中 backbone 与 ranking head 联合更新，backbone 不冻结。

## 6. 优化目标与类别不平衡

模型使用带正例权重的 `BCEWithLogitsLoss`。正例权重由完整训练集统计得到：

```text
pos_weight = 负例数量 / 正例数量
loss = BCEWithLogits(logit, ranking_label, pos_weight=pos_weight)
```

例如，训练集包含 9 万个负例和 1 万个正例时，`pos_weight=9`。该权重提高了少数正例在损失
中的贡献，但不会删除负例、复制正例或改变训练样本数量。

数据样本中的字符串字段 `labels="<behavior_cvr>"` 仅作为行为元数据保留。真正进入 ranking
loss 的字段是数值型 `ranking_labels`，因此当前训练不包含 behavior token 的生成损失。

## 7. 验证与测试

### 7.1 训练期验证

当前训练脚本默认从 valid 候选中抽取 2048 条样本计算 `eval_auc_sampled`。该指标用于：

- 比较并选择 checkpoint；
- 配合 `patience` 执行 early stopping；
- 在控制验证成本的前提下监控 CVR 区分能力。

`eval_auc_sampled` 是抽样验证指标，不等同于最终完整测试结果。

### 7.2 默认测试路径

测试脚本默认请求 AUC、PRAUC、LogLoss、Accuracy、Precision、Recall、F1 和 GAUC。这些指标均
属于二分类指标，因此评估程序走快速 CVR 路径：

1. 读取每位用户最后一个 session；
2. 将该 session 中的实际交互作为候选；
3. 对每条候选输出 CVR logit；
4. 根据候选对应的 behavior 构造 0/1 标签；
5. 在完整测试记录上汇总指标。

该结果衡量模型在目标 session 实际交互上的 CVR 区分能力，不代表全 item 库召回性能。

### 7.3 全库排序路径

测试代码同时支持全 item 集打分。当请求 Recall@K、NDCG@K 等非二分类指标时，评估程序会对
全量 item 候选进行排序。该路径与默认 CVR 二分类评估使用同一个 checkpoint 和 ranking head，
但候选范围及指标含义不同，汇报结果时应明确区分。

## 8. 设计边界

当前实现可归纳为以下边界：

- 任务是“是否达到目标行为层级”的二分类，不是 `p3s/click/cvr` 三分类；
- 候选不携带真实 behavior token，统一使用最低层级 `<behavior_p3s>` 条件和 action index 1；
- 负例来自目标 session 内的真实非 CVR 交互，不使用随机负采样；
- `history_state`、`candidate_state` 和交叉特征仅在模型内部使用，不导出独立特征文件；
- ranking head 直接输出最终候选分数，不存在额外训练的下游排序模型；
- 默认 CVR 指标与全库 Top-K 排序指标属于两种不同的评估设置。

## 9. 代码入口

| 模块 | 代码位置 |
| --- | --- |
| session 划分、样本与标签构造 | [`ranking.py`](../SeqRec/datasets/session_behavior/ranking.py) |
| ranking batch 字段整理 | [`generative.py`](../SeqRec/datasets/collators/generative.py) 中的 `DecoderOnlyRankingCollator` |
| 正例权重、训练配置与抽样 AUC | [`train_SMB_ranking_decoder.py`](../SeqRec/tasks/training/train_SMB_ranking_decoder.py) |
| 历史—候选特征与 ranking head | [`wrappers.py`](../SeqRec/models/generative/common/wrappers.py) |
| Temporal-Hierarchical Attention | [`temporal_hierarchical.py`](../SeqRec/models/generative/qwen3/temporal_hierarchical.py) |
| 默认训练参数 | [`train_SMB_ranking_decoder.sh`](../scripts/train_SMB_ranking_decoder.sh) |
| 默认测试参数与评估入口 | [`test_SMB_ranking_decoder.sh`](../scripts/test_SMB_ranking_decoder.sh) |

## 10. CTR 任务

CTR 版本复用上述 CVR 实验的 session 划分、候选粒度、模型、训练超参数、类别权重和二分类指标，
仅将正例阈值改为 `click`：

| 候选交互的 behavior | `ranking_label` |
| --- | ---: |
| `click` 或 `cvr` | 1 |
| `p3s` | 0 |

CTR 的任务名为 `smb_ctr_ranking_decoder`。CTR 和 CVR 的候选查询都使用
`<behavior_p3s>`；缓存和 checkpoint 路径会因任务名及数据集类不同而隔离。训练与测试继续
使用原脚本：

```bash
tasks=smb_ctr_ranking_decoder suffix=frozen_probe pretrained_model=/path/to/checkpoint \
  bash ./scripts/train_SMB_ranking_decoder.sh

tasks=smb_ctr_ranking_decoder test_task=smb_ctr_ranking_decoder suffix=frozen_probe \
  bash ./scripts/test_SMB_ranking_decoder.sh
```

也可以使用直接入口对比 P3s condition 与原始 target-behavior condition（base），并分别运行
`frozen_probe`、`cold_start` 和 `full_finetune`。每组训练结束后都会自动选择各自的 best
checkpoint 测试：

```bash
# 默认运行 CTR 的 2 种 condition × 3 种训练策略
bash ./scripts/train_test_SMB_ranking_decoder.sh

# CVR 的完整对比
task=cvr bash ./scripts/train_test_SMB_ranking_decoder.sh

# 只运行 base 或 P3s
conditions=base bash ./scripts/train_test_SMB_ranking_decoder.sh
conditions=p3s bash ./scripts/train_test_SMB_ranking_decoder.sh

# 也可以限制训练策略
conditions=base strategies=full_finetune bash ./scripts/train_test_SMB_ranking_decoder.sh
```

完整 CTR 实验可使用统一入口：

```bash
# GAMER 三种设置 + 全部判别式 baseline
bash ./scripts/train_SMB_CTR_all.sh

# 也可以只运行其中一组
experiments=gamer bash ./scripts/train_SMB_CTR_all.sh
experiments=baselines bash ./scripts/train_SMB_CTR_all.sh
```

该入口将 `Qwen3TemporalHierarchicalFixedSoft` 的 `frozen_probe`、`cold_start` 和
`full_finetune` 依次运行在 GPU 0–3 上。`frozen_probe` 与 `full_finetune` 从指定 decoder
checkpoint 的 `original/` 模型初始化；按定义，`cold_start` 只使用相同模型配置并随机初始化，
不加载 checkpoint 权重。

判别式 baseline 包括 MeanPooling、DIN、DIENCVR、BSTCVR、HSTUCVR、SASRecCVR 和 DSIN。
它们沿用各自 CVR 脚本中的 batch size、epoch 数和二分类指标；MeanPooling 首先建立共享 CTR
缓存，其余模型随后分配到 GPU 0–3 并行训练。可通过 `checkpoint_root`、`gpu` 环境变量覆盖
默认 checkpoint 和设备。

为确保 CTR/CVR 可比，统一入口会清空外部 `extra_args` 与 `extra_flags`，并显式固定以下配置：

| 配置 | GAMER 三种设置 | 判别式 baseline |
| --- | ---: | ---: |
| seed | 42 | 42 |
| max history length | 100 | 100 |
| script batch size | 1024（4 卡时每卡 256） | 1024（单卡） |
| optimizer | AdamW (`adamw_torch`) | AdamW (`adamw`) |
| learning rate | 1e-3 | 1e-3 |
| weight decay | 0.01 | 0.01 |
| maximum epochs | 3 | 3 |
| validation interval | 每个 epoch | 每个 epoch |
| early stopping patience | 2 | 2 |
| early stopping metric | sampled validation AUC | validation GAUC |
| checkpoint used for test | best model | best model |
| metrics | sampled AUC + 完整二分类指标 | 完整二分类指标 |

GAMER 三种设置之间唯一的训练差异是初始化与冻结策略：

| 设置 | decoder 初始化 | 可训练参数 |
| --- | --- | --- |
| frozen_probe | 指定 checkpoint | ranking head |
| cold_start | 随机初始化 | 全模型 |
| full_finetune | 指定 checkpoint | 全模型 |

CTR 相对 CVR 的任务差异是标签目标从 `cvr` 改为 `click`，以及相应的任务名、缓存和输出路径；
两者的 candidate 输入条件均为 `p3s`，模型配置与上述实验配置保持不变。
