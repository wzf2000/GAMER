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

候选只包含 item semantic-ID token，不包含其真实 behavior token：

```text
[历史 behavior-item 序列] [candidate_token_1 ... candidate_token_4]
```

候选 behavior 是模型需要预测的目标。如果将真实 behavior token 加入候选输入，模型会在打分
前直接获得标签信息，造成数据泄漏。

历史与候选被拼接到同一条自回归序列中，而不是分别进入两个独立编码器。因果注意力保证：

- 候选 token 可以访问其之前的完整用户历史；
- 历史 token 无法访问位于其后的候选。

因此，候选的 hidden state 已经融合用户历史。同一个 item 面对不同用户时，可以形成不同的
上下文表示和 CVR 分数。

## 4. 行为层级信息的注入

### 4.1 action index 映射

`behavior_level` 与模型实际接收的 action index 并不相同。已知行为的编号统一加 1，将 0
保留给未知或不适用的情况：

| token 类型 | action index |
| --- | ---: |
| padding、特殊 token、behavior 未知的候选 | 0 |
| 历史 `p3s` | 1 |
| 历史 `click` | 2 |
| 历史 `cvr` | 3 |

候选使用 action index 0 表示“behavior 未知”，并不表示候选属于最低层级 `p3s`。这一设计既
避免泄漏预测目标，也使未知状态与所有已知行为保持独立。

### 4.2 Temporal-Hierarchical Attention

`Qwen3TemporalHierarchical` 在部分注意力层中使用 action index 注入行为层级信息。当前训练
脚本默认选择 `Qwen3TemporalHierarchicalFactorized`，其 `relation_bias` 采用可学习的低秩
参数化形式。

action index 为 0 的候选仍然具有对应的 behavior embedding，并参与 relation bias 计算；它
不会被模型忽略。当前配置将 relation bias 的初始输出设为 0，随后由训练数据更新相关参数。

若切换到 `multi_view` 模式，涉及 action index 0 的 token 对不会进入同层、向上或向下的行为
mask，避免因候选行为未知而受到错误的层级约束。所有模式均继续使用因果 mask。

## 5. 候选打分模型

训练任务将 `ranking_score_type` 设置为 `llm_pair`。该模式从 backbone 最后一层 hidden state
中提取以下特征：

1. `history_state`：候选起始位置前一个 token 的 hidden state；
2. `candidate_state`：候选全部 semantic-ID token 的 hidden state 均值；
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

- 任务是“是否发生最高层级行为”的二分类，不是 `p3s/click/cvr` 三分类；
- 候选不携带真实 behavior token，action index 统一设为 0；
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
