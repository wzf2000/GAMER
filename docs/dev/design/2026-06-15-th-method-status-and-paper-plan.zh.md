# 时序-层级方法现状、取舍与后续论文计划

## 文档目的

本文档用于内部沟通同步和后续论文写作，统一记录当前 Temporal-Hierarchical（TH）方向中：

- 已经实现和完成实验的方案。
- 已经实现但仍等待实验结果的方案。
- 暂不建议继续作为主线、预计降级为消融或放弃的方案。
- 后续考虑实现的模型、辅助目标和序列增强方向。
- 推荐的论文方法组织方式和实验优先级。

当前方法目标仍然是：

```text
Generative Temporal-Hierarchical Behavioral Modeling for Recommendation
```

核心观点是：用户多行为交互同时具有时间顺序和行为层级两个维度。模型应在遵守因果性的前提下，同时保留普通 next item prediction、行为层级内部 next item prediction 和跨层级 action/conversion prediction 的能力。

## 当前统一建模框架

当前主线继续使用每个 item 的最深行为状态序列，而不是将完整行为链铺平。

例如一个 item 达到：

```text
pxs -> click -> activation -> conversion
```

输入中仍然只保留：

```text
conversion
```

该 token 被解释为 item 的行为链状态，而不是孤立行为标签。

模型架构采用 replacement-style TH attention：

```text
Layer 0-1: standard causal attention
Layer 2-5: temporal-hierarchical attention
Layer 6-7: standard causal attention / generation refinement
```

TH attention 的共同基础组件包括：

- 行为层级 Q/K/V embedding。
- attention output gating。
- causal 和 item 内 semantic-token mask。
- behavior-aware MoE/FFN。

因此，即使不加入额外 scalar relation bias，TH 模型也与普通 Qwen3 有本质区别。

## 已尝试方案总览（持续更新）

### 状态表

| 类别 | 方案 | 状态 | 当前定位 |
|---|---|---|---|
| TH Base | FixedBias / FixedZero | 已实现、已实验 | 干净的 TH base；实际是无 scalar relation bias 的 TH base |
| Relation Bias | Factorized | 已实现、已实验 | 可学习 relation bias 主线候选 |
| Relation Bias | FixedSoft | 已实现、已实验 | 新 TH 变体中 merged behavior 和多数靠前 CVR 指标最强 |
| Relation Bias | FactorizedSoft | 已实现、已实验 | 带弱先验初始化；未优于 Factorized/FixedSoft |
| Relation Bias | FactorizedScale | 已实现、已实验 | 当前 relation-bias 扩展中在若干 CVR 覆盖指标上最好 |
| Relation Bias | FactorizedAlpha | 已实现、待实验 | 每层可学习 alpha 控制 relation bias 强度 |
| Relation Bias | Naive trainable table | 已实现并 profiling | 极慢 backward，预计放弃 |
| Multi-View | Hard MultiView | 已实现、已实验 | 明显弱于 TH Base/relation-bias 系列；结构化消融 |
| Multi-View | Soft MultiView | 已实现、已实验 | 优于 Hard MultiView，但仍弱于 FixedSoft/Factorized |
| Multi-View | Gated MultiView | 已实现、待实验 | 每个 head 学习 view mixture |
| Objective | Next behavior-level auxiliary objective | 已实现、已实验 | CVR 结果有正有负，保留为辅助目标消融 |
| Objective | Relation regularization | 已实现、已实验 | 稳定改善 FactorizedSoft CVR，优先保留的 objective-side 扩展 |
| Data | Hybrid random-ratio + semantic multi-view augmentation | 已实现、已实验 | 当前最强增强结果，等待固定样本预算对照 |

## Relation-Bias 方案（主线候选，部分待实验）

### 基础公式

Relation-bias 系列在 TH attention score 上加入行为层级关系项：

```text
score(i, j)
  = q_i k_j / sqrt(d)
  + causal_mask(i, j)
  + alpha * relation_bias(level_i, level_j)
```

它保留所有因果历史的可见性，通过连续 bias 调整不同行为层级 pair 的相对重要性，比 hard mask 更自由。

### FixedBias / FixedZero（已实现、已完成实验）

配置：

```json
"th_relation_bias_type": "table",
"th_relation_bias_trainable": false,
"th_relation_bias_init": "zero"
```

该版本的 scalar relation bias 恒为零，因此实际贡献来自：

- replacement-style TH attention。
- behavior-aware Q/K/V。
- output gating。
- behavior-aware MoE。

它当前在 ShortVideoAD 的 conversion 和 merged behavior-specific 任务上表现最好。

内部和论文中建议改称：

```text
TH Base
TH Attention w/o Relation Bias
TH Embedding-only
```

不建议继续将其解释为真正的 Fixed Relation Bias，因为 zero table 本身没有提供 relation prior。

### Factorized Relation Bias（已实现、已完成实验）

Factorized 版本学习：

```text
query_factor[level, head, rank]
key_factor[level, head, rank]
```

并计算：

```text
bias(q_level, k_level, head)
  = query_factor[q_level, head] · key_factor[k_level, head]
```

优势：

- 保留可学习层级 pair relation。
- 反向传播效率正常。
- 性能接近 TH Base。
- 方法叙事最能体现“learnable temporal-hierarchical relation modeling”。

风险：

- 当前 zero-init factorized 结果略低于 TH Base。
- relation bias 直接影响 attention logits，可能对已经有效的 TH Q/K/V 表示产生干扰。
- rank 和 bias 强度仍需要调节。

### Soft Prior（已实现、已实验）

FixedSoft 和 FactorizedSoft 使用：

```json
"th_relation_bias_init": "soft",
"th_relation_bias_soft_scale": 0.05
```

当前 soft prior 抑制低层级 query 关注更高层级 key，使信息流更偏向：

```text
shallow evidence -> deeper behavior prediction
```

最新 test-set 结果显示，soft prior 对 fixed/frozen relation family 是有帮助的：FixedSoft 是新 TH 变体中 merged behavior 和多数靠前 CVR 指标最强的版本。但同样结论不适用于 factorized family，FactorizedSoft 弱于 Factorized。这说明 prior 方向、prior 强度和参数化方式需要一起调整。

### Fixed Scale 和 Learnable Alpha（部分已实验，继续优先考虑）

FactorizedScale：

```text
score = base_score + 0.1 * relation_bias
```

用于验证 Factorized 略弱是否源于 relation bias 过强。最新结果显示 FactorizedScale 在 CVR `HR@5/R@5/N@5` 上超过 TH Base，但在 `HR@1/R@1/HR@10/N@10` 上仍不如 FixedBias，说明 scale control 有用但还没有形成稳定主模型优势。

FactorizedAlpha：

```text
score = base_score + alpha_l * relation_bias
```

其中每个 TH layer 有一个可学习标量，初始值为 `0.1`。

该方案允许模型从接近 TH Base 的状态开始，自主决定每层需要多少 relation bias。

建议重点记录训练后每层 alpha：

- alpha 接近零：该层不需要额外 relation bias。
- 中间层 alpha 较大：支持不同深度承担不同 TH relation modeling。
- alpha 为负：需要检查 relation 定义或训练稳定性。

### Naive Trainable Table（已完成性能分析、预计放弃）

原始可学习完整 table：

```text
level_pair_bias[q_level, k_level, head]
```

在 `batch=8, seq_len=1024` 的 profiling 中：

```text
trainable table: ~12918 ms/step
factorized:        ~121 ms/step
multi-view:         ~97 ms/step
```

原因是 advanced indexing 展开到 `[B,H,L,L]` 后，反向传播需要将巨大梯度 scatter-add 回小 table。

结论：

- 不再作为正式模型方向。
- 只保留为 profiling 和说明 factorized 设计必要性的工程记录。
- 论文主实验不需要报告该版本的推荐性能。

## Multi-View 方案（部分完成，作为结构化对照）

### Hard MultiView（已实现、已完成实验，考虑降级为消融）

Hard MultiView 将 attention heads 固定分配为：

- temporal：所有因果历史可见。
- same：只保留同层级关系。
- up：强调浅层历史到深层 query。
- down：强调深层历史到浅层 query。

优势：

- 直接对应 TH 二维图中的不同建模视角。
- 可解释性强。
- 计算效率正常。

当前结果：

- 相对 Original GAMER 并不稳定占优。
- 弱于主要 FixedSoft/Factorized TH 候选。

可能原因：

- hard mask 过度限制信息流。
- 固定 head allocation 未必适合所有层和所有 target behavior。
- temporal/same/up/down 的重要性会随 query level、用户和 session 阶段变化。

当前定位：

- 保留为证明多视角 TH 分解有效的重要消融。
- 暂不作为最终主模型。

### Soft MultiView（已实现、已完成实验）

Soft MultiView 不再将不符合 view 的位置设为负无穷，而是使用有限负偏置。

当前实现使用均匀 view mixture：

```text
soft_bias = -scale * average(block_temporal, block_same, block_up, block_down)
```

目的：

- 保留 view prior。
- 允许被抑制关系仍可参与建模。
- 验证 hard mask 是否是 MultiView 弱于 relation-bias 的主要原因。

最新结果说明：

- Soft MultiView 优于 Hard MultiView，尤其在 CVR 靠前指标上有所恢复。
- 但 merged behavior 弱于 FixedSoft，深层 CVR 指标弱于 Factorized。
- 因此它适合作为 stronger structured-view ablation，而不是当前最终主模型。
- `th_multi_view_soft_bias_scale` 后续仍可做消融。

### Gated MultiView（已实现、待实验，优先于复杂动态 Gate）

Gated MultiView 为每个 head 学习 temporal/same/up/down 的 mixture weight：

```text
view_weight[h] = softmax(gate_logits[h])
```

当前 gate 按原 hard head allocation 初始化，再允许训练调整。

优势：

- 保留 MultiView 的可解释性。
- 避免固定 view/head 对应关系。
- 可以分析不同 layer/head 最终偏好的关系类型。

当前限制：

- gate 只依赖 head，不依赖 query level、用户或 hidden state。
- 它是 static learned gate，不是真正 context-aware gate。

如果 Gated MultiView 有效，下一步可考虑：

```text
gate = f(query_hidden, query_level, layer)
```

但动态 gate 会增加复杂度，建议只在 static gated 结果明确有效后实现。

## 辅助 Objectives 与正则（已实现、已实验）

辅助目标目前已经以 opt-in config 形式实现。原则仍然是以 next-token generation 为主目标；所有新增 loss 默认关闭，只有配置中的权重大于 `0` 时才生效。

### 方案 A：Next Behavior Level Prediction（已实现）

```text
L = L_next_token + lambda_level * L_next_level
```

当前实现是在“下一个 token 是 behavior token”的位置，用当前位置 hidden state 预测下一个行为层级。这样可以补上当前 decoder loss 的监督空缺：behavior token 会作为上下文输入，但主 LM loss 中这些 token 被 mask 掉，不会直接预测 behavior token。

配置字段：

- `th_level_auxiliary_loss_weight`: 大于 `0` 时启用 level head。
- `th_level_auxiliary_position`: 当前为 `next_behavior_token`。
- `th_level_auxiliary_ignore_index`: 默认 `-100`。
- `th_level_auxiliary_head_bias`: level head 是否使用 bias。

新增配置：

- `Qwen3TemporalHierarchicalMultiViewSoftLevelAux`
- `Qwen3TemporalHierarchicalFixedSoftLevelAux`
- `Qwen3TemporalHierarchicalFactorizedSoftLevelAuxReg`

建议初始权重：`0.05`。

### 方案 B：Relation-Bias Regularization（已实现）

Relation regularization 对有效的 learned relation-bias matrix 和目标先验之间加入弱 MSE 约束：

```text
L = L_next_token
  + lambda_level * L_next_level
  + lambda_relation * MSE(relation_bias, relation_prior)
```

当前第一版 prior 使用与 FixedSoft/FactorizedSoft 一致的 soft hierarchy prior。该正则只在 relation-bias 模块存在可训练 relation 参数时产生贡献，因此 frozen fixed-table 配置不会因为该实现而改变原有行为。

配置字段：

- `th_relation_regularization_weight`: 大于 `0` 时启用 regularizer。
- `th_relation_regularization_target`: `soft` 或 `zero`。
- `th_relation_regularization_soft_scale`: soft prior 的 scale。
- `th_relation_regularization_include_special_level`: level `0` 是否参与 MSE。

新增配置：

- `Qwen3TemporalHierarchicalFactorizedSoftReg`
- `Qwen3TemporalHierarchicalFactorizedSoftLevelAuxReg`

建议初始权重：`0.01`。

### ShortVideoAD Test-Set 实验结果

第一批辅助目标实验均使用 `smb_explicit_decoder_4`、ShortVideoAD `smb_explicit` test set，并针对每个实验使用同结构无辅助 loss 版本作为基线。以下以 CVR 为主要目标，merged behavior 为辅助指标。

CVR 目标行为结果：

| Variant | HR@5 | HR@10 | R@5 | R@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MultiViewSoft | 0.1274 | 0.1958 | 0.0966 | 0.1504 | 0.0708 | 0.0885 |
| MultiViewSoft + LevelAux | 0.1293 | 0.1955 | 0.0977 | 0.1474 | 0.0700 | 0.0865 |
| FactorizedSoft | 0.1274 | 0.1947 | 0.0972 | 0.1503 | 0.0690 | 0.0867 |
| FactorizedSoft + RelationReg | **0.1305** | 0.1972 | **0.0985** | **0.1518** | **0.0702** | **0.0878** |
| FactorizedSoft + LevelAux + RelationReg | 0.1304 | **0.1981** | 0.0975 | 0.1493 | 0.0698 | 0.0870 |
| FixedSoft 参照 | 0.1349 | **0.1981** | 0.1007 | 0.1513 | 0.0735 | 0.0900 |

Merged behavior 结果：

| Variant | HR@5 | HR@10 | R@5 | R@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MultiViewSoft | 0.1418 | 0.2102 | 0.0718 | 0.1130 | 0.0609 | 0.0742 |
| MultiViewSoft + LevelAux | 0.1432 | 0.2111 | 0.0728 | 0.1133 | 0.0614 | 0.0745 |
| FactorizedSoft | 0.1434 | 0.2099 | 0.0729 | 0.1126 | 0.0615 | 0.0744 |
| FactorizedSoft + RelationReg | 0.1413 | 0.2099 | 0.0718 | 0.1128 | 0.0607 | 0.0739 |
| FactorizedSoft + LevelAux + RelationReg | **0.1441** | **0.2123** | **0.0732** | **0.1140** | **0.0619** | **0.0751** |
| FixedSoft 参照 | 0.1450 | 0.2121 | 0.0743 | 0.1142 | 0.0628 | 0.0756 |

两类目标的作用可以明确区分：

1. RelationReg 是当前最清晰的正向结果。相对 FactorizedSoft，它使 CVR 八项指标全部提升，其中 `HR@5 +2.41%`、`HR@10 +1.29%`、`R@5 +1.32%`、`R@10 +1.02%`、`N@5 +1.71%`、`N@10 +1.30%`，rank 1 指标也同时提升。Merged 小幅下降说明它更像是把模型容量重新分配到稀疏高层目标，而不是对所有行为无差别增益。
2. LevelAux 单独使用没有形成稳定 CVR 提升。在 MultiViewSoft 上，`HR@5/R@5` 上升，但 `R@10` 下降 `2.00%`，`N@10` 下降 `2.25%`。Merged 指标小幅改善，说明它有利于一般行为序列表征，但可能与目标 item 排序产生梯度冲突。
3. LevelAux 和 RelationReg 叠加后，在 FactorizedSoft objective 消融中取得最好的 merged 结果并提升 CVR hit rate，但相对只使用 RelationReg，CVR recall 和 NDCG 下降。LevelAux 可能将 RelationReg 的部分高层收益重新分配给浅层或 merged behavior。
4. FixedSoft 的绝对 CVR 表现仍然更强。RelationReg 缩小了 learnable FactorizedSoft 与 frozen prior 之间的差距，但还不能证明可学习 relation matrix 已经优于固定先验。

当前取舍是：优先将 RelationReg 作为 objective-side 主线候选；LevelAux 保留为有解释价值的消融，只有降低权重或改成 target-aware/high-level transition supervision 后再继续。由于当前差异多数只有零点几个百分点，正式稳定性结论仍需多随机种子验证。

### 方案 C：Behavior Transition Type Prediction（后续考虑）

预测相邻或 query-key pair 的 relation：

```text
same / up / down / temporal-mixed
```

它与 MultiView 的语义直接对应，可用于：

- 辅助监督 Gated MultiView。
- 约束 Factorized learned bias。
- 提供更强的 attention 解释。

风险是 pair 数量大，不建议对所有 token pair 做 dense classification。可只在 item/behavior token 位置采样少量 pair。

### 方案 D：Conversion / Upward Progression Objective（后续考虑）

给定浅层行为交互，预测同一 item 是否在后续达到更深行为：

```text
L_upward = BCE(progress_to_deeper_level)
```

优点：

- 直接匹配 conversion 任务。
- 可以缓解高层行为稀疏。

风险：

- 需要定义观察窗口。
- 对未发生 conversion 的样本存在 censoring，不能简单视为真实负样本。
- 容易让主模型过度偏向 conversion，而损伤 merged behavior。

建议作为第二阶段 objective，不作为第一个辅助目标。

后续 relation-side 扩展可以继续考虑层间一致性、低秩/稀疏性、单调性约束或 sampled relation-type supervision。仍不建议直接做 dense pair classification。

## 序列增强现状（已实现，但需要重新设计）

### 当前 Augmentation（已实现，保留为基础对照）

`SMBExplicitDatasetForDecoder` 当前为每个用户生成统一数量的增强序列：

```text
downsample_ratios = 1/augment, 2/augment, ..., 1
behavior_drop_ratio = ratio / (level + 1)
```

特点：

- 所有用户使用相同的 ratio schedule。
- target behavior 不删除。
- 低层行为删除更多，高层行为删除更少。
- 删除位置在同一行为层级内随机选择。
- 不考虑时间间隔、session 边界、用户活跃度和用户自身行为分布。

### 当前 Fixed-Ratio Dataset（已实现，适合作为控制实验）

`SMBFixedRatioDatasetForDecoder` 将每个用户的低层行为数量裁剪到相对于 target behavior 的统一比例。

例如：

```text
level_ratios = [5, 1, 1]
```

表示最低层行为最多保留到 target-level 行为数量的 5 倍。

优势：

- 控制低层行为过多导致的训练分布失衡。
- train/valid/test 通过确定性用户 seed 保持 prefix 一致。

限制：

- 所有用户共享同一比例。
- target behavior 数为零时无法做比例归一。
- 随机删除不考虑交互新旧程度。
- 可能删除近期高价值浅层证据，保留很久以前的低价值行为。
- 全局比例可能抹平用户真实偏好强度和漏斗差异。

## 后续序列增强设计设想（待实现）

以下方案先作为设计候选，不在本阶段直接实现。

### 方案 1：Time-Decayed Behavior Dropout（优先考虑）

删除概率同时取决于行为层级和时间距离：

```text
p_drop(i)
  = base_ratio(level_i)
  * time_decay(delta_t_i)
```

设计原则：

- 越旧的交互越容易被删除。
- 越新的交互越应保留。
- 高层行为使用更低删除概率。
- target behavior 永远保留。

可选时间函数：

```text
linear rank decay
exponential decay exp(-delta_t / tau)
piecewise recent/mid/old buckets
```

与当前 TH 模型的契合点：

- TH attention 负责建模保留下来的时序和层级关系。
- augmentation 模拟不同长度和不同历史完整度的时间窗口。
- 避免纯随机 dropout 破坏最关键的近期 action evidence。

建议优先级：高。

### 方案 2：Session-Aware Dropout（优先考虑）

以 session 为单位，而不是独立 interaction 随机删除：

- 最近 session 完整保留。
- 历史 session 按时间或活跃度采样。
- session 内部保持原始时序和行为链状态。

可以进一步规定：

- 含 conversion 的 session 更高概率保留。
- 只含浅层行为的远期 session 更高概率删除。
- 至少保留一个最近 session 和一个含高层行为的 session。

优势：

- 不破坏 session 内行为结构。
- 与当前 session IDs 和 TH 时间维度一致。
- 更接近用户历史窗口截断或 session retrieval。

建议优先级：高。

### 方案 3：User-Adaptive Ratio（中高优先级）

不使用所有用户统一的 `[5,1,1,...]`，而是根据用户自身行为漏斗估计 ratio：

```text
r_u,l = smoothed_count_u,l / smoothed_count_u,target
```

再将它收缩到全局先验：

```text
r'_u,l = beta_u * r_u,l + (1 - beta_u) * r_global,l
```

其中：

- 行为较多的用户使用更个性化 ratio。
- 行为较少的用户更多依赖 global ratio。

可设置上下界，避免极端用户 ratio 失控。

优势：

- 保留不同用户真实的行为强度和转化漏斗差异。
- 避免统一 ratio 过度修剪高活跃用户或几乎不处理低活跃用户。

风险：

- ratio 本身可能泄露用户最终 conversion propensity。
- 对稀疏用户估计不稳定。
- train/valid/test 必须只使用对应时间点之前的历史统计，避免未来泄漏。

建议优先级：中高。

### 方案 4：Dataset-Level Fixed Behavior Proportion（中等优先级，适合作为对照）

与当前“相对每个用户 target count”不同，先从训练集估计目标 behavior proportion：

```text
pi = [pi_pxs, pi_click, pi_activation, pi_conversion]
```

augmentation 时让每个样本向该比例靠近，但不强制完全一致。

可以有两种目标：

- Natural proportion：接近原始训练集分布。
- Balanced proportion：提高高层行为相对权重。

建议使用 soft cap，而不是强制精确比例：

```text
keep_count_l <= cap_l(sequence_length, target_level)
```

优势：

- 比用户 target count 为零时更稳。
- 易于控制训练集层级分布。

风险：

- 仍然是用户无关的 global 策略。
- balanced proportion 可能造成训练与测试分布偏移。

建议优先级：中。

### 方案 5：Target-Conditioned Augmentation（中高优先级）

根据当前训练样本的 target behavior 决定历史保留策略。

示例：

- target=conversion：更多保留 click/activation 和近期浅层证据。
- target=click：保留更多 same-level 和 temporal history，不强制保留远期 conversion。
- merged task：按 target level 动态选择增强策略。

形式：

```text
p_keep(i | target_level)
```

与 TH 模型关系最紧密，因为它直接构造不同 query level 下的训练视图。

风险：

- augmentation 策略和 target 强绑定，可能让模型通过输入分布猜 target。
- 需要保证评测时 target behavior prompting 与训练协议一致。

建议优先级：中高，但应在时间/session-aware 策略之后。

### 方案 6：Multi-View Sequence Augmentation（中高优先级，后续考虑）

对同一个用户历史生成少量具有明确语义的视图，而不是多个无语义随机 ratio：

```text
full temporal view
recent-window view
same-level-preserving view
upward-evidence-preserving view
session-subsampled view
```

这些视图可以与模型侧 MultiView 或 relation-bias 形成对应：

- full temporal view 对应普通 causal next-item modeling。
- same-level view 对应 per-level next-item prediction。
- upward view 对应 action/conversion prediction。

优势：

- 每个增强样本有明确 TH 解释。
- 论文叙事一致。
- 比统一 ratio schedule 更适合内部分析和消融。

风险：

- 数据量随 view 数增长。
- 不同 view 可能高度重复。
- 需要控制每类 view 的采样权重。

建议优先级：中高，适合作为最终论文的数据侧 TH 扩展。

### 方案 7：Curriculum Augmentation（低优先级，远期考虑）

训练早期使用较完整历史，后期逐步提高 dropout 或增加困难视图：

```text
early: full / weak dropout
middle: time-aware / session-aware dropout
late: aggressive sparse-history views
```

目的：

- 先学习稳定 item/behavior 表示。
- 再提升对历史缺失和行为稀疏的鲁棒性。

风险：

- 训练流程更复杂。
- 与 early stopping、resume 和缓存 dataset 的兼容需要额外设计。

建议优先级：低于静态 time/session-aware augmentation。

## 预计放弃或降级的方向（当前决策）

### 放弃：Naive Trainable Relation Table

原因：长序列 backward 成本不可接受，Factorized 已提供更合理替代。

### 降级为消融：Hard MultiView

原因：有效但当前性能弱于 relation-bias 系列；hard mask 可能限制信息流。

保留价值：证明 temporal/same/up/down 结构化分解有效。

### 暂不作为主输入：完整 Flatten Behavior-Event Sequence

原因：

- 混合 item 内行为进展和 item 间时间顺序。
- 显著增长序列长度。
- 使高层行为更稀疏。
- 方法收益难以和输入变化分离。

保留为输入表示消融。

### 暂缓：复杂动态 MultiView Gate

原因：static gated view 尚未完成实验。在确认 static gate 有效前，不建议实现 query/user-conditioned dynamic gate。

### 不建议继续使用：无语义统一 Ratio Schedule 作为最终增强方法

当前统一 ratio augmentation 可以保留为 baseline，但最终增强设计应至少加入时间、session 或用户行为分布因素。

## 推荐的后续执行顺序（行动计划）

### 第一阶段：完成已实现模型实验（当前优先）

1. FactorizedAlpha。
2. MultiViewGated。
3. scaled setting 下的 Factorized rank 消融。
4. 可选的 Soft MultiView scale 消融。

统一记录：

- conversion 指标。
- merged behavior 指标。
- p3s/click/cvr 分行为指标。
- 训练时间和显存。
- learned alpha 或 view gate。

### 第二阶段：确定最终模型主线（紧随实验结果）

推荐判定规则：

- 如果 FactorizedAlpha 达到/超过 TH Base：作为 relation-bias 主模型。
- 如果 relation-bias 扩展仍表现为局部提升或整体弱于 TH Base：主模型采用 TH Base，并将核心贡献定义为 behavior-aware replacement TH attention，而不是 scalar relation bias。
- 如果 Gated MultiView 明显提升：可考虑与 Factorized 组合，但应先验证复杂度和可解释性收益。

### 第三阶段：细化已完成首轮实验的辅助目标与正则

`lambda_level=0.05` 和 `lambda_relation=0.01` 的第一轮实验已经完成。下一步优先对 FactorizedSoft + RelationReg 做多随机种子验证，并进行 `0.003/0.01/0.03` 的小范围 relation weight 搜索。LevelAux 仅建议以 `0.01` 等更低权重或面向高层 transition 的监督形式继续；如果需要严格归因，再补充 FactorizedSoft + LevelAux、不含 RelationReg 的独立对照。

### 第四阶段：重新设计序列增强（后续重点）

推荐先实现：

1. Time-Decayed Behavior Dropout。
2. Session-Aware Dropout。
3. User-Adaptive Ratio。

之后再考虑：

4. Target-Conditioned Augmentation。
5. Multi-View Sequence Augmentation。

## ShortVideoAD 已完成 TH 变种结果（已实验）

结果路径：

```text
results/ShortVideoAD/smb_explicit_decoder_4/
```

评测任务为 test set 上的 `smb_explicit` behavior-specific next-item prediction。以下表格记录 test-set 口径下的 merged behavior 与 conversion/cvr 行为。其中 MBGen 作为主要公开 baseline，Original GAMER / Old GAMER SID 作为此前方法参考。

### 主 baseline 对比

Merged behavior 使用主要 baseline 可对齐的四个指标进行对比：

| Model | HR@5 | HR@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: |
| MBGen (SID) | 0.1129 | 0.1696 | 0.0461 | 0.0564 |
| Original GAMER / Old GAMER SID | 0.1443 | 0.2129 | 0.0621 | 0.0753 |
| TH-FixedBias | 0.1444 | 0.2116 | 0.0620 | 0.0750 |
| TH-FixedSoft | 0.1450 | 0.2121 | 0.0628 | 0.0756 |
| TH-Factorized | 0.1430 | 0.2117 | 0.0614 | 0.0746 |
| TH-FactorizedScale | 0.1432 | 0.2113 | 0.0614 | 0.0745 |
| TH-FactorizedSoft | 0.1434 | 0.2099 | 0.0615 | 0.0744 |
| TH-MultiView | 0.1391 | 0.2062 | 0.0595 | 0.0723 |
| TH-MultiViewSoft | 0.1418 | 0.2102 | 0.0609 | 0.0742 |

Merged behavior 上，所有 TH 变体都明显强于 MBGen。与 Original GAMER 相比，结论更细：TH-FixedSoft 是新 TH 变体中最强的版本，并在 `HR@5/N@5/N@10` 上小幅提升，但 `HR@10` 仍略低于 Original GAMER。因此 merged behavior 不能写成全面超过原始 GAMER，更合适的表述是保持竞争力，并在部分靠前排序指标上有小幅改善。

CVR 目标行为对比：

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MBGen (SID) | 0.0261 | 0.0985 | 0.1576 | 0.0187 | 0.0698 | 0.1139 | 0.0491 | 0.0637 |
| Original GAMER / Old GAMER SID | 0.0394 | 0.1280 | 0.1944 | 0.0292 | 0.0966 | 0.1478 | 0.0687 | 0.0856 |
| TH-FixedBias | 0.0390 | 0.1283 | 0.1974 | 0.0290 | 0.0963 | 0.1507 | 0.0693 | 0.0873 |
| TH-FixedSoft | 0.0435 | 0.1349 | 0.1981 | 0.0326 | 0.1007 | 0.1513 | 0.0735 | 0.0900 |
| TH-Factorized | 0.0409 | 0.1342 | 0.2042 | 0.0302 | 0.1011 | 0.1565 | 0.0721 | 0.0902 |
| TH-FactorizedScale | 0.0393 | 0.1331 | 0.1987 | 0.0301 | 0.0988 | 0.1514 | 0.0706 | 0.0877 |
| TH-FactorizedSoft | 0.0385 | 0.1274 | 0.1947 | 0.0294 | 0.0972 | 0.1503 | 0.0690 | 0.0867 |
| TH-MultiView | 0.0381 | 0.1283 | 0.1949 | 0.0275 | 0.0958 | 0.1461 | 0.0678 | 0.0845 |
| TH-MultiViewSoft | 0.0427 | 0.1274 | 0.1958 | 0.0331 | 0.0966 | 0.1504 | 0.0708 | 0.0885 |

CVR 目标行为上，新 TH 变体相对 MBGen 和 Original GAMER 的提升更明确。TH-FixedSoft 在多数靠前 CVR 指标上最好，TH-Factorized 在更深的排序和覆盖指标上最好。这是目前支持 Temporal-Hierarchical redesign 的最强证据：改动不只是改变 merged behavior 的指标取舍，而是实质提升了目标高层行为建模。

### TH 变体细节：Merged Behavior

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TH-FixedBias | 0.0465 | 0.1444 | 0.2116 | 0.0223 | 0.0731 | 0.1135 | 0.0620 | 0.0750 |
| TH-FixedSoft | 0.0476 | 0.1450 | 0.2121 | 0.0225 | 0.0743 | 0.1142 | 0.0628 | 0.0756 |
| TH-Factorized | 0.0465 | 0.1430 | 0.2117 | 0.0216 | 0.0725 | 0.1131 | 0.0614 | 0.0746 |
| TH-FactorizedScale | 0.0454 | 0.1432 | 0.2113 | 0.0218 | 0.0726 | 0.1133 | 0.0614 | 0.0745 |
| TH-FactorizedSoft | 0.0460 | 0.1434 | 0.2099 | 0.0218 | 0.0729 | 0.1126 | 0.0615 | 0.0744 |
| TH-MultiView | 0.0439 | 0.1391 | 0.2062 | 0.0210 | 0.0709 | 0.1105 | 0.0595 | 0.0723 |
| TH-MultiViewSoft | 0.0460 | 0.1418 | 0.2102 | 0.0220 | 0.0718 | 0.1130 | 0.0609 | 0.0742 |

在新 TH 变体内部，TH-FixedSoft 在当前 test-set 口径下是整体最强版本。TH-FixedBias 仍是很接近且稳定的 TH base；Hard MultiView 明显偏弱，MultiViewSoft 只能部分弥补 hard partition 带来的损失。

整体 merged 排序可暂记为：

```text
FixedSoft > FixedBias ~= Factorized > FactorizedSoft ~= FactorizedScale > MultiViewSoft > Hard MultiView
```

相对 TH-FixedBias：

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TH-FixedSoft | +2.53% | +0.42% | +0.26% | +1.31% | +1.54% | +0.62% | +1.19% | +0.79% |
| TH-Factorized | +0.07% | -0.94% | +0.06% | -2.77% | -0.85% | -0.36% | -0.98% | -0.63% |
| TH-FactorizedScale | -2.33% | -0.81% | -0.13% | -1.83% | -0.73% | -0.24% | -1.03% | -0.77% |
| TH-FactorizedSoft | -0.94% | -0.67% | -0.80% | -2.23% | -0.32% | -0.81% | -0.82% | -0.90% |
| TH-MultiView | -5.56% | -3.62% | -2.55% | -5.46% | -3.12% | -2.70% | -4.09% | -3.59% |
| TH-MultiViewSoft | -1.11% | -1.80% | -0.63% | -0.99% | -1.80% | -0.50% | -1.80% | -1.11% |

### TH 变体细节：Conversion / CVR Behavior

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TH-FixedBias | 0.0390 | 0.1283 | 0.1974 | 0.0290 | 0.0963 | 0.1507 | 0.0693 | 0.0873 |
| TH-FixedSoft | 0.0435 | 0.1349 | 0.1981 | 0.0326 | 0.1007 | 0.1513 | 0.0735 | 0.0900 |
| TH-Factorized | 0.0409 | 0.1342 | 0.2042 | 0.0302 | 0.1011 | 0.1565 | 0.0721 | 0.0902 |
| TH-FactorizedScale | 0.0393 | 0.1331 | 0.1987 | 0.0301 | 0.0988 | 0.1514 | 0.0706 | 0.0877 |
| TH-FactorizedSoft | 0.0385 | 0.1274 | 0.1947 | 0.0294 | 0.0972 | 0.1503 | 0.0690 | 0.0867 |
| TH-MultiView | 0.0381 | 0.1283 | 0.1949 | 0.0275 | 0.0958 | 0.1461 | 0.0678 | 0.0845 |
| TH-MultiViewSoft | 0.0427 | 0.1274 | 0.1958 | 0.0331 | 0.0966 | 0.1504 | 0.0708 | 0.0885 |

CVR test 结果改变了之前的判断。TH-FixedSoft 在多数靠前 CVR 指标（`HR@1/HR@5/N@5`）上最强，TH-Factorized 在更深的覆盖和排序指标 `HR@10/R@5/R@10/N@10` 上最强。TH-MultiViewSoft 的 `R@1` 最高，但其余 CVR 指标偏弱。TH-FixedBias 仍是强 base，但在修正后的 test-set 对比下已经不是最好的 CVR 目标行为模型。相对 Original GAMER，TH-FixedSoft 带来更广泛的靠前 CVR 提升，TH-Factorized 则带来最明确的更深排序 CVR 提升。

### 当前解释

1. TH-FixedBias 的收益主要不来自 scalar relation-bias 数值本身。当前 FixedBias 是 frozen zero scalar bias，因此更准确的解释是 replacement-style TH attention、behavior Q/K/V、gating 和 behavior-aware MoE 带来的结构收益。
2. 可学习 relation bias 在 CVR 目标行为上是有价值的。Factorized 改善了更深的 CVR 指标，FixedSoft 则改善了多数靠前 CVR 指标和 merged behavior。相对 Original GAMER，最可靠的提升集中在 CVR，而不是 merged behavior 的所有指标。
3. Soft prior 对 fixed/frozen relation family 有帮助，但对 factorized family 并不稳定。这说明 prior 强度和参数化方式需要一起调整。
4. Hard MultiView 明显偏弱。Soft MultiView 能部分弥补损失，尤其在靠前 CVR 指标上更好，但还不足以作为强于 FixedSoft 或 Factorized 的主模型。

### 对方法主线的影响

当前更稳妥的论文主线应更新为：

```text
TH Base
+ fixed soft hierarchy prior 或 factorized relation bias
```

TH-FixedBias 应保留为基础消融，而不是最终默认主模型。TH-FixedSoft 是新 TH 变体中 merged behavior 和多数靠前 CVR 指标上的最强候选；TH-Factorized 是更深 CVR 排序和覆盖指标上的最强候选。加入 Original GAMER 对比后，最终论文 claim 应更强调目标行为 CVR 增益，并将 merged behavior 表述为具有竞争力且部分指标小幅提升。

Hard MultiView 建议降级为结构化消融；Soft MultiView 应作为更强的 structured-view comparison，因为它保留视角先验但避免了 hard visibility cut。

## 论文写作建议

### 方法结构

论文方法可以组织成三层：

1. Compact behavior-chain-state sequence。
2. Replacement-style Temporal-Hierarchical Attention。
3. TH-aware training enhancement，包括 relation control、辅助目标或序列增强。

### 主实验和消融

建议命名：

- `TH Base`: behavior-aware Q/K/V + gating，无 scalar relation bias。
- `TH-FRB`: Factorized Relation Bias。
- `TH-FRB-Soft` 或 `TH-FRB-Alpha`: 最终 relation-bias 主模型候选。
- `TH-MV`: Hard MultiView。
- `TH-MV-Gated`: Gated MultiView。

消融应回答：

- replacement TH attention 是否有效。
- scalar relation bias 是否额外有效。
- soft prior/alpha 是否改善 relation learning。
- hard view 和 soft/gated view 有何差异。
- TH-aware augmentation/objective 是否进一步提升高层行为和 merged behavior。

### 当前最稳妥的论文结论

当前 test-set 结果已经支持：

- replacement-style TH attention 优于旧 added cross-level attention GAMER。
- 显式行为层级 Q/K/V 建模对 conversion 和 merged behavior 都有效。
- controlled relation bias 可以提升 CVR 排序和覆盖。
- fixed soft hierarchy prior 可以提升 merged behavior 和多数靠前 CVR 指标。
- Soft MultiView 优于 Hard MultiView，但还不是最强模型线。
- soft-prior relation regularization 在首轮 test-set 实验中稳定改善 FactorizedSoft CVR。
- 当前 next-level auxiliary objective 小幅改善 merged behavior，但没有形成稳定 CVR 增益。
- 相对 MBGen，TH 变体在 merged behavior 和 CVR 上都有大幅提升；相对 Original GAMER，最稳定的提升集中在 CVR。

仍需实验才能支持：

- 某一种 relation-bias 参数化可以稳定优于所有其他版本。
- factorized soft prior 一定有效。
- gated view mixture 可以稳定超过 relation-bias 系列。
- gated MultiView 优于 hard MultiView。
- auxiliary-objective 增益能够跨随机种子和 relation-regularization 权重保持稳定。
- hybrid augmentation 的收益在控制生成视图数量后仍然成立。

## 当前结论

当前模型侧主线应从 FixedBias-only 更新为：

```text
TH Base
+ FixedSoft hierarchy prior 用于 merged/top-rank behavior
+ Factorized relation bias 用于 CVR depth/coverage
```

最新 ShortVideoAD test-set 结果显示，`TH-FixedSoft` 在新 TH 变体中 merged behavior 和多数靠前 CVR 指标上最好，`TH-Factorized` 在更深 CVR 指标上最好。相对 MBGen，这些变体明显更强；相对 Original GAMER，CVR 提升明确，但 merged behavior 更复杂，FixedSoft 在 `HR@5/N@5/N@10` 上小幅提升，同时在 `HR@10` 上略低。`TH-FixedBias` 因为 scalar relation bias 是 frozen zero，仍应作为干净的 TH base ablation，但不应再描述为最佳最终变体。`TH-MultiViewSoft` 仍是有价值的 structured-view 消融；Hard MultiView 主要说明过硬的 view partition 会限制模型表达。

辅助目标侧，第一轮 test-set 消融支持 soft-prior RelationReg 作为 learnable FactorizedSoft relation 的有效稳定器。当前 LevelAux 形式不适合作为主模型增益点：它偏向改善 merged behavior 和短列表 hit rate，但会损伤若干 CVR recall/NDCG 指标。论文中应明确区分这两个目标，而不是将它们合并表述为统一正向的 auxiliary-objective 结果。

数据侧不建议继续仅依赖所有用户统一的随机 ratio schedule。更符合 TH 设计的增强方向是：

```text
time-aware
+ session-aware
+ behavior-level-aware
+ optionally user-adaptive
```

最终应让模型结构、辅助目标和序列增强共同围绕同一个二维时序-层级视角，而不是各自独立增加复杂度。
