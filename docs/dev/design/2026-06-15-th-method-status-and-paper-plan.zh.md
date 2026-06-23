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
| TH Base | FixedBias / FixedZero | 已实现、已实验 | 当前性能最强；实际是无 scalar relation bias 的 TH base |
| Relation Bias | Factorized | 已实现、已实验 | 可学习 relation bias 主线候选 |
| Relation Bias | FixedSoft | 已实现、已实验 | 固定弱层级先验；整体接近但弱于 FixedBias |
| Relation Bias | FactorizedSoft | 已实现、已实验 | 带弱先验初始化；未优于 Factorized/FixedBias |
| Relation Bias | FactorizedScale | 已实现、已实验 | 当前 relation-bias 扩展中在若干 CVR 覆盖指标上最好 |
| Relation Bias | FactorizedAlpha | 已实现、待实验 | 每层可学习 alpha 控制 relation bias 强度 |
| Relation Bias | Naive trainable table | 已实现并 profiling | 极慢 backward，预计放弃 |
| Multi-View | Hard MultiView | 已实现、已实验 | 明显弱于 TH Base/relation-bias 系列；结构化消融 |
| Multi-View | Soft MultiView | 已实现、已实验 | merged 指标优于 Hard MultiView，但 CVR 仍弱于 TH Base |
| Multi-View | Gated MultiView | 已实现、待实验 | 每个 head 学习 view mixture |
| Objective | Behavior-level auxiliary objectives | 待实现 | 后续增强 TH supervision |
| Data | Existing ratio augmentation | 已实现、已使用 | 用户无关、时间无关，需重新设计 |

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

最新结果显示 soft prior 没有带来稳定收益：

- FixedSoft 在 CVR 上弱于 FixedBias。
- FactorizedSoft 只在 CVR `R@10` 上略高，但靠前排序和 NDCG 不占优。
- 相比 soft prior，FactorizedScale 的固定强度控制更值得保留。

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

- conversion 和 merged 任务均超过旧 GAMER。
- 但弱于 TH Base 和 Factorized。

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

- Soft MultiView 在 merged `HR@5/N@5` 上最好，明显优于 Hard MultiView。
- 但在 conversion/CVR 目标行为上仍弱于 FixedBias。
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

## 辅助 Objectives（待实现，后续重点考虑）

辅助目标尚未实现。原则是继续以 next-token generation 为主目标，仅添加轻量 TH supervision。

### 方案 A：Next Behavior Level Prediction（优先考虑）

```text
L = L_next_token + lambda_level * L_next_level
```

在 behavior token 或 item-level hidden state 上预测下一个行为层级。

优点：

- 标签来自现有行为序列，不需要额外标注。
- 直接强化层级转移意识。
- 实现成本相对最低。

建议优先级：最高。

### 方案 B：Behavior Transition Type Prediction（后续考虑）

预测相邻或 query-key pair 的 relation：

```text
same / up / down / temporal-mixed
```

它与 MultiView 的语义直接对应，可用于：

- 辅助监督 Gated MultiView。
- 约束 Factorized learned bias。
- 提供更强的 attention 解释。

风险是 pair 数量大，不建议对所有 token pair 做 dense classification。可只在 item/behavior token 位置采样少量 pair。

### 方案 C：Conversion / Upward Progression Objective（后续考虑）

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

### 方案 D：Relation-Bias Regularization（可选，依赖主模型结果）

对 Factorized learned relation matrix 加弱约束：

- 与 soft prior 的距离。
- 层间一致性。
- 低秩/稀疏性。
- 对称或单调性约束。

建议先通过可视化观察 learned matrix，再决定是否需要正则，避免过早强加错误先验。

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

### 第三阶段：实现第一个辅助 Objective（下一轮开发）

优先实现 next behavior level prediction：

```text
L = L_next_token + lambda_level * L_next_level
```

建议先测试 `lambda_level=0.05/0.1`。

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

评测任务为 `smb_explicit_valid` behavior-specific next-item prediction。以下表格记录 test 结果中的 merged behavior 与 conversion/cvr 行为。

### Merged Behavior

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TH-FixedBias | 0.0490 | 0.1502 | 0.2227 | 0.0237 | 0.0778 | 0.1220 | 0.0656 | 0.0799 |
| TH-FixedSoft | 0.0492 | 0.1497 | 0.2216 | 0.0236 | 0.0780 | 0.1221 | 0.0652 | 0.0795 |
| TH-Factorized | 0.0487 | 0.1500 | 0.2220 | 0.0234 | 0.0781 | 0.1215 | 0.0655 | 0.0796 |
| TH-FactorizedScale | 0.0482 | 0.1502 | 0.2221 | 0.0234 | 0.0779 | 0.1216 | 0.0654 | 0.0795 |
| TH-FactorizedSoft | 0.0486 | 0.1494 | 0.2199 | 0.0229 | 0.0773 | 0.1203 | 0.0649 | 0.0787 |
| TH-MultiView | 0.0463 | 0.1478 | 0.2162 | 0.0219 | 0.0760 | 0.1179 | 0.0632 | 0.0766 |
| TH-MultiViewSoft | 0.0496 | 0.1508 | 0.2218 | 0.0239 | 0.0780 | 0.1212 | 0.0657 | 0.0796 |

Merged behavior 上，TH-MultiViewSoft 在 `HR@1/HR@5/R@1/N@5` 上最好，TH-FixedBias 仍在 `HR@10/N@10` 上最好，TH-FixedSoft 在 `R@10` 上略高。头部几个版本差距很小，但这改变了此前对 MultiView 的判断：hard partition 弱，而 soft view penalty 在 merged behavior 上是有竞争力的。

整体 merged 排序可暂记为：

```text
FixedBias ~= MultiViewSoft ~= FactorizedScale ~= Factorized > FixedSoft > FactorizedSoft > Hard MultiView
```

相对 TH-FixedBias：

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TH-Factorized | -0.64% | -0.12% | -0.31% | -1.34% | +0.40% | -0.41% | -0.08% | -0.34% |
| TH-FactorizedScale | -1.65% | +0.02% | -0.29% | -1.47% | +0.13% | -0.38% | -0.17% | -0.50% |
| TH-FactorizedSoft | -0.99% | -0.55% | -1.26% | -3.37% | -0.62% | -1.44% | -1.08% | -1.50% |
| TH-FixedSoft | +0.29% | -0.33% | -0.50% | -0.67% | +0.35% | +0.03% | -0.48% | -0.46% |
| TH-MultiView | -5.56% | -1.56% | -2.93% | -7.89% | -2.30% | -3.43% | -3.56% | -4.07% |
| TH-MultiViewSoft | +1.24% | +0.39% | -0.40% | +0.92% | +0.34% | -0.72% | +0.26% | -0.32% |

### Conversion / CVR Behavior

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TH-FixedBias | 0.0438 | 0.1368 | 0.2068 | 0.0348 | 0.1052 | 0.1597 | 0.0756 | 0.0936 |
| TH-FactorizedScale | 0.0428 | 0.1371 | 0.2046 | 0.0340 | 0.1062 | 0.1586 | 0.0758 | 0.0932 |
| TH-FactorizedSoft | 0.0427 | 0.1358 | 0.2045 | 0.0324 | 0.1054 | 0.1600 | 0.0748 | 0.0926 |
| TH-Factorized | 0.0419 | 0.1354 | 0.2044 | 0.0331 | 0.1052 | 0.1588 | 0.0747 | 0.0924 |
| TH-FixedSoft | 0.0405 | 0.1338 | 0.2048 | 0.0320 | 0.1044 | 0.1588 | 0.0735 | 0.0916 |
| TH-MultiView | 0.0394 | 0.1345 | 0.2018 | 0.0309 | 0.1028 | 0.1556 | 0.0723 | 0.0898 |
| TH-MultiViewSoft | 0.0417 | 0.1354 | 0.2038 | 0.0328 | 0.1036 | 0.1577 | 0.0739 | 0.0918 |

CVR 上 TH-FixedBias 仍最稳。FactorizedScale 在 `HR@5/R@5/N@5` 上略高，FactorizedSoft 在 `R@10` 上略高，但二者在靠前排序质量和/或 `N@10` 上仍弱于 FixedBias。这说明可控 relation bias 可以改善候选覆盖类指标，但还没有提升主要排序质量。

### 当前解释

1. TH-FixedBias 的收益主要不来自 scalar relation-bias 数值本身。当前 FixedBias 是 frozen zero scalar bias，因此更准确的解释是 replacement-style TH attention、behavior Q/K/V、gating 和 behavior-aware MoE 带来的结构收益。
2. Factorized 和 FactorizedScale 与 FixedBias 接近，说明可学习 relation-bias 是可行扩展，但当前结果不足以证明它稳定优于 TH Base。
3. Soft prior 没有带来稳定收益。FixedSoft/FactorizedSoft 均未超过 FixedBias；相比之下，scale control 比 soft-prior initialization 更有价值。
4. Hard MultiView 明显落后，但 Soft MultiView 挽回了大量损失，并在 merged behavior 上很强。这支持 soft view constraint，而不是 hard head partition。

### 对方法主线的影响

当前更稳妥的论文主线应改为：

```text
TH Base / FixedBias
```

而不是把可学习 relation bias 作为最终主模型。FactorizedScale 是当前最强 relation-bias 扩展，但收益集中在部分覆盖类指标，应作为重要 extension/ablation。

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

现有结果已经支持：

- replacement-style TH attention 优于旧 added cross-level attention GAMER。
- 显式行为层级 Q/K/V 建模对 conversion 和 merged behavior 都有效。
- controlled relation bias 可以提升覆盖类指标，但还没有全面改善排序质量。
- Soft MultiView 优于 Hard MultiView，并且在 merged behavior 上有竞争力。

仍需实验才能支持：

- 可学习 relation bias 一定优于 TH Base。
- soft hierarchy prior 一定有效。
- controlled relation bias 或 gated view mixture 可以稳定超过 TH Base。
- gated MultiView 优于 hard MultiView。
- TH-aware auxiliary objective 和 sequence augmentation 带来额外收益。

## 当前结论

当前模型侧主线仍建议围绕：

```text
TH Base
+ controlled learnable Factorized Relation Bias
```

最新 ShortVideoAD 结果显示 `FixedSoft`、`Factorized`、`FactorizedScale`、`FactorizedSoft`、`MultiView` 和 `MultiViewSoft` 均未在 CVR 目标行为上稳定超过 TH-FixedBias，因此当前最终模型应优先以 `TH Base / FixedBias` 为主线。`FactorizedScale` 和 `MultiViewSoft` 相比未缩放/硬划分版本更值得保留；`FactorizedAlpha` 和 `Gated MultiView` 仍是最有信息量的待测模型侧实验。

数据侧不建议继续仅依赖所有用户统一的随机 ratio schedule。更符合 TH 设计的增强方向是：

```text
time-aware
+ session-aware
+ behavior-level-aware
+ optionally user-adaptive
```

最终应让模型结构、辅助目标和序列增强共同围绕同一个二维时序-层级视角，而不是各自独立增加复杂度。
