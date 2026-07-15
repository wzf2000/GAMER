# 生成式时序-层级行为建模

本文档是 `2026-06-01-temporal-hierarchical-behavior-modeling.md` 的中文对应版本，并补充了后续实现和 profiling 后得到的修正认识。

## 背景

下一版 GAMER 的方法定位从一般的多层级行为建模，调整为：

```text
Generative Temporal-Hierarchical Behavioral Modeling for Recommendation
```

核心建模对象不再是一条一维行为序列，而是用户交互序列中的二维结构：

- 时序维度：不同 item/session 之间的交互顺序。
- 层级维度：行为从浅层意图到强偏好的深度，例如 PXS/impression、click、activation、payment/conversion。

当前 session-wise GAMER 输入仍然采用每个 item 一个行为 token 的形式，该 token 表示该 item 行为链中观测到的最深层行为。这种表示紧凑，但也要求模型从同一个 token 流里同时恢复时序转移和行为层级转移。

## 一维建模方案的问题

多行为推荐常见方案可以概括为三类：

1. 将多行为按时间顺序完全铺平：

```text
pxs -> click -> pxs -> click -> activation -> payment -> ...
```

这种方式保留全局时间顺序，但会把行为语义混在一条路径里，高层行为稀疏性和层级依赖关系不容易被显式建模。

2. 只保留每个 item 的最深层行为：

```text
click -> payment -> pxs -> click -> ...
```

这种方式压缩序列长度，适合生成式推荐，但中间行为链证据被隐式化。

3. 每个行为层级各建一条序列：

```text
pxs:        i1 -> i2 -> i5 -> ...
click:      i1 -> i3 -> ...
activation: i3 -> ...
payment:    i4 -> ...
```

这种方式更适合每个层级内部的 next item prediction，但跨层级的原始时间交错关系会被削弱。

目标方法应该统一这些视角：保留因果时序，显式感知行为层级，并允许不同层级之间自由协作，而不是固定成单一硬方向。

## 核心设计目标

模型目标可以写成：

```text
p(x_{t+1}, b_{t+1} | x_{\le t}, b_{\le t})
```

它需要满足：

- 严格遵守因果性，不能看到未来交互或未来语义 token。
- 普通 next item prediction 是它的特例。
- 每个行为层级内部的 next item prediction 是它的特例。
- action/CTR/CVR 风格的层级跃迁预测是它的特例。
- 模型可以学习 same-level、lower-to-higher、higher-to-lower、mixed-level 证据分别在什么时候有用。
- 行为层级顺序通过 embedding、bias、routing 或目标函数显式进入模型，而不只是二值 mask。

## 关键设计选择

### 输入表示

主实验建议继续使用当前最深层行为序列，也就是把每个行为 token 解释为“行为链状态”。

例如：

```text
pxs -> click -> activation -> payment
```

在输入中仍表示为：

```text
payment
```

但论文叙事中应强调：这不是简单丢弃低层行为，而是用最深行为表示该 item 当前达到的行为链状态。

这样做的原因：

- 保持当前训练和评测协议稳定。
- 避免方法贡献和输入格式变化混在一起。
- 避免把 item 内部垂直行为链和 item 间时序转移都铺平成一条事件流。
- 论文主张更清楚：模型在紧凑行为链状态序列上学习时序-层级关系。

完整行为链展开可以作为后续扩展或消融，而不建议作为第一主方法。

### Attention 结构

原始 Qwen3Multi/GAMER 更接近：

```text
self-attention -> cross-level attention -> MoE/FFN
```

这种结构方便实现，但相比标准 Transformer 层额外增加了 attention 容量。作为最终论文方法时，性能提升可能同时来自额外 attention 层和层级归纳偏置，归因不够干净。

推荐的最终主方法是 replacement-style Temporal-Hierarchical Attention：

```text
temporal-hierarchical attention -> MoE/FFN
```

也就是选定部分 Transformer 层，将普通因果 self-attention 替换成时序-层级 attention，而不是额外增加一层 cross attention。

推荐层分配：

```text
Layer 0-1: 局部 token/item 表示。
Layer 2-5: 时序-层级关系建模。
Layer 6-7: 目标层级生成和语义 ID 解码 refinement。
```

当前配置对应：

```json
"sparse_layers_decoder": [0, 1, 2, 3, 4, 5, 6, 7],
"behavior_injection_decoder": [0, 1, 2, 3, 4, 5],
"temporal_hierarchical_attention_decoder": [2, 3, 4, 5]
```

## 方法家族

### 1. TH Attention 基础结构

当前 Temporal-Hierarchical Attention 的基础结构包括：

- 行为层级 Q/K/V embedding。
- attention output gating。
- 仍保留 causal mask 和 item 内 semantic token 的局部因果约束。
- 继续结合 MoE/behavior injection FFN。

这意味着即使 relation bias 是零，模型也不是普通 Qwen3。它已经可以通过行为层级 embedding 让不同层级 token 进入不同的 attention 子空间。

### 2. FixedBias / FixedZero

`Qwen3TemporalHierarchicalFixedBias` 当前配置是：

```json
"th_relation_bias_type": "table",
"th_relation_bias_trainable": false,
"th_relation_bias_init": "zero"
```

因此它的 scalar relation bias 是全零且不可训练。这个版本真正有效的部分不是 fixed scalar bias，而是 TH attention 的 Q/K/V 行为层级 embedding、gating 和 behavior-aware MoE。

更准确的方法解释是：

```text
TH Attention w/o scalar relation bias
```

或：

```text
TH Embedding-only
```

### 3. Factorized Relation Bias

`Qwen3TemporalHierarchicalFactorized` 在 TH attention 基础上加入可学习低秩 relation bias：

```json
"th_relation_bias_type": "factorized",
"th_relation_bias_rank": 4,
"th_relation_bias_trainable": true
```

它不直接学习完整表：

```text
level_pair_bias[q_level, k_level, head]
```

而是学习两个低秩 factor：

```text
query_factor[level, head, rank]
key_factor[level, head, rank]
```

行为层级 pair bias 由二者点积得到：

```text
bias(q_level, k_level, head)
  = sum_r query_factor[q_level, head, r]
        * key_factor[k_level, head, r]
```

这样保留了可学习 relation modeling，同时避免原始 trainable table 在长序列上的极慢 backward。

### 4. MultiView

`Qwen3TemporalHierarchicalMultiView` 将 attention head 分成多个视角：

- temporal：普通因果时序视角。
- same：同层级视角。
- up：低层级历史到高层级 query。
- down：高层级历史到低层级 query。

MultiView 的优点是结构清楚、可解释性强；缺点是 hard mask 限制更强，不如 relation-bias 系列自由。

### 5. Soft Prior 版本

已补充两个 soft init 配置：

- `Qwen3TemporalHierarchicalFixedSoft`
- `Qwen3TemporalHierarchicalFactorizedSoft`

它们使用：

```json
"th_relation_bias_init": "soft",
"th_relation_bias_soft_scale": 0.05
```

soft prior 的含义是：对低层级 query 关注高层级 key 施加负 bias，使信息流更偏向“浅层行为证据支持深层行为预测”。这是一种方向性层级先验，需要实验验证是否会损失高层级行为反向辅助低层级行为的能力。

### 6. Auxiliary Objectives 和 Relation Regularization

当前已实现两个 opt-in 训练增强项，默认均关闭，不影响已有模型配置：

1. Next behavior-level prediction：

```text
L = L_next_token + lambda_level * L_next_level
```

实现方式是在“下一个 token 是 behavior token”的位置，用当前位置 hidden state 预测下一个行为层级。该目标补充了当前 decoder 训练中的监督空缺：behavior token 会作为上下文输入，但在主 LM labels 中被 mask 掉。

2. Relation regularization：

```text
L_relation = MSE(relation_bias, relation_prior)
```

当前 prior 支持 `soft` 或 `zero`，第一批实验使用与 FixedSoft/FactorizedSoft 一致的 soft hierarchy prior。该正则只在 relation-bias 参数可训练时生效，因此 frozen fixed-table 配置不会改变原有行为。

第一批 ShortVideoAD test-set 结果：

- Relation regularization 是更明确的 objective-side 方向。`FactorizedSoftReg` 相对 `FactorizedSoft` 的 CVR 指标全部提升，其中 `HR@5 +2.41%`、`HR@10 +1.29%`、`N@5 +1.71%`、`N@10 +1.30%`，但 merged behavior 略有下降。
- 当前 LevelAux 的结果有正有负。在 MultiViewSoft 上，它提升 CVR `HR@5/R@5` 和多数 merged 指标，但降低 CVR `R@10/N@10`；叠加在 RelationReg 上也表现为 merged 改善、CVR recall/NDCG 相对 RelationReg 单独使用下降。
- 当前优先级是对 RelationReg 做多随机种子和权重验证。LevelAux 暂时保留为消融，除非更低权重或面向高层/target-aware transition 的目标能够消除 CVR 冲突。

## Profiling 修正

最初的 trainable table 版本虽然概念简单，但 profiling 显示不可用于长序列训练：

```text
table_trainable: ~12917 ms/step
table_fixed:       ~108 ms/step
factorized:        ~121 ms/step
multi_view:         ~97 ms/step
```

瓶颈在于：

```python
level_pair_bias[query_level, key_level]
```

会把一个很小的参数表展开成 `[batch, heads, seq_len, seq_len]`，反向传播时要把巨大 dense bias 的梯度 scatter-add 回小表，导致极慢 backward。

因此：

- 不建议将 naive trainable table 作为正式实验配置。
- fixed table 可以作为高效先验或 no-scalar-bias baseline。
- factorized 是当前保留可学习 relation-bias 的主要实现。

## 当前配置映射

- `Qwen3TemporalHierarchicalFixedBias`: zero fixed table，更准确是 TH embedding-only。
- `Qwen3TemporalHierarchicalFactorized`: learnable low-rank relation bias。
- `Qwen3TemporalHierarchicalMultiView`: head-partitioned multi-view hard mask。
- `Qwen3TemporalHierarchicalFixedSoft`: fixed soft hierarchy prior，scale=0.05。
- `Qwen3TemporalHierarchicalFactorizedSoft`: learnable factorized relation bias with soft prior init，scale=0.05。
- `Qwen3TemporalHierarchicalMultiViewSoftLevelAux`: Soft MultiView + next behavior-level auxiliary loss。
- `Qwen3TemporalHierarchicalFixedSoftLevelAux`: FixedSoft + next behavior-level auxiliary loss。
- `Qwen3TemporalHierarchicalFactorizedSoftReg`: FactorizedSoft + soft-prior relation regularization。
- `Qwen3TemporalHierarchicalFactorizedSoftLevelAuxReg`: FactorizedSoft + next behavior-level auxiliary loss + soft-prior relation regularization。
- `Qwen3TemporalHierarchical`: 保留为兼容入口，当前等价于 factorized zero-init。

## 后续开放问题

- zero fixed bias 版本很强，说明 TH 基础结构已经有效；是否需要 scalar relation bias 取决于 soft/factorized 进一步实验。
- factorized 是否需要更好的初始化、rank、relation regularization 或学习率策略。
- MultiView 的 hard partition 是否应改成 soft gating 或可学习 view fusion。
- level auxiliary loss 是否能强化层级感知且不损伤 item generation。
- 是否要对不同目标行为层级分别选择不同 TH 策略。
