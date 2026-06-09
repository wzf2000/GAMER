# TH 实验结果分析与后续设计建议

## 背景

当前已经完成 ShortVideoAD `smb_explicit_decoder_4` 上三个主要 TH 变体的测试：

- `Qwen3TemporalHierarchicalFixedBias`
- `Qwen3TemporalHierarchicalFactorized`
- `Qwen3TemporalHierarchicalMultiView`

另外已新增两个待测配置：

- `Qwen3TemporalHierarchicalFixedSoft`
- `Qwen3TemporalHierarchicalFactorizedSoft`

本文档总结当前结果，并给出如果最终模型希望突出 Temporal-Hierarchical 特色，应该基于哪个方向继续改进。

## 当前结果概览

### Conversion / CVR 目标行为结果

旧 `GAMER (SID)` 与三个 TH 版本对比：

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Old GAMER SID | 0.0394 | 0.1280 | 0.1944 | 0.0292 | 0.0966 | 0.1478 | 0.0687 | 0.0856 |
| FixedBias | **0.0438** | **0.1368** | **0.2068** | **0.0348** | **0.1052** | **0.1597** | **0.0756** | **0.0936** |
| Factorized | 0.0419 | 0.1354 | 0.2044 | 0.0331 | 0.1052 | 0.1588 | 0.0747 | 0.0924 |
| MultiView | 0.0394 | 0.1345 | 0.2018 | 0.0309 | 0.1028 | 0.1556 | 0.0723 | 0.0898 |

观察：

- 三个 TH 版本均超过旧 GAMER SID。
- FixedBias 最强，Factorized 很接近，MultiView 稍弱但仍有效。
- CVR 上 FixedBias 相比旧 GAMER SID 的提升约为 `+6.4%` 到 `+19.3%`。
- Factorized 保留可学习 relation bias，但当前结果略弱于 FixedBias。

### Merged behavior-specific 结果

旧 `GAMER (SID)` 与三个 TH 版本对比：

| Model | HR@5 | HR@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|
| Old GAMER SID | 0.1443 | 0.2129 | 0.0621 | 0.0753 |
| FixedBias | **0.1502** | **0.2227** | **0.0656** | **0.0799** |
| Factorized | 0.1500 | 0.2220 | 0.0655 | 0.0796 |
| MultiView | 0.1478 | 0.2162 | 0.0632 | 0.0766 |

观察：

- 三个 TH 版本在 merged task 上也都超过旧 GAMER SID。
- FixedBias 和 Factorized 几乎持平，说明 relation-bias 系列整体比 MultiView 更稳。
- MultiView 的提升幅度较小，但仍说明 temporal/same/up/down 的多视角分解有价值。

## 对三个方向的解释

### FixedBias 实际代表什么

当前 `FixedBias` 是：

```json
"th_relation_bias_type": "table",
"th_relation_bias_trainable": false,
"th_relation_bias_init": "zero"
```

因此它的 scalar relation bias 是全零，不提供显式层级方向 bias。

它强的原因更可能是：

- TH attention 替换普通 self-attention。
- Q/K/V 中注入行为层级 embedding。
- attention output gating。
- 与 behavior-aware MoE/FFN 共同作用。

所以 FixedBias 更准确地说是：

```text
TH Attention without scalar relation bias
```

或：

```text
TH Embedding-only
```

它是非常强的基础版本，但如果论文主方法要强调“可学习层级 relation bias”，它的叙事不够完整。

### Factorized 代表什么

Factorized 在 FixedBias 的基础结构上，加入可学习低秩行为层级 pair bias：

```text
bias(q_level, k_level, head)
  = query_factor[q_level, head] · key_factor[k_level, head]
```

它保留了 learnable temporal-hierarchical relation modeling 的方法卖点，并且 avoids naive trainable table 的极慢 backward。

当前略弱于 FixedBias，可能说明：

- TH Q/K/V embedding 已经足够强，额外 logit bias 边际收益有限。
- zero-init factorized bias 需要更好的初始化或正则。
- 直接作用在 attention logits 上的 bias 可能会轻微扰动已经有效的 attention 分布。
- rank=4 不一定是最优。

### MultiView 代表什么

MultiView 将 head 分成：

- temporal
- same
- up
- down

它是最有可解释性的结构化版本，能直接对应“时序 + 行为层级二维图”的几个局部视角。

但它使用 hard mask，限制了不同 head 的可见范围，因此自由度低于 relation-bias 系列。当前结果显示它有效但不最强，适合作为重要 ablation 或补充模块，而不是当前最优主模型。

## 如果最终模型要强调 TH 特色，建议基于哪个继续改

建议以 `Factorized` 为主线继续改进，而不是直接以 FixedBias 为最终主模型。

理由：

- FixedBias 当前性能最好，但它的 scalar bias 为 zero，论文里很难声称它利用了显式层级 relation bias。
- Factorized 性能只略低于 FixedBias，同时保留“可学习时序-层级关系”的核心方法叙事。
- Factorized 比 MultiView 更强，说明 soft/continuous relation modeling 比 hard view partition 更适合作为主路径。
- FixedBias 可以作为 strong TH-base 或 no-relation-bias ablation，用来证明 TH attention 基础结构本身有效。

推荐论文定位：

```text
主模型：Factorized Temporal-Hierarchical Relation Bias
强消融：TH Attention w/o scalar relation bias (FixedBias/FixedZero)
结构化消融：Multi-View Temporal-Hierarchical Attention
```

如果后续 `FactorizedSoft` 超过或接近 FixedBias，则主模型叙事会更稳：

```text
learnable factorized TH relation bias initialized with a weak hierarchy prior
```

## 后续改进方向

### P1. 测试 soft prior 初始化

已新增：

- `Qwen3TemporalHierarchicalFixedSoft`
- `Qwen3TemporalHierarchicalFactorizedSoft`

建议优先跑这两个。scale 当前设为 `0.05`，比 `0.1` 更温和。

要验证的问题：

- FixedSoft 是否优于 FixedZero。
- FactorizedSoft 是否优于 FactorizedZero。
- soft prior 是否帮助 CVR，但损伤 click/p3s 或 merged behavior。

如果 FixedSoft 强于 FixedZero，说明显式层级方向先验有用。

如果 FactorizedSoft 强于 FactorizedZero，说明 learnable relation bias 需要更合理的先验初始化。

### P2. Factorized rank 消融

当前 rank=4。建议测试：

```text
rank = 1, 2, 4, 8
```

预期：

- rank 太低：表达不足。
- rank 太高：可能过拟合或扰动 attention。
- 如果 rank=1/2 表现接近 rank=4，说明层级 relation 很低秩，论文叙事更简洁。

### P3. Relation bias scale 与正则

Factorized 直接加到 attention logits 上，影响较强。可以考虑：

```text
score = qk / sqrt(d) + alpha * relation_bias
```

其中 `alpha` 可为：

- 固定小值，如 `0.1`、`0.3`。
- 可学习标量，初始化为 0 或小值。
- layer-specific alpha。

这样可以避免 relation bias 训练早期过度扰动基础 TH attention。

建议配置：

```json
"th_relation_bias_scale": 0.1
```

或：

```json
"th_relation_bias_learnable_scale": true
```

### P4. 将 MultiView 从 hard partition 改成 soft/gated view

当前 MultiView 每个 head 固定属于一个 view，约束较硬。

可以改为：

```text
每个 head 对 temporal/same/up/down 有可学习混合权重
```

或者：

```text
query-level gate: 根据当前 query 行为层级动态选择 view 权重
```

这样保留 MultiView 的可解释性，同时减少 hard mask 带来的自由度损失。

建议作为中期方向，而不是立刻替换 Factorized 主线。

### P5. Relation bias 的层共享/层特异性

当前每个 TH layer 有自己的 relation bias 参数。

可尝试：

- 全 TH layers 共享一套 relation factors。
- 低层/中层/高层分组共享。
- 每层独立但加相似性正则。

如果共享后性能不降，可以说明 relation pattern 是稳定的行为层级规律；如果独立更强，则说明不同深度层学习不同层级关系。

### P6. 加行为层级辅助目标

可以增加轻量辅助任务：

```text
L = L_next_token + lambda_level * L_next_behavior_level
```

只在 behavior token 位置预测下一个行为层级，lambda 设小，例如 `0.05` 或 `0.1`。

目的：

- 强化模型对层级转移的显式理解。
- 帮助 Factorized relation bias 学到更稳定的层级关系。

风险：

- 如果 lambda 太大，可能伤害 item semantic ID generation。

### P7. Attention diagnostics

建议增加诊断脚本或评估输出：

- same / up / down / mixed relation 的 attention mass。
- 不同 target behavior 上 relation mass 的差异。
- Factorized bias 学到的 level-pair matrix 可视化。
- FixedSoft 与 FactorizedSoft 的 learned bias 是否偏离初始方向。

这对论文解释非常有帮助，尤其是证明模型确实学到了 TH 关系，而不是只靠额外 embedding。

## 推荐实验优先级

1. `FixedSoft` 和 `FactorizedSoft`，scale=0.05。
2. Factorized rank 消融：1/2/4/8。
3. Relation bias scale 或 learnable alpha。
4. MultiView soft/gated view。
5. 行为层级辅助目标。
6. attention/bias 可视化诊断。

## 当前建议结论

如果只看现有指标，FixedBias 是当前最强。

但如果最终模型设计希望强调 TH 特色，建议主线继续基于 Factorized：

```text
Temporal-Hierarchical Attention
+ behavior-aware Q/K/V
+ learnable factorized relation bias
```

FixedBias 应该作为强基础消融，说明 TH attention 本身有效；MultiView 应作为结构化视角消融，说明二维关系分解有效但 hard partition 稍弱；Factorized/FactorizedSoft 最适合作为最终主方法候选。
