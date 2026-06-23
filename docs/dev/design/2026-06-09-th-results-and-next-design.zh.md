# TH 实验结果分析与后续设计建议

## 背景

当前已经完成 ShortVideoAD `smb_explicit_decoder_4` 上已实现 TH 结构变体的测试：

- `Qwen3TemporalHierarchicalFixedBias`
- `Qwen3TemporalHierarchicalFactorized`
- `Qwen3TemporalHierarchicalFactorizedScale`
- `Qwen3TemporalHierarchicalFactorizedSoft`
- `Qwen3TemporalHierarchicalFixedSoft`
- `Qwen3TemporalHierarchicalMultiView`
- `Qwen3TemporalHierarchicalMultiViewSoft`

本文档总结当前结果，并给出如果最终模型希望突出 Temporal-Hierarchical 特色，应该基于哪个方向继续改进。

## 当前结果概览

### Conversion / CVR 目标行为结果

旧 `GAMER (SID)` 与三个 TH 版本对比：

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Old GAMER SID | 0.0394 | 0.1280 | 0.1944 | 0.0292 | 0.0966 | 0.1478 | 0.0687 | 0.0856 |
| FixedBias | **0.0438** | **0.1368** | **0.2068** | **0.0348** | **0.1052** | **0.1597** | **0.0756** | **0.0936** |
| Factorized | 0.0419 | 0.1354 | 0.2044 | 0.0331 | 0.1052 | 0.1588 | 0.0747 | 0.0924 |
| FactorizedScale | 0.0428 | **0.1371** | 0.2046 | 0.0340 | **0.1062** | 0.1586 | **0.0758** | 0.0932 |
| FactorizedSoft | 0.0427 | 0.1358 | 0.2045 | 0.0324 | 0.1054 | **0.1600** | 0.0748 | 0.0926 |
| FixedSoft | 0.0405 | 0.1338 | 0.2048 | 0.0320 | 0.1044 | 0.1588 | 0.0735 | 0.0916 |
| MultiView | 0.0394 | 0.1345 | 0.2018 | 0.0309 | 0.1028 | 0.1556 | 0.0723 | 0.0898 |
| MultiViewSoft | 0.0417 | 0.1354 | 0.2038 | 0.0328 | 0.1036 | 0.1577 | 0.0739 | 0.0918 |

观察：

- 已测试的 TH 结构变体在大多数 CVR 指标上都超过旧 GAMER SID。
- FixedBias 仍是目标行为上最稳的模型，尤其在靠前排序质量（`HR@1`、`R@1`、`N@10`）上更强。
- FactorizedScale 在 `HR@5`、`R@5` 和 `N@5` 上是最强 relation-bias 扩展，说明控制 relation-bias 强度比单纯 soft prior 更值得保留。
- FactorizedSoft 只在 `R@10` 上最好，更像是扩大候选覆盖，而不是提升靠前排序。
- Soft MultiView 明显优于 Hard MultiView，但在 CVR 目标行为上仍弱于 FixedBias。

### Merged behavior-specific 结果

旧 `GAMER (SID)` 与三个 TH 版本对比：

| Model | HR@5 | HR@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|
| Old GAMER SID | 0.1443 | 0.2129 | 0.0621 | 0.0753 |
| FixedBias | 0.1502 | **0.2227** | 0.0656 | **0.0799** |
| Factorized | 0.1500 | 0.2220 | 0.0655 | 0.0796 |
| FactorizedScale | 0.1502 | 0.2221 | 0.0654 | 0.0795 |
| FactorizedSoft | 0.1494 | 0.2199 | 0.0649 | 0.0787 |
| FixedSoft | 0.1497 | 0.2216 | 0.0652 | 0.0795 |
| MultiView | 0.1478 | 0.2162 | 0.0632 | 0.0766 |
| MultiViewSoft | **0.1508** | 0.2218 | **0.0657** | 0.0796 |

观察：

- 大多数 TH 变体在 merged task 上都超过旧 GAMER SID。
- MultiViewSoft 在 merged `HR@5` 和 `N@5` 上最好，说明 soft view penalty 明显优于 hard head partition。
- FixedBias 仍在 merged `HR@10` 和 `N@10` 上最好，因此仍是最稳妥的整体排序模型。
- Factorized 和 FactorizedScale 与 FixedBias 非常接近；relation-bias control 改变了指标取舍，但没有形成稳定全面优势。

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

当前略弱于 FixedBias，同时 FactorizedScale 在部分 CVR 覆盖指标上更强，可能说明：

- TH Q/K/V embedding 已经足够强，额外 logit bias 边际收益有限。
- zero-init factorized bias 需要更好的初始化或正则。
- 直接作用在 attention logits 上的 bias 可能会轻微扰动已经有效的 attention 分布。
- bias 强度很关键；固定 scale 比当前 soft prior 初始化更有效。
- rank=4 不一定是最优。

### MultiView 代表什么

MultiView 将 head 分成：

- temporal
- same
- up
- down

它是最有可解释性的结构化版本，能直接对应“时序 + 行为层级二维图”的几个局部视角。

但它使用 hard mask，限制了不同 head 的可见范围，因此自由度低于 relation-bias 系列。当前结果显示 Hard MultiView 有效但明显偏弱。

Soft MultiView 对这个问题有一定缓解。它在 merged `HR@5/N@5` 上最好，但在 CVR 目标行为上仍弱于 FixedBias。因此 MultiView 更适合作为有解释性的结构化对照，而当前 soft 版本还不足以替代 TH Base 成为最终主模型。

## 如果最终模型要强调 TH 特色，建议基于哪个继续改

如果最终模型既要突出 TH 特色又要忠实于当前结果，建议以 `FixedBias / TH Base` 作为默认主模型，并将可控 relation bias 作为扩展。

理由：

- FixedBias 在目标 CVR 行为和 merged 高 rank 指标上仍最稳。
- 它的 scalar table 为零，因此当前证据支持的核心主张应是 behavior-aware replacement TH attention，而不是非零 scalar relation bias。
- FactorizedScale 是当前最好的 relation-bias 扩展，应保留为主要 relation-bias 消融/候选，但还没有稳定超过 TH Base。
- MultiViewSoft 说明 soft view constraint 优于 hard mask，但 CVR 结果仍弱于 FixedBias。

推荐论文定位：

```text
主模型：TH Base / FixedBias
Relation-bias 扩展：带 scale control 的 Factorized Temporal-Hierarchical Relation Bias
结构化消融：Soft/Hard Multi-View Temporal-Hierarchical Attention
```

论文仍可将 relation control 描述为 TH-aware enhancement，但最新指标不支持把它作为唯一核心贡献。

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

1. Factorized relation bias 的 learnable alpha，并记录每层 alpha。
2. Factorized rank 消融：1/2/4/8，优先在 scaled setting 下做。
3. Gated MultiView，因为 Soft MultiView 已经在 merged 指标上显著优于 Hard MultiView。
4. 行为层级辅助目标。
5. attention/bias 可视化诊断。
6. 在模型侧 baseline 固定后再系统测试 sequence augmentation。

## 当前建议结论

如果只看现有指标，FixedBias / TH Base 仍是最稳妥的最终模型。

如果最终设计要强调 Temporal-Hierarchical modeling，目前最有支撑的说法应是：

```text
Temporal-Hierarchical Attention
+ behavior-aware Q/K/V
+ attention gating
+ optional controlled relation/view bias
```

除非后续 FactorizedAlpha 或 Gated MultiView 明显超过 FixedBias，否则 FixedBias 应作为主模型。FactorizedScale 是当前最值得保留的 relation-bias 扩展；MultiViewSoft 是有价值的结构化视角消融，说明 soft constraint 比 hard partition 更合理。
