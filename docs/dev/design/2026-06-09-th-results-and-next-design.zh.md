# TH 实验结果分析与后续设计建议

## 背景

当前已经完成 ShortVideoAD `smb_explicit_decoder_4` 上已实现 TH 结构变体的测试。本文档现在统一使用 `smb_explicit` test-set 结果，不再使用此前的验证集对比。

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

旧 `GAMER (SID)` 与 TH 版本对比：

| Model | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Old GAMER SID | 0.0394 | 0.1280 | 0.1944 | 0.0292 | 0.0966 | 0.1478 | 0.0687 | 0.0856 |
| FixedBias | 0.0390 | 0.1283 | 0.1974 | 0.0290 | 0.0963 | 0.1507 | 0.0693 | 0.0873 |
| FixedSoft | **0.0435** | **0.1349** | 0.1981 | 0.0326 | 0.1007 | 0.1513 | **0.0735** | 0.0900 |
| Factorized | 0.0409 | 0.1342 | **0.2042** | 0.0302 | **0.1011** | **0.1565** | 0.0721 | **0.0902** |
| FactorizedScale | 0.0393 | 0.1331 | 0.1987 | 0.0301 | 0.0988 | 0.1514 | 0.0706 | 0.0877 |
| FactorizedSoft | 0.0385 | 0.1274 | 0.1947 | 0.0294 | 0.0972 | 0.1503 | 0.0690 | 0.0867 |
| MultiView | 0.0381 | 0.1283 | 0.1949 | 0.0275 | 0.0958 | 0.1461 | 0.0678 | 0.0845 |
| MultiViewSoft | 0.0427 | 0.1274 | 0.1958 | **0.0331** | 0.0966 | 0.1504 | 0.0708 | 0.0885 |

观察：

- 修正后的 test-set 对比比此前验证集结论更复杂。
- FixedSoft 在靠前 CVR 指标（`HR@1/HR@5/N@5`）上最强，如果重点是早期排序，它比 FixedBias 更适合作为最终候选。
- Factorized 在更深 CVR 指标（`HR@10/R@5/R@10/N@10`）上最强，是当前最值得保留的 relation-bias 主候选。
- FixedBias 仍是干净的 TH base，但不应再被描述为 CVR 目标行为上的最稳最终模型。
- Soft MultiView 明显优于 Hard MultiView，但作为主模型线仍弱于 FixedSoft/Factorized。

### Merged behavior-specific 结果

旧 `GAMER (SID)` 与 TH 版本对比：

| Model | HR@5 | HR@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|
| Old GAMER SID | 0.1443 | 0.2129 | 0.0621 | 0.0753 |
| FixedBias | 0.1444 | 0.2116 | 0.0620 | 0.0750 |
| FixedSoft | **0.1450** | **0.2121** | **0.0628** | **0.0756** |
| Factorized | 0.1430 | 0.2117 | 0.0614 | 0.0746 |
| FactorizedScale | 0.1432 | 0.2113 | 0.0614 | 0.0745 |
| FactorizedSoft | 0.1434 | 0.2099 | 0.0615 | 0.0744 |
| MultiView | 0.1391 | 0.2062 | 0.0595 | 0.0723 |
| MultiViewSoft | 0.1418 | 0.2102 | 0.0609 | 0.0742 |

观察：

- FixedSoft 在四个 reported merged behavior 指标上都是最强版本。
- FixedBias 和 Factorized 仍然接近，但在 merged test-set 对比下都没有超过 FixedSoft。
- MultiViewSoft 仍明显优于 Hard MultiView，但 soft-view line 已经不是 merged behavior 上最强候选。
- 因此 test-set 排序更支持：fixed soft hierarchy prior 负责 merged behavior，factorized relation bias 负责 CVR depth。

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

当前 Factorized 在更深 CVR 指标上更强、但 merged behavior 更弱，可能说明：

- TH Q/K/V embedding 已经足够强，额外 logit bias 边际收益有限。
- zero-init factorized bias 需要更好的初始化或正则。
- 直接作用在 attention logits 上的 bias 可能会轻微扰动已经有效的 attention 分布。
- bias 强度很关键，但当前固定 scale 版本在 test set 上并没有稳定优于未缩放的 factorized 版本。
- rank=4 不一定是最优。

### MultiView 代表什么

MultiView 将 head 分成：

- temporal
- same
- up
- down

它是最有可解释性的结构化版本，能直接对应“时序 + 行为层级二维图”的几个局部视角。

但它使用 hard mask，限制了不同 head 的可见范围，因此自由度低于 relation-bias 系列。当前结果显示 Hard MultiView 有效但明显偏弱。

Soft MultiView 对这个问题有一定缓解，但修正后的 test-set 结果显示，它在 merged behavior 上弱于 FixedSoft，在 CVR depth 上弱于 Factorized。因此 MultiView 更适合作为有解释性的结构化对照，而当前 soft 版本还不足以替代 relation-bias 系列成为最终主模型。

## 如果最终模型要强调 TH 特色，建议基于哪个继续改

如果最终模型既要突出 TH 特色又要忠实于当前 test-set 结果，建议把 `TH Base` 作为基础消融，并根据论文重点在 `FixedSoft` 和 `Factorized` 中选择最终变体。

理由：

- FixedSoft 是当前最好的 merged behavior 和靠前 CVR 模型。
- Factorized 是当前最好的更深 CVR 排序和覆盖模型。
- FixedBias 的 scalar table 为零，因此仍需要作为干净的 TH base。
- MultiViewSoft 说明 soft view constraint 优于 hard mask，但结果仍弱于最好的 relation-bias/prior 版本。

推荐论文定位：

```text
主模型候选：TH-FixedSoft 和 TH-Factorized
基础消融：TH Base / FixedBias
Relation-bias 扩展：Factorized Temporal-Hierarchical Relation Bias
结构化消融：Soft/Hard Multi-View Temporal-Hierarchical Attention
```

论文可以将 relation control 和 soft hierarchy prior 都描述为 TH-aware enhancement，但最新 test 指标不支持把 Hard MultiView 或 FixedBias-only 作为最终故事。

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

1. 根据目标指标优先级，在 FixedSoft 和 Factorized 之间确定最终 test-set 主变体。
2. Factorized relation bias 的 learnable alpha，并记录每层 alpha。
3. Factorized rank 消融：1/2/4/8，优先在 scaled 或 alpha-controlled setting 下做。
4. Gated MultiView，因为 Soft MultiView 优于 Hard MultiView，但还不够强。
5. 行为层级辅助目标。
6. attention/bias 可视化诊断。
7. 在模型侧 baseline 固定后再系统测试 sequence augmentation。

## 当前建议结论

如果 merged behavior 和靠前 CVR 最重要，FixedSoft 是当前最稳的最终模型。如果 CVR depth/coverage 更重要，Factorized 是当前最强最终候选。

如果最终设计要强调 Temporal-Hierarchical modeling，目前最有支撑的说法应是：

```text
Temporal-Hierarchical Attention
+ behavior-aware Q/K/V
+ attention gating
+ optional controlled relation/view bias
```

FixedBias 应作为基础消融，而不是默认最终主模型。Factorized 是当前最值得保留的 relation-bias 扩展；FixedSoft 是最强 fixed-prior 版本；MultiViewSoft 是有价值的结构化视角消融，说明 soft constraint 比 hard partition 更合理。
