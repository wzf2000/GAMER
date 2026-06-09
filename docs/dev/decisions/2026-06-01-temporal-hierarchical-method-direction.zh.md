# 时序-层级方法方向

## 状态

已接受，用于下一版 GAMER 的方法规划。

## 背景

下一版 GAMER 被重新表述为：

```text
Generative Temporal-Hierarchical Behavioral Modeling for Recommendation
```

这意味着主要建模重点从一般 cross-level 行为交互，转向用户行为序列的二维视角：

- 时序维度：item 与 session 之间的交互顺序。
- 层级维度：行为从浅层意图到强偏好的深度。

有三个早期设计选择会影响实现路径、实验设置和论文叙事：

- 主输入继续使用最深层行为序列，还是改成铺平的行为事件序列。
- 时序-层级建模作为第二个 attention 模块加入，还是替换部分 self-attention。
- behavior injection、temporal-hierarchical attention 和 MoE 层应该如何分配。

详细方法设计见：

```text
docs/dev/design/2026-06-01-temporal-hierarchical-behavior-modeling.md
docs/dev/design/2026-06-01-temporal-hierarchical-behavior-modeling.zh.md
```

## 决策

主方法输入继续使用当前最深层行为序列。每个行为 token 被解释为行为链的最终状态，而不是孤立的单一行为标签。

铺平的行为事件序列作为 ablation 或 baseline，不作为默认主方法输入。

短期原型可以先在现有 Qwen3Multi 双 attention 结构中实现时序-层级 relation modeling：

```text
self-attention -> temporal-hierarchical cross-attention -> MoE/FFN
```

论文主方法应逐步转向 replacement-style Temporal-Hierarchical Attention：

```text
temporal-hierarchical attention -> MoE/FFN
```

该 replacement attention 应保留普通因果 next-token 建模，并通过 relation bias 或 view-specific attention heads 引入层级信息。

对于 8 层模型，优先采用：

```text
Layer 0-1: standard causal attention + behavior-injected MoE
Layer 2-5: temporal-hierarchical attention + behavior-injected MoE
Layer 6-7: standard causal attention + MoE
```

默认保留所有层的 MoE。behavior injection 至少覆盖所有 temporal-hierarchical attention 层。

## 理由

最深层行为序列保持当前训练与评测协议稳定，同时支持行为链状态解释，避免把收益和大规模输入格式变化混在一起。

铺平所有行为事件虽然保留原始时序，但会混合 item 内部行为进展和 item 间时间转移，使时序-层级方法贡献不够清晰。

现有双 attention 结构便于快速实现，因为代码已有 cross-level attention hook。但作为最终主方法，它比标准 Transformer 层多了 attention 容量。replacement-style attention 更公平，因为总 attention 层数保持不变。

中间层适合放置 temporal-hierarchical attention，因为它能在最终生成前塑造序列表示；最后几层可更专注于目标条件生成和语义 ID 解码。

## 影响

第一版实现可以通过扩展现有 cross-level attention 模块来降低风险。

论文主方法最终应报告公平的 replacement-style 版本，而不只是 added-attention 版本。

实验应包含输入格式消融：

- 最深层行为 / 行为链状态序列。
- 铺平行为事件序列。
- 如果实现，加入可选行为链展开序列。

实验也应包含架构公平性消融：

- 普通 Qwen3 baseline。
- 当前 added cross-attention Qwen3Multi。
- attention 层数相同的 replacement-style temporal-hierarchical attention。
