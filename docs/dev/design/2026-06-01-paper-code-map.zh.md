# GAMER 论文到代码映射

## 来源

本文档对应英文版 `2026-06-01-paper-code-map.md`，用于将 GAMER 论文中的方法组件、baseline、训练流程和评测流程映射到当前代码库。

论文版本：

```text
Generative Sequential Recommendation via Hierarchical Behavior Modeling
```

## 论文方法概览

GAMER 面向 session-wise multi-behavior generative recommendation。论文中的核心组件包括：

- 将 item 转换为语义 ID token，并用生成式 decoder 预测目标 item。
- 使用行为 token 表示多行为历史中的行为层级。
- 使用 session-wise 序列构造，让模型在多行为上下文中执行 next item prediction。
- 在 Qwen3 / MoE 风格 backbone 上加入行为感知模块。
- 评测目标包括 target behavior next item prediction 和 behavior-specific next item prediction。

## 高层代码映射

核心代码区域：

- `main.py`: 统一任务入口。
- `SeqRec/tasks/`: 训练、评测、tokenization、analysis 任务。
- `SeqRec/datasets/`: 数据集、loader 和 collator。
- `SeqRec/models/generative/`: 生成式推荐模型和 tokenizer 相关实现。
- `SeqRec/trainers/`: 训练器实现。
- `scripts/`: 训练和评测 shell 工作流。
- `config/s2s-models/`: backbone 和 tokenizer 配置。

当前重构后，Qwen3 相关生成模型主要位于：

```text
SeqRec/models/generative/qwen3/
```

## 方法模块

### Session-Wise 协议

Session-wise 数据构造主要围绕 SMB dataset 系列展开。输入序列中，每个 item 通常由行为 token 和 item semantic ID token 组成。目标行为通过 evaluation prompt 或 behavior-specific 任务指定。

相关模块：

- `SeqRec/datasets/session_behavior/`
- `SeqRec/datasets/loaders/session_behavior.py`
- `SeqRec/datasets/collators/generative.py`
- `SeqRec/tasks/training/train_SMB_decoder.py`
- `SeqRec/tasks/evaluation/test_SMB_decoder.py`

### 多行为序列增强

多行为版本扩展了普通 sequential recommendation，使模型可以利用不同行为层级的历史。行为 token 会被 router 映射到 position/behavior/action indices。

相关模块：

- `SeqRec/models/generative/qwen3/multi_router.py`
- `SeqRec/tasks/training/train_MB_decoder.py`
- `SeqRec/tasks/evaluation/test_MB_decoder.py`

### Qwen3 MoE 与 GAMER 架构

GAMER 主要基于 Qwen3 decoder-only 结构，并加入：

- position-aware / behavior-aware MoE。
- behavior injection。
- cross-level behavior attention 或 replacement-style temporal-hierarchical attention。

重要配置字段：

```json
"sparse_layers_decoder": [...],
"behavior_injection_decoder": [...],
"cross_attention_decoder": [...],
"temporal_hierarchical_attention_decoder": [...]
```

### Cross-Level 行为交互

旧 GAMER / Qwen3Multi 使用：

```text
self-attention -> cross-level attention -> MoE/FFN
```

cross-level attention 通过 action/behavior mask 控制不同行为层级之间的可见性，并在 Q/K/V 中注入行为 embedding。

新的 TH 方向将其改为：

```text
temporal-hierarchical attention -> MoE/FFN
```

即 selected layers 中替换普通 self-attention，而不额外增加 attention 模块。

### Position-and-Behavior-Aware MoE

MoE 模块根据 semantic token position 和 behavior 信息路由或调制 FFN。相关实现位于：

```text
SeqRec/models/generative/qwen3/moe_ffn.py
```

## 训练流程

主训练入口：

- `scripts/train_decoder.sh`
- `scripts/train_MB_decoder.sh`
- `scripts/train_SMB_decoder.sh`

对应 Python task：

- `train_decoder`
- `train_MB_decoder`
- `train_SMB_decoder`

训练流程大致为：

1. 解析 shell env 和 trailing CLI 参数。
2. 根据 `backbone` 解析模型类和 base config。
3. 加载 dataset 和 tokenizer。
4. 根据数据集新增 item / behavior token。
5. 配置 position、behavior、MoE 和 TH 参数。
6. 构造 Hugging Face `TrainingArguments` 和 trainer。
7. 保存 checkpoint、tokenizer 和 config。

## 评测流程

主评测入口：

- `scripts/test_decoder.sh`
- `scripts/test_MB_decoder.sh`
- `scripts/test_SMB_decoder.sh`

评测一般通过 beam search 生成 item semantic ID，并使用 trie 或候选约束保证生成有效 item。指标包括：

- HR@K
- Recall@K
- NDCG@K
- per-behavior metrics
- merged behavior metrics

## Baseline 映射

论文表中的 baseline 大致映射为：

- Rule-Based: rule evaluation task。
- GRU4Rec / SASRec / BERT4Rec: discriminative sequential recommendation baselines。
- PBAT / MBHT / MB-STR: 多行为 sequential baselines。
- TIGER / MBGen / GAMER: 生成式推荐相关方法。

部分 graph baseline 可能是外部实现或当前快照未包含，需要继续确认。

## Tokenization 映射

生成式推荐依赖 item semantic ID。当前脚本中通过 tokenization helper 区分：

- CID / RQ tokenization。
- SID / original index。
- `original=1` 时使用 `.index.json`。

相关脚本和文档：

- `scripts/lib/tokenization.sh`
- `docs/scripts.md`
- `docs/datasets.md`

## 复现实验入口

ShortVideoAD 的典型 SMB decoder 训练：

```bash
dataset=ShortVideoAD original=1 gpu=0,1,2,3 batch_size=512 tasks=smb_explicit_decoder_4 backbone=Qwen3Multi bash scripts/train_SMB_decoder.sh
```

TH 变体可将 `backbone` 替换为：

```text
Qwen3TemporalHierarchicalFixedBias
Qwen3TemporalHierarchicalFactorized
Qwen3TemporalHierarchicalMultiView
Qwen3TemporalHierarchicalFixedSoft
Qwen3TemporalHierarchicalFactorizedSoft
```

## 注意事项与开放问题

- 论文中的部分路径在后续重构中已经移动，英文原文中一些旧路径可视为历史映射。
- 当前主线方法从 Qwen3Multi added cross-attention 转向 replacement-style TH attention。
- 需要继续确认 graph baselines 是否在外部仓库。
- 需要对 test split、valid split 和不同 behavior target 的结果保持严格区分。
