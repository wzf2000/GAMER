# 框架复用重构报告

## 范围

本文档对应英文版 `2026-06-03-framework-reuse-refactor-report.md`，总结当前框架中可通过复用降低冗余的部分，以及已经完成和后续建议的重构方向。

关注区域包括：

- dataset loader / collator。
- generative training tasks。
- generative evaluation tasks。
- analysis tasks。
- generative model family。
- backbone metadata registry。
- utils 命名和模块边界。

## 发现与推荐顺序

### 1. Collator 辅助字段处理

不同任务对 `session_ids`、`actions`、`extended_session_ids` 等字段的处理曾存在重复逻辑。重构目标是让 helper 根据 backbone metadata 判断是否需要传递 sessions/actions，减少 task 中硬编码分支。

关键注意事项：

- 对 `BatchEncoding` 中可能被属性赋值修改的字段，应使用 `getattr(inputs, name, None)`。
- 不要在 helper 中使用 `inputs.get(name)` 读取这些被 task 局部修改过的字段。

详见：

```text
docs/dev/notes/2026-06-03-batch-encoding-attr-vs-dict.md
docs/dev/notes/2026-06-03-batch-encoding-attr-vs-dict.zh.md
```

### 2. `train_decoder` 与 `train_MB_decoder` 采用 registry

训练任务中原本有多个 backbone-specific 分支。重构后通过 generative backbone registry 统一解析：

- 是否 decoder-only。
- 是否使用 action/session。
- config/tokenizer 类型。
- model class。

这样新增模型 family 时不需要在每个 task 里重复添加 `if backbone == ...`。

### 3. 通用生成式训练构建器

将训练任务中重复的 parser、config、tokenizer、collator、model、TrainingArguments 和 trainer 构造逻辑抽取到共享 helper/base class。

目标：

- 让 `train_decoder`、`train_MB_decoder`、`train_SMB_decoder` 只保留任务特有的数据加载和少量配置差异。
- 降低新增 generative training task 的成本。

### 4. 通用生成式评测工具

评测任务中重复了模型加载、DDP/gather、beam search、结果保存和 per-user metrics 保存逻辑。重构后建立 `_BaseDecoderTestTask`，将共通逻辑放入 base class。

子类主要负责：

- 数据加载。
- task-specific prompt/target 处理。
- 指标维度选择。

### 5. Dataset Loader Task Registry

dataset loading 的命名和入口原本较分散。重构方向：

- 将 loader 移到 `SeqRec/datasets/loaders/`。
- 使用更明确的 snake_case 模块名。
- 保留必要兼容 wrapper。
- 减少 task 中直接导入旧文件名的情况。

### 6. 生成式模型 family 微复用

Qwen3Multi、Qwen3SessionMulti、Qwen3SessionMoe、Qwen3TemporalHierarchical 等 family 曾有较多重复 decoder loop 和 cache/mask 逻辑。

重构后：

- 抽取 `SeqRec/models/generative/qwen3/_decoder_base.py`。
- 抽取 `SeqRec/models/generative/common/` 下的 cache、decoder_loop、temperature、wrappers、attention、session_masks。
- 让不同模型 family 通过 hook 扩展 layer setup 和 layer kwargs。

## 执行计划

推荐顺序：

1. 先做低风险 helper 和 registry 抽取。
2. 再做 training/evaluation task base class。
3. 最后做 model family base class 和 component 抽取。

原因：

- 任务层重构更容易验证，能用现有 checkpoint 做结果一致性检查。
- 模型层重构风险更高，应在测试路径稳定后进行。

## 2026-06-03 执行记录

已完成的主要重构包括：

- `SeqRec/datasets/`: 拆分 SMB dataset，移动 loaders/collators，清理旧 shim。
- `SeqRec/models/generative/`: 拆分 common mixins，按 qwen3/llama/pba_transformer/tiger family 重组模型。
- `SeqRec/tasks/`: 按 training/evaluation/analysis/tokenization 分类重组，并使用 lazy registry。
- `SeqRec/utils/`: `futils.py` -> `fs.py`，`pipe.py` -> `runtime.py`，`func_util.py` -> `decorators.py`，`parse.py` -> `args.py`。
- evaluation tasks: 建立 rule base、decoder test base，移除多个 hard-coded backbone 分支。
- analysis tasks: 抽取共享 analysis base。
- qwen3 model base: 将多个 Qwen3 decoder model base 合并到 `Qwen3DecoderModelBase`。

验证方式：

- Toy checkpoint 上对比重构前后的 results JSON。
- 对 Qwen3Multi MB、Qwen3SessionMulti SMB、Qwen3SessionMoe SMB、Qwen3TemporalHierarchical SMB 做 byte-identical 或指标一致性检查。

## P4 后续可选重构

### P1. Training Parser 参数分组

将 argparse 中的模型、数据、训练参数明确分组，减少 task invoke 的长参数列表。

### P2. 生成式训练 Profile Setup

将不同 train profile 的 config/tokenizer/model setup 收敛到共享 helper，避免任务内部重复 profile-specific 逻辑。

### P3. Backbone Metadata 单一事实源

让 registry 成为判断 backbone 能力、config 路径、model class、tokenizer 类型的唯一来源。

### P4. Qwen/Llama 多行为组件复用

继续抽取 Qwen 和 Llama 多行为模型之间相似的 router、mask、temperature wrapper 和 generation/session helper。

### P5. Lazy Model-Class Import Registry

进一步降低 import 侧效应，使 task 只在真正需要某个模型时导入对应模块。

## 建议后续顺序

优先继续：

1. Parser / args 结构清理。
2. Generative training profile helper。
3. Backbone registry 完整化。
4. Model family component reuse。
5. Lazy import 和轻量化启动。

这些方向能继续减少新增模型和新增任务时的重复修改点。
