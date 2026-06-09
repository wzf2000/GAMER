# Scripts 维护性重构

## 背景

训练、评测、tokenization 和 analysis 的 shell 脚本原本包含较多重复逻辑。每新增一个 backbone，例如 `Qwen3TemporalHierarchical` 或其变体，通常需要同步修改多个脚本中的分支判断、base model 路径和输出目录规则。

这种方式维护成本高，容易出现脚本之间不一致的问题。

## 设计目标

将 shell 脚本中重复的逻辑抽到 `scripts/lib/`，并把 backbone 元信息集中到 Python registry 中，使新增 backbone 时尽量只需要新增 config 和 registry 元数据，而不是到处改 shell 分支。

目标包括：

- 减少 `train/test` 脚本中的重复条件分支。
- 统一 backbone -> base_model 的解析。
- 统一 tokenization 命名和路径构建。
- 支持 trailing CLI 参数透传，减少 `extra_args="a=b,c=d"` 这种不便用法。
- 保留 legacy `extra_args` / `extra_flags` 兼容。

## 已完成重构批次

### Backbone 解析

新增或使用：

```text
scripts/lib/s2s_backbone.sh
SeqRec/models/generative/registry.py
```

shell 中通过 Python registry 解析：

- task 内部使用的 backbone 名称。
- 对应 base model config 路径。

这使 `Qwen3TemporalHierarchical*` 这类变体可以自动映射到同一个模型类，同时使用各自 config 目录。

### Shell 参数 helper

新增：

```text
scripts/lib/args.sh
```

支持：

```bash
bash scripts/train_SMB_decoder.sh --learning_rate 5e-4 --num_train_epochs 200
```

同时保留旧接口：

```bash
extra_args=max_his_len=100,warmup_ratio=0.04
extra_flags=debug
```

### Tokenization 命名

新增或整理：

```text
scripts/lib/tokenization.sh
```

统一处理：

- CID / RQ tokenization。
- SID / original index。
- `token_tag` 命名。
- `.index.json` 选择。

### 路径构建

新增或整理：

```text
scripts/lib/paths.sh
```

统一 checkpoint、result、task_dir 和 suffix 相关路径构建，避免训练和评测脚本输出路径不一致。

### Runtime helper

新增或整理：

```text
scripts/lib/runtime.sh
```

统一：

- GPU 数量统计。
- per-device batch size 计算。
- single GPU / torchrun 分支。
- 端口参数。

### Python 生成式 Backbone Registry

`SeqRec/models/generative/registry.py` 记录生成式 backbone 的关键信息：

- model class path。
- decoder-only 与否。
- 是否使用 sessions/actions。
- tokenizer/config 类型。
- train profile。
- 默认 base model。

它减少了 task 和 shell 中硬编码 `if backbone == ...` 的数量。

## 验证规则

脚本改动至少运行：

```bash
bash -n scripts/<script>.sh
```

Python 改动运行：

```bash
python -m compileall main.py SeqRec
conda run -n GAMER flake8 --max-line-length=500 --ignore=F401,E203,W503,F841 <changed-python-files>
```

行为改动使用最小相关 `--help` 或单卡 smoke invocation，避免直接启动完整训练。

## 剩余优化空间

- 进一步减少 shell 中 task-specific 参数列表。
- 对 train/test 脚本建立更统一的 task profile。
- 将更多 dataset/tokenization/path 规则移动到 Python 或 declarative config。
- 对常用实验命令生成模板，降低人工拼接长命令的成本。
