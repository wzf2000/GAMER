# BatchEncoding 属性与字典访问差异问题

## 背景

在执行 `test_MB_decoder` 的 P4 重构时，目标是将 Qwen3Multi 内联的 `generate` 调用迁移到共享的 `build_generation_kwargs` helper。使用 1-epoch Qwen3Multi Toy MB checkpoint 做运行验证时，在 `Qwen3MultiModel._update_session_multi_cross_mask` 内遇到 tensor shape mismatch：

```text
RuntimeError: The size of tensor a (21) must match the size of tensor b (20)
              at non-singleton dimension 3
```

原内联调用可以工作，而 helper 构造的调用失败，尽管表面上 kwargs 看起来相同。

## 根因

`transformers.BatchEncoding` 会通过 `__getattr__` 回退到底层 `data` 字典，但普通属性赋值只会写在 Python 对象属性上，不会同步写回底层字典。

`test_MB_decoder.test_single_type` 在 `TARGET_BEHAVIOR` / `BEHAVIOR_SPECIFIC` 分支中执行：

```python
action = [[dataset.behavior_level[u]] for u in behaviors]
inputs.actions = torch.cat([inputs.actions, torch.tensor(action, device=...)], dim=1)
```

执行后：

| 访问方式 | 返回值 |
|---|---|
| `inputs.actions` | 新的扩展 tensor，长度 21 |
| `inputs["actions"]` | 原 collator tensor，长度 20 |
| `inputs.get("actions")` | 原 collator tensor，长度 20 |
| `getattr(inputs, "actions", None)` | 新的扩展 tensor，长度 21 |

原内联调用使用 `generate(..., actions=inputs.actions, ...)`，因此拿到长度 21 的新 tensor。helper 第一版使用 `inputs.get("actions")`，因此误拿到旧长度 20 tensor，最终在模型 mask builder 中与 input_ids 长度不一致。

## 修复

对于来自测试任务、且可能被局部属性赋值修改过的 `BatchEncoding`，helper 内读取可选字段时统一使用：

```python
actions = getattr(inputs, "actions", None)
```

而不是：

```python
inputs.get("actions")
inputs["actions"]
```

`getattr` 能同时兼容显式属性赋值和 BatchEncoding 字典 fallback；字段真的不存在时返回默认值。

## 验证

使用同一个 Qwen3Multi Toy MB checkpoint 和同一脚本调用，修复前后生成的 `results-original.json` 与 per-uid `user_level_metrics_*.json` 字节级一致。

## 经验规则

- helper 如果可能在测试任务修改 `inputs.<field>` 后被调用，不要使用 `inputs.get(key)` 或 `inputs[key]` 读取该字段。
- 如果 helper 需要写回字段，应与周围代码风格保持一致；当前风格是属性赋值，如 `inputs.actions = ...`。
- 对 collator 可能不存在的可选字段，如 MB 数据无 `session_ids`，`getattr(inputs, name, None)` 可以保持 MB / SMB / sequential 数据集之间的兼容。

## 相关位置

- Commit `52e8bfb`: 应用修复并移除 Qwen3Multi 内联分支。
- `SeqRec/tasks/evaluation/helpers.py::build_generation_kwargs`: 修复所在 helper。
- `SeqRec/tasks/evaluation/test_MB_decoder.py::TestMBDecoder.test_single_type`: 修改 `inputs.actions` 的调用点。
