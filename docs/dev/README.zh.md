# 开发文档

`docs/dev/` 用于保存应该留在 `dev` 分支上的规划、设计说明、技术决策和开发记录。

## 目录结构

- `roadmap.md` / `roadmap.zh.md`: 长期项目规划、里程碑、优先级和当前工作。
- `design/`: 具体功能、实验或重构的设计文档。
- `decisions/`: 重要技术或流程选择的决策记录，可使用轻量 ADR 风格。
- `notes/`: 临时开发记录、调试记录和实验观察。

## 当前索引

- `design/2026-06-01-paper-code-map.md` / `.zh.md`: 将 GAMER 论文概念映射到当前代码实现、baseline 和训练/测试流程。
- `design/2026-06-01-temporal-hierarchical-behavior-modeling.md` / `.zh.md`: 围绕时序-层级行为建模和 cross-level 模块改造提出下一版 GAMER 方法方向。
- `design/2026-06-09-th-results-and-next-design.md` / `.zh.md`: 总结 TH 变体实验结果，并给出后续模型设计建议。
- `decisions/2026-06-01-temporal-hierarchical-method-direction.md` / `.zh.md`: 记录下一版 GAMER 在输入表示、attention 架构和层分配上的已接受方向。

## 文件命名

英文 Markdown 文件使用：

```text
YYYY-MM-DD-short-topic.md
```

中文对应版本使用同名 `.zh.md` 后缀：

```text
YYYY-MM-DD-short-topic.zh.md
```

示例：

```text
2026-06-01-branch-and-rules-policy.md
2026-06-01-branch-and-rules-policy.zh.md
```

## 双语规则

`docs/dev/` 下的 Markdown 文档应保留英文和中文两个版本。新增、重命名或大幅更新文档时，同步更新对应语言版本；如果某个版本暂时无法完整同步，应在文件中明确标注差异和待补内容。
