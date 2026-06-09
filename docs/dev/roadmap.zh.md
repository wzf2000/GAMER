# 开发路线图

## 当前优先级

- 将项目级 agent 规则和开发流程文档保留在 `dev` 分支。
- 维护 GAMER arXiv 版本的 paper-to-code 映射，方便后续实现工作和论文方法、实验设置对应。

## 计划工作

- 原型化时序-层级 cross-level 建模变体，最初从 Qwen3Multi cross-attention 中的可学习 relation bias 开始。
- 确认论文 appendix 中提到的 graph baseline 是外部实现、当前快照遗漏，还是以其他名称存在于仓库中。

## 更新记录

- 2026-06-01: 创建开发文档结构。
- 2026-06-01: 添加 arXiv:2511.03155 的初始 paper-to-code 映射。
- 2026-06-01: 添加下一版 GAMER 的时序-层级行为建模设计计划。
- 2026-06-03: 完成 `SeqRec` 包结构重构，涉及 datasets/loaders/collators、generative model family、task registry 和 utils 命名清理。该批重构减少重复代码，并移除旧 shim 文件。
- 2026-06-04: 继续完成 task/model 复用抽取，包括 evaluation shared base、analysis shared base、Qwen3 decoder base consolidation，以及 generative backbone registry 的进一步使用。使用 Toy checkpoint 验证多个 family 的结果文件和 per-uid 指标保持一致。

## 说明

详细重构计划和执行记录分别见：

- `docs/dev/design/2026-06-03-framework-reuse-refactor-report.md`
- `docs/dev/design/2026-06-03-framework-reuse-refactor-report.zh.md`
- `docs/dev/design/2026-06-03-scripts-maintenance-refactor.md`
- `docs/dev/design/2026-06-03-scripts-maintenance-refactor.zh.md`
