# SASRec/TIGER 指标差距：已确认实现缺陷与设置差异

英文对应：`2026-10-27-sasrec-tiger-gap-audit.md`。核查 checkout：`ca0d931`；本次未修改模型／训练源码，也未启动正式训练。

**后续修复：** 用户已授权修复并优先重跑 Tmall 两种 SASRec。提交 `d8d06c3` 已纠正残差条件，通过 CPU 测试、规定 lint 及完整词表规模的 GPU 单步验证；已在 GPU 1、5 启动两组新训练，保留现有流程。详见 `2026-10-27-sasrec-ffn-fix-reruns.zh.md`。下文描述修复前的核查快照，不代表当前已纠正的前向。

**首个完成结果：** 普通 SASRec 修复重训已于 2026-10-27 完成全量 test，alipay HR@10 为 **0.216621899**、NDCG@10 为 **0.178387853**，旧归档为 **0.076789382／0.053329522**，达到旧值的 2.82099／3.34501 倍；历史混杂条件尚未全部隔离。证据及边界见重训记录；本快照中 SASRec_B 仍在运行。

## 指标差距能说明什么

已找回的 Tmall alipay 结果中，TIGER/SASRec 的 HR@10 比值为 **7.5658 倍**，NDCG@10 为 **8.7901 倍**。两条已定位任务都忽略显式行为身份，但这不能证明网络结构、目标、训练样本、选模方式或历史超参已对齐，不能仅据这些结果将差距归因于生成式推荐。

## 当前 SASRec 路径存在已确认的 FFN 缺陷

`SeqRec/modules/layers/transformer.py` 中 `FeedForward` 默认 `residual=True`，也仅在该条件下创建 dropout／LayerNorm；但 forward 将 dropout、残差相加和 LayerNorm 放在 **`if not self.residual`** 下。两个 SASRec 变体使用的 `TransformerEncoderLayer` 没有覆盖这个参数，因此默认 FFN 跳过了这三步。

这与仓库重构前的实现以及 [RecBole FeedForward 参考实现](https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/layers.py) 不同；参考实现对 FFN 输出执行 dropout 与残差 LayerNorm。条件反向由 **2025-09-28** 的提交 `7aae2a34f9d865a6db6fad153dc48b39a2e38cf5` 引入。压缩包记录的 SASRec 结果文件时间为 2025-10-05，但文件时间不能证明实际训练 checkout。

CPU 探针，seed 42：

- 只将 FFN 的两层 dense 投影置零，保留 LayerNorm weight=1、bias=0：实际输出最大绝对值为 **0**，预期残差 LayerNorm 输出的最大绝对值为 **2.3465452194**。
- forward hook 记录 FFN dropout 和 LayerNorm 均被调用 **0 次**。
- 完整两层微型 SASRec 的 loss／反向传播中，四个 FFN LayerNorm weight/bias 张量均没有梯度，其余注册参数有梯度。
- 显式设置 `residual=False` 则触发 `AttributeError: 'FeedForward' object has no attribute 'dropout'`，进一步确认构造和前向分支不一致。

这是已确认的实现错误，但尚不是论文完整指标差距的实测解释。缺少原训练 commit／checkpoint，影响大小需要受控的修正后训练验证；仅修改推理代码，不能修复按错误计算过程训练出的模型。

## 当前代码下其他已确认差异

| 方面 | SASRec `smb_dis_sample` | TIGER `smb` |
| --- | --- | --- |
| item 表示 | 原子 item ID，全量 item softmax | CID token 序列，自回归 token 预测 |
| 源默认结构 | 2 层 causal encoder、FFN 256 | 4 层 encoder＋4 层 decoder、FFN 1024 |
| 源 dropout | 0.5 | 0.1 |
| 默认选模 | alipay 采样 validation ranking，默认 NDCG@10 | 全行为 validation token loss，除非显式过滤 |
| 空历史训练样本 | 跳过 | 除每用户第一个事件外保留 |

源默认值不能证明历史最终设置。单射 CID 改变参数共享和优化方式，但仅重命名 item 不会带来行为身份或外部语义信息。也不能从层数推断总参数容量：SASRec 的大量原子 item embedding 占据其主要参数量。

只读扫描当前 Tmall 事件／session JSON，按现有默认 session 训练规则计算、未使用历史缓存样本替代：

- 217,374 个用户、3,818,122 个事件、**361,932 个唯一 item**，ID 为 0 至 361,931，中间没有空洞。
- 当前可用的 V2 CID 索引包含相同的 item key 集合，且 **361,932 个 CID 字符串均唯一**、没有碰撞。该索引不是独立找回的历史索引。
- 生成式 `smb` 构造 **1,782,306** 个样本；判别式构造 **1,057,649** 个样本。差出的 **724,657** 个是没有此前 session 历史的首 session 目标；生成式 builder 保留，判别式 builder 跳过。已验证所有用户的 session ID 非递减。
- 以上是当前构造规则的计数，不是历史训练 manifest，也没有测量这些差异的性能影响。历史缓存、历史长度或代码版本的差异仍需原记录。

当前 test 代码下，普通 SASRec 与 `smb` TIGER 的 item catalogue 基本一致，差别包括 padding 处理及精确全量排序／beam search。所给 TIGER 结果记录 target SID 碰撞数为零；三份历史 alipay HR 均与 6,329 个用户上的整数命中数相容。SASRec 没有逐用户 UID 文件，因此历史用户集合是否逐项一致仍未独立核验。

## 判断与下一步核验

原核查建议优先修正并独立验证 SASRec FFN，再对齐混合历史输入、非空历史训练目标、历史长度、全候选 alipay validation/test、选模方式和调参预算。用户随后已授权修复重训，其他流程改动暂缓；实测结果见上方更新，原核查本身没有给出修正后分数或显著性结论。

诊断产物：`results/rebuttal/2026-10-27-sasrec-tiger-gap-audit/diagnostics.json`。原始结果来源见 `2026-10-27-tmall-baseline-provenance.zh.md`。运行产物不纳入 Git。
