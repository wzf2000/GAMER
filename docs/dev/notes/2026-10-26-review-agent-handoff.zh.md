# 可转交 review 整理会话的四项核查补充

核查日期：2026-10-26；依据版本：`dev` / `38a90d2`。核查对象为投稿版 **TH-FixedSoft**。以下内容可直接转交；它记录工程事实与待补证据，不代写审稿回应。

**2026-10-27 补充：** 用户提供的两个 results 压缩包已恢复 Tmall 表格三行的结果来源。`results_wzf.zip` 中 SASRec=`smb_dis_sample/SASRec`、SASRec_B=`smb_dis_sample_diff/SASRec`、TIGER=`smb/TIGER`，均取 alipay 行，六个四位小数与截图完全匹配。TIGER 结果标记 CID-shuffle-64，未显式编码行为；另一个 `smb_explicit/TIGER` 是不同结果。仍缺历史 checkpoint/config/命令、调参和运行 manifest，不能确认最终超参。详见 `2026-10-27-tmall-baseline-provenance.zh.md`；下文“未找到 baseline 结果”的旧快照应结合这次更新阅读。

**SASRec 正确性补充：** CPU 探针确认当前两个 SASRec 变体的 FFN 残差条件反向，由 `7aae2a3` 引入，默认跳过 FFN 残差／dropout／LayerNorm。这是 baseline 实现缺陷，但对历史分数的影响尚未测量。应优先纠正并受控重训，再将 TIGER/SASRec 大幅差距解释为模型能力。详见 `2026-10-27-sasrec-tiger-gap-audit.zh.md`。

**修复／重训更新：** FFN 随后已在 `d8d06c3` 修复并验证。按用户授权，Tmall SASRec／SASRec_B 已在 GPU 1／5 启动新训练，使用独立输出，保留现有采样／选模／test 流程。完成进展见下方更新；实际状态、明确设置、缓存等价性检查、命令及来源记录见 `2026-10-27-sasrec-ffn-fix-reruns.zh.md`。

**首个完成重训：** 普通 SASRec 已完成全量 test，alipay HR@10 为 **0.216621899**、NDCG@10 为 **0.178387853**，旧论文归档为 **0.076789382／0.053329522**，达到旧值的 2.82099／3.34501 倍。最佳 epoch 17，epoch 37 早停，seed 42，test 用户数 6,329。本次检查时 SASRec_B 仍在训练。历史超参仍缺失，不能认定为严格 FFN 单因素效果或配对显著性；详见重训记录。

**历史配方补充：** 用户随后找回普通 SASRec 命令，原设置为 `max_his_len=50、patience=10`，修复重训为 20／20。其余显式训练设置一致，config 快照与 2025-10-04 受版本管理的配置逐字节相同。因此已完成结果不是严格配方匹配的 FFN 单因素对照；详见重训记录中的历史命令部分，本次核查未另启匹配配方的新训练。

1. **ShortVideoAD 数据口径尚不能从实验仓库确定。** 当前代码读取已经处理好的 `ShortVideoAD.SMB.time.json`，将时间字符串减去用户首次交互时间并除以 1800，得到半小时单位。这个归一化不代表归因窗口。仓库没有可确认时间戳业务含义、归因窗口、延迟转化截断/补偿或点击回填规则的上游实现。需数据生产方提供字段与归因/快照说明，或生成这些 JSON 的脚本；目前不能声称“使用转化发生时间”“窗口为某个长度”或“没有回填”。证据：`SeqRec/datasets/session_behavior/base.py:77`、`:126`。

2. **公开数据只能确认下游切分，不能确认上游行为链合并。** 未找到 Tmall/JData 原始日志到最深行为状态序列的预处理脚本。代码加载既有 session，再将倒数第二个 session 留作 validation、最后一个 session 留作 test，之后构造训练前缀、增强与截断。合并窗口、是否跨 session、合并与 session 划分先后顺序仍需原始预处理脚本或记录。不能从 `Tmall-24-0.25` 的命名推断 24 小时合并窗口。V2 数据文件主要链接到此前处理后文件，不证明保留了 raw。证据：`SeqRec/datasets/session_behavior/base.py:98`；`data/Tmall-24-0.25-V2/`、`data/JData-V2/`。

3. **当前 SMB 执行路径可确认行为前缀与训练监督范围，但最终历史配置尚缺。** 若 CTR 指 click 行，cvr 使用 cvr prompt/prefix、click 使用 click prompt/prefix；两者不会都固定使用 conversion prefix。各行为候选 item catalogue 来自整个索引文件，不能将其描述成仅倒数第三个 session 的候选。`smb_explicit_decoder_4` 对 validation 之前的训练前缀保留原视图并增加最多四个随机视图；末尾目标通常位于倒数第三个 session，但 causal shifted loss 还监督历史窗口内更早训练 session 的 item tokens，padding/行为 token 被屏蔽。这里的 cvr/click 指行为条件的 next-item ranking，不是转化/点击概率预测。`smb_explicit` 对应 test，`smb_explicit_valid` 对应 validation ranking，`--valid_loss` 是 validation loss。结果文件没有完整运行 manifest，历史实际 split 仍应结合对应命令/日志核验。证据：`SeqRec/tasks/evaluation/test_SMB_decoder.py:93`、`:104`；`SeqRec/datasets/session_behavior/decoder.py:71`；`SeqRec/datasets/collators/generative.py:129`。

4. **lambda 的实现可确认，选择过程与最终 baseline 记录不能确认。** 用户确认投稿 Eq. (8) 的 lambda=0.05 对应 `th_relation_bias_soft_scale`。FixedSoft 源配置与实现使用冻结 relation buffer，初始化为 `0.05 * min(a-b,0)`，额外运行时倍率默认 1；不是 loss 系数。固定配置本身不能证明 0.05 是预设还是经 validation 选择，需补搜索/设计记录。本机有 FixedSoft 的 test/validation 命名结果及 8784 条 cvr test 逐用户指标，但未找到最终 FixedSoft checkpoint 和 SASRecB/TIGER 的实际调参与最终模型记录。源配置 dropout 为 SASRec 0.5、TIGER 0.1、FixedSoft 0.2，仅能说明源默认值，不能据此断言最终设置或调参公平性。现存 Qwen3Multi checkpoint 的 trainer state 指向 MB 路径，不能替代投稿 FixedSoft。证据：`config/s2s-models/Qwen3TemporalHierarchicalFixedSoft/config.json:20`；`SeqRec/models/generative/qwen3/temporal_hierarchical.py:_init_level_pair_bias`；`results/ShortVideoAD/smb_explicit_decoder_4/Qwen3TemporalHierarchicalFixedSoft/`。

需要补齐的材料清单：ShortVideoAD 字段/归因/回填规则；Tmall/JData 实际预处理脚本；最终训练与评测命令及 split/候选清单；FixedSoft checkpoint/config/training_args/代码版本；lambda 选择记录；SASRecB/TIGER 搜索空间、validation 分数、选模规则和最终 test 记录。

完整核查和成本假设见 `2026-10-26-rebuttal-experiment-audit.zh.md`。后续新增开发支持不能反向证明历史实验采用了相同实现或设置。
