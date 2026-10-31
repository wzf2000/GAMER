# 可转交 review 整理会话的四项核查补充

核查日期：2026-10-26；依据版本：`dev` / `38a90d2`。核查对象为投稿版 **TH-FixedSoft**。以下内容可直接转交；它记录工程事实与待补证据，不代写审稿回应。

**2026-10-31 层级纠正与启动：** 作者明确论文 Tmall 主方法采用弱辅助层级，而不是早期实验清单的 legacy identity-order 对照。Tmall 层级干预暂缓，改做 JData-V3。所给主方法结果压缩包的名义 test buy HR@10／NDCG@10 为 **0.726086957／0.550338765**，230 个用户。补充命令为四卡、每卡128、累积8，即有效 batch4096。按要求预先指定 **pv=2、click=1、cart=collect=0、buy=3**，已于 **2026-10-31 19:59:13 Asia/Shanghai** 在四张 A100 启动，后续自动进行 validation／test ranking。`389c99a` 增加 V3 显式启用支持，保持 legacy 默认兼容，并与历史 V3 实现通过权重／logits／loss／梯度／cache 一致性测试。启动时尚无最终结果；历史数据／索引／checkpoint／评测来源仍需核对。详见 `2026-10-31-jdata-v3-hierarchy-intervention.zh.md`。

**2026-10-27 补充：** 用户提供的两个 results 压缩包已恢复 Tmall 表格三行的结果来源。`results_wzf.zip` 中 SASRec=`smb_dis_sample/SASRec`、SASRec_B=`smb_dis_sample_diff/SASRec`、TIGER=`smb/TIGER`，均取 alipay 行，六个四位小数与截图完全匹配。TIGER 结果标记 CID-shuffle-64，未显式编码行为；另一个 `smb_explicit/TIGER` 是不同结果。仍缺历史 checkpoint/config/命令、调参和运行 manifest，不能确认最终超参。详见 `2026-10-27-tmall-baseline-provenance.zh.md`；下文“未找到 baseline 结果”的旧快照应结合这次更新阅读。

**SASRec 正确性补充：** CPU 探针确认当前两个 SASRec 变体的 FFN 残差条件反向，由 `7aae2a3` 引入，默认跳过 FFN 残差／dropout／LayerNorm。这是 baseline 实现缺陷，但对历史分数的影响尚未测量。应优先纠正并受控重训，再将 TIGER/SASRec 大幅差距解释为模型能力。详见 `2026-10-27-sasrec-tiger-gap-audit.zh.md`。

**修复／重训更新：** FFN 随后已在 `d8d06c3` 修复并验证。按用户授权，Tmall SASRec／SASRec_B 已在 GPU 1／5 启动新训练，使用独立输出，保留现有采样／选模／test 流程。完成进展见下方更新；实际状态、明确设置、缓存等价性检查、命令及来源记录见 `2026-10-27-sasrec-ffn-fix-reruns.zh.md`。

**首个完成重训：** 普通 SASRec 已完成全量 test，alipay HR@10 为 **0.216621899**、NDCG@10 为 **0.178387853**，旧论文归档为 **0.076789382／0.053329522**，达到旧值的 2.82099／3.34501 倍。最佳 epoch 17，epoch 37 早停，seed 42，test 用户数 6,329。本次检查时 SASRec_B 仍在训练。历史超参仍缺失，不能认定为严格 FFN 单因素效果或配对显著性；详见重训记录。

**历史配方补充：** 用户随后找回普通 SASRec 命令，原设置为 `max_his_len=50、patience=10`，首轮修复重训为 20／20。其余显式训练设置一致，config 快照与 2025-10-04 受版本管理的配置逐字节相同。因此历史 20 的结果不是严格配方匹配的 FFN 单因素对照。按要求新增的 **history-50／patience-10** 普通 SASRec 已完成全候选 test：alipay HR@10 **0.215357876**、NDCG@10 **0.177510631**，最佳 epoch **11**，epoch **21** 早停，seed 42，6,329 个用户。结果为 `results/rebuttal/2026-10-27-sasrec-ffn-fixed-history50-patience10/sasrec/seed_42/results/result-smb_dis.json`。原始历史训练 checkout／数据／环境仍未核验。

**2026-10-28 SASRec_B 更新：** 修复后的 history-20／patience-20 SASRec_B 也已完成全候选 test：alipay HR@10 **0.012166219**、NDCG@10 **0.009992610**，最佳 epoch **19**，epoch **39** 早停；结果为 `results/rebuttal/2026-10-27-sasrec-ffn-fixed/sasrec_b/seed_42/retry-expandable/results/result-smb_dis_diff.json`。三组已完成运行的退出码均为 0，alipay 逐用户均值与汇总一致。用户随后要求新增 **history-50／patience-10 SASRec_B**，已于 **2026-10-28 00:37:36 Asia/Shanghai** 在 GPU 5 启动，目录为 `results/rebuttal/2026-10-28-sasrec-b-ffn-fixed-history50-patience10/sasrec_b/seed_42/`，seed 42、batch 4096、FP32，将自动用最佳 validation checkpoint 执行全候选 test。该运行已于 **2026-10-28 02:44:40 Asia/Shanghai** 完成训练和 test，退出码 0，alipay HR@10 **0.011376205**、NDCG@10 **0.009331325**，最佳 epoch **14**、epoch **24** 早停。相比 history-20／patience-20，HR@10 下降 **6.49%**，NDCG@10 下降 **6.62%**。已核验两组 test UID 均为相同的 6,329 个用户，全部逐用户均值与汇总一致。这是 seed 42、历史长度和 patience 同时变化的结果，不能隔离任一因素。这是明确要求的新配方，历史 SASRec_B 命令本身仍未找回。准确命令、验证和来源见重训记录。

**2026-10-28 P0 最终 test：** seed 42 的 Full／TH-FixedSoft、等容量普通 decoder 和 no-QKV 均已完成最终最佳权重的 **test／cvr** 评测，HR@10／NDCG@10 分别为 **0.198542805／0.087807118**、**0.197973588／0.087893460**、**0.203551913／0.089708056**。三组均使用 original SID，已核验相同的 8,784 个 UID／16,812 个 held-out cvr 事件。按全行为 validation loss 选出的最佳 step 分别为 6318／7137／6201。普通 decoder 的 cvr 在核验权重／config／tokenizer／generation 设置相同且打分代码未变后复用；no-QKV 的旧 step4914 结果未用于替代 step6201。

click／p3s 及 merged 也均已完成。Merged HR@10／NDCG@10 为 Full **0.208987251／0.073674579**、普通 decoder **0.212072800／0.074915730**、no-QKV **0.212652320／0.075616105**。已独立核验逐用户均值、各行为 UID／target 一致性及 merged 加权值。该单 seed 下 no-QKV 的 cvr HR@10／NDCG@10 高于 Full，Full 与普通 decoder 的这两项指标接近，没有多 seed 显著性结论。完整事实、协议、复用来源及路径见 `2026-10-28-p0-final-test.zh.md`。汇总文件为 `results/rebuttal/2026-10-26-p0/{full,plain_matched,no_qkv}/seed_42/test-ranking.json`；完整 CSV 和核验文件位于 `results/rebuttal/2026-10-28-final-test/`。

**找回投稿命令后的配方对照：** 用户提供了原 TH-FixedSoft 训练命令，显式 lr=5e-4、每卡 batch 128、梯度累积 8、warmup 0.04、patience 20、temperature 0.7、最多 200 epoch、历史 100、任务及 original SID 名称均与新 Full 一致。结合作者此前的四卡 A800 资源记录，旧有效 batch 为 **4096**，新双卡 A100 的 Full／plain／no-QKV 系列为 **2048**。模型模板未变，在当前输入上解析后与新 Full 配置完全一致。因此本轮受控系列与投稿的 optimizer batch 不同，不能说 lambda、QKV／gate／FFN 或学习率改变了。旧 launcher／最终 checkpoint／数据 hash／测试命令仍缺失，旧 test 与 validation 结果文件应继续区分。详见 `2026-10-28-full-submission-recipe-comparison.zh.md`。

1. **ShortVideoAD 数据口径尚不能从实验仓库确定。** 当前代码读取已经处理好的 `ShortVideoAD.SMB.time.json`，将时间字符串减去用户首次交互时间并除以 1800，得到半小时单位。这个归一化不代表归因窗口。仓库没有可确认时间戳业务含义、归因窗口、延迟转化截断/补偿或点击回填规则的上游实现。需数据生产方提供字段与归因/快照说明，或生成这些 JSON 的脚本；目前不能声称“使用转化发生时间”“窗口为某个长度”或“没有回填”。证据：`SeqRec/datasets/session_behavior/base.py:77`、`:126`。

2. **公开数据只能确认下游切分，不能确认上游行为链合并。** 未找到 Tmall/JData 原始日志到最深行为状态序列的预处理脚本。代码加载既有 session，再将倒数第二个 session 留作 validation、最后一个 session 留作 test，之后构造训练前缀、增强与截断。合并窗口、是否跨 session、合并与 session 划分先后顺序仍需原始预处理脚本或记录。不能从 `Tmall-24-0.25` 的命名推断 24 小时合并窗口。V2 数据文件主要链接到此前处理后文件，不证明保留了 raw。证据：`SeqRec/datasets/session_behavior/base.py:98`；`data/Tmall-24-0.25-V2/`、`data/JData-V2/`。

3. **当前 SMB 执行路径可确认行为前缀与训练监督范围，但最终历史配置尚缺。** 若 CTR 指 click 行，cvr 使用 cvr prompt/prefix、click 使用 click prompt/prefix；两者不会都固定使用 conversion prefix。各行为候选 item catalogue 来自整个索引文件，不能将其描述成仅倒数第三个 session 的候选。`smb_explicit_decoder_4` 对 validation 之前的训练前缀保留原视图并增加最多四个随机视图；末尾目标通常位于倒数第三个 session，但 causal shifted loss 还监督历史窗口内更早训练 session 的 item tokens，padding/行为 token 被屏蔽。这里的 cvr/click 指行为条件的 next-item ranking，不是转化/点击概率预测。`smb_explicit` 对应 test，`smb_explicit_valid` 对应 validation ranking，`--valid_loss` 是 validation loss。结果文件没有完整运行 manifest，历史实际 split 仍应结合对应命令/日志核验。证据：`SeqRec/tasks/evaluation/test_SMB_decoder.py:93`、`:104`；`SeqRec/datasets/session_behavior/decoder.py:71`；`SeqRec/datasets/collators/generative.py:129`。

4. **lambda 的实现可确认，选择过程与最终 baseline 记录不能确认。** 用户确认投稿 Eq. (8) 的 lambda=0.05 对应 `th_relation_bias_soft_scale`。FixedSoft 源配置与实现使用冻结 relation buffer，初始化为 `0.05 * min(a-b,0)`，额外运行时倍率默认 1；不是 loss 系数。固定配置本身不能证明 0.05 是预设还是经 validation 选择，需补搜索/设计记录。本机有 FixedSoft 的 test/validation 命名结果及 8784 条 cvr test 逐用户指标，但未找到最终 FixedSoft checkpoint 和 SASRecB/TIGER 的实际调参与最终模型记录。源配置 dropout 为 SASRec 0.5、TIGER 0.1、FixedSoft 0.2，仅能说明源默认值，不能据此断言最终设置或调参公平性。现存 Qwen3Multi checkpoint 的 trainer state 指向 MB 路径，不能替代投稿 FixedSoft。证据：`config/s2s-models/Qwen3TemporalHierarchicalFixedSoft/config.json:20`；`SeqRec/models/generative/qwen3/temporal_hierarchical.py:_init_level_pair_bias`；`results/ShortVideoAD/smb_explicit_decoder_4/Qwen3TemporalHierarchicalFixedSoft/`。

需要补齐的材料清单：ShortVideoAD 字段/归因/回填规则；Tmall/JData 实际预处理脚本；最终训练与评测命令及 split/候选清单；FixedSoft checkpoint/config/training_args/代码版本；lambda 选择记录；SASRecB/TIGER 搜索空间、validation 分数、选模规则和最终 test 记录。

完整核查和成本假设见 `2026-10-26-rebuttal-experiment-audit.zh.md`。后续新增开发支持不能反向证明历史实验采用了相同实现或设置。
