# Rebuttal 补实验可执行性与证据核查

日期：2026-10-26。代码快照：`dev` / `38a90d2`。
英文对应文件：`2026-10-26-rebuttal-experiment-audit.md`。

本记录描述上述快照的核查结论；之后的开发支持与验证见 `../design/2026-10-26-rebuttal-experiment-support.zh.md`。新增接口不改变本记录对历史产物缺失的判断。

## 核查对象与边界

- 投稿模型以 **TH-FixedSoft** 为准，不以 Original GAMER / Qwen3Multi 或最新增强变体替代。
- 用户确认正文 Eq. (8) 为 `r_soft(a,b) = lambda * min(a-b, 0)`，`lambda=0.05`；这是固定 hierarchy prior 的尺度，不是辅助 loss 权重。
- 本机可检查代码、处理后数据、已有结果、日志及部分 checkpoint。用户引用的 macOS 论文与资源记录路径未在本机读取，相关公式和硬件耗时以本次用户提供内容为依据。
- 本轮仅提交先前已验证的 loader 默认参数修复。未增加消融实现，未启动完整训练或大规模评测。

## 四项事实核查

### 1. ShortVideoAD 时间戳、归因与延迟转化

可确认：`BaseSMBDataset._load_data()` 读取 `*.SMB.time.json` 中的日期字符串，使用 `%Y-%m-%d %H:%M:%S` 解析，减去该用户第一次交互时间，再除以 1800；缓存 `time` 的单位是半小时。样本中的 `time` 又经过相对目标时间的转换。

不能确认：时间戳是曝光、点击、转化发生、归因、入库还是压缩后的代表时间；归因窗口长度；延迟转化截断/补偿方式；是否回填此前点击。读取处理后 JSON 的代码没有提供这些业务定义。半小时归一化也不是半小时归因窗口的证据。

需要补充：数据生产方字段说明、归因与快照规则，或生成这些 JSON 的上游实现。

证据：`SeqRec/datasets/session_behavior/base.py:77`、`:126`、`:_generate_times`。

### 2. Tmall/JData 行为链构造

当前仓库及可见 Git 历史中未找到原始事件日志到最深行为状态序列的实际预处理脚本。`analysis/ShortVideoAD_behavior_analysis.ipynb` 读取既有 JSON/pickle，不能作为上游原始日志构造脚本。

可确认的下游顺序：读取已经构造好的 interaction / behavior / session → 用倒数第二个 session 的起点划出 validation、最后一个 session 的起点划出 test → 构造训练前缀和增强视图 → 按历史长度截断并构造 collator 输入。

不能确认：同一 user-item 的合并窗口、是否跨 session 合并，以及上游合并与 session 划分的先后顺序。不能从 `Tmall-24-0.25` 名称推断 24 小时合并窗口。

可用处理后输入位于 `data/Tmall-24-0.25-V2/` 和 `data/JData-V2/`，均有 CID 索引。V2 的 interaction / behavior / session / time 文件分别链接到原目录；V2 名称不代表保留了另一份 raw 日志。

层级文件还存在需明确的差异：

| 数据目录 | `behavior_level.json` |
| --- | --- |
| Tmall | click=0, collect=1, cart=1, alipay=2 |
| Tmall-24-0.25-V2 | click=0, collect=1, cart=1, alipay=1 |
| JData-V2 | pv=0, click=1, cart=2, collect=2, buy=2 |

基类遇到多个最高层行为时，会警告并选择最后一个作为 `target_behavior`。这不等于 TH prior 已保留这些并列关系，见后面的层级干预核查。

证据：`SeqRec/datasets/session_behavior/base.py:98`、`:140`；上述数据目录。

### 3. CVR/CTR 的执行路径

若 CTR 指结果中的 click 行，当前 `test_SMB_decoder` 的结论是：

- cvr 使用 cvr prompt 和对应的 candidate prefix；click 使用 click prompt 和对应的 candidate prefix。两者不会都固定使用 conversion prefix。
- 各行为的候选 item catalogue 来自整个 `index_file`，然后添加对应行为 token；不是从倒数第三个 session 动态构造候选。索引的上游生成是否采用 train-only item 集合，不能由这段代码证明。
- 当前任务计算行为条件下的 next-item ranking 指标，不提供转化/点击概率预测的 AUC、LogLoss 或校准评估。
- `test_task=smb_explicit` 使用 test；`smb_explicit_valid` 使用 validation ranking；`--valid_loss` 走 validation loss。`test_MB_decoder` 的 Target Behavior / Behavior Specific / Behavior Item 是另一组协议，不能混用。

训练范围必须区分三个概念：

1. `smb_explicit_decoder_4` 使用 validation 之前的全部训练前缀，构造原序列和最多四个增强序列；最后一个目标来自该前缀最后保留的事件，原视图通常位于倒数第三个 session。
2. decoder-only 训练采用整序列 causal shifted loss，padding 和行为 token 被屏蔽；监督涵盖截断后保留的较早训练 session，不仅是上述末尾目标。
3. 历史窗口外的事件没有进入当前样本；训练交叉熵使用 token vocabulary，没有单独按 session 采样 item 候选集合。

原始 random4 的删除 ratio 为 `1/4, 2/4, 3/4, 4/4`，另保留原视图；过短增强视图会跳过，不能简单保证每人恰好五条。

已找到的配置/缓存位置：

- `config/s2s-models/Qwen3TemporalHierarchicalFixedSoft/config.json`
- `data/ShortVideoAD/ShortVideoAD.index.json`
- `data/ShortVideoAD/ShortVideoAD.SMB.data.pkl`
- `data/ShortVideoAD/ShortVideoAD.SMBExplicitDatasetForDecoder.100.SMB.aug4.train.pkl`
- `data/ShortVideoAD/ShortVideoAD.SMBExplicitDataset.100.SMB.valid.pkl`
- `data/ShortVideoAD/ShortVideoAD.SMBExplicitDataset.100.SMB.test.pkl`

以上不能替代投稿最终训练命令或 checkpoint 内的有效配置。

证据：`SeqRec/tasks/evaluation/test_SMB_decoder.py:93`、`:104`；`SeqRec/tasks/evaluation/helpers.py:62`；`SeqRec/datasets/session_behavior/base.py:398`；`SeqRec/datasets/session_behavior/decoder.py:28`、`:71`；`SeqRec/datasets/collators/generative.py:129`；`SeqRec/models/generative/common/temperature.py`。

### 4. lambda、最终 checkpoint 与 baseline 记录

- 已验证 FixedSoft 配置的 `th_relation_bias_soft_scale=0.05`、`th_relation_bias_init=soft`、`th_relation_bias_trainable=false`；运行时 `th_relation_bias_scale` 默认 1、learnable scale 默认关闭。relation table 是 buffer，不计入训练参数。
- 冻结偏置与 Eq. (8) 的代数形式一致。但是“0.05 是预设还是经 validation 选择”需要实验设计/搜索记录；现有固定配置及结果不能回答选择过程。
- 保留了 FixedSoft 的 `results-smb_explicit-original.json`、`results-smb_explicit_valid-original.json` 和 test 逐用户指标。test cvr 有 8784 个用户，HR@5=0.13490437、NDCG@10=0.08998073。
- 上述 test/validation 归类符合文件命名与当前 task 路由；结果文件未保存完整运行配置/split manifest，缺少对应历史评测日志时，不能仅凭文件名独立证明历史运行实际使用的 split。
- 本机未找到 ShortVideoAD FixedSoft / FixedBias 最终 checkpoint 目录，不能核验投稿最终 seed、优化器、精度、batch、停止轮数、attention backend 或实际参数量。
- `checkpoint/SMB-decoder/ShortVideoAD/smb_explicit_decoder_4/Qwen3Multi/original/` 存在权重，但 trainer state 的 best path 指向 `MB-decoder/.../mb_explicit_decoder_4/...`，且与 MB 目录元数据一致；它不是可直接认定的投稿 FixedSoft checkpoint。
- 未找到 SASRecB / TIGER 的实际模型训练、调参、最终模型评测与 checkpoint 记录。部分测试日志中的默认 `base_model=.../TIGER` 实际未用于加载模型；rule task 中的默认 backbone 也不代表训练过 TIGER。
- 代码支持 SASRec + `smb_dis_diff` 的 behavior-item 重映射，但没有记录证明这就是投稿 SASRecB 的最终选择。
- 源配置中 SASRec dropout=0.5、TIGER dropout=0.1、FixedSoft residual/FFN dropout 与 attention dropout=0.2。这只是源配置，不能证明 baseline 的最终有效设置或搜索预算。
- 当前 generative 训练使用 HF Trainer 的 validation loss 选择逻辑；判别模型 Trainer 按 `metrics[-1]`（默认 NDCG@10）选 best。比较公平性还需核实最终选模指标和调参协议。

需要外部补齐：FixedSoft 最终 checkpoint/config/training_args/命令与代码版本，lambda 选择记录，SASRecB/TIGER 的搜索空间、validation 分数、选模规则及最终 test 记录。

## 六项实验的代码支持

### P0 组件消融和等容量普通 decoder

full、zero-prior、共享 FFN 和普通 decoder 均有现成计算路径；还需要独立配置和统一运行清单。去 Q/K/V 条件化、去 attention output gate 尚无独立开关，应在 TH attention 中增加默认兼容的选项，并验证训练、缓存生成和关闭新选项时的兼容性。

- zero-prior：`Qwen3TemporalHierarchicalFixedBias` 与 FixedSoft 仅在 prior 初始化/尺度配置上不同；其 frozen zero table 等价于 lambda=0。也可在 FixedSoft 副本中设置 soft scale=0。
- 共享 FFN：在 FixedSoft 副本中设 `sparse_layers_decoder=[]`，保留 `mlp_type=Qwen3` 和原有 behavior injection 列表，可共享不同 token 位置的专家而保留其余机制。不要直接使用 Qwen3MultiDense 替代该消融。
- 普通 decoder：在 TH 家族配置副本中清空 `temporal_hierarchical_attention_decoder`、`sparse_layers_decoder`、`behavior_injection_decoder` 并关闭 `behavior_injection`。该路径保留相同的 causal mask 和 dropout 实现，再调大 FFN 宽度匹配参数量。

以当前源配置及 `vocab_size=23675`、3 种行为、每 item 5 个 token 为条件，实际构造模型计数如下；这不是最终投稿 checkpoint 的计数：

| 配置 | 可训练参数 |
| --- | ---: |
| FixedSoft full | 29,935,104 |
| zero-prior | 29,935,104 |
| 共享 FFN，保留行为条件化 | 12,240,384 |
| 普通 decoder，同 FFN 宽度 512 | 11,571,200 |
| 普通 decoder，FFN 宽度 3501 | 29,935,616 |

等容量版本仅多 512 个参数。总参数匹配不代表每 token 激活参数/FLOPs 或运行时间匹配；普通 dense FFN 扩宽后需要单独计时。

### P0 full/等容量 decoder 各三 seed

已有 `--seed` 和脚本 `suffix` 可用，但存在已复现的缓存依赖：`SMBExplicitDatasetForDecoder._process_train_data()` 调用 `set_seed(42)`。冷缓存下请求 seed=7，数据构建后全局 torch seed 变为 42；同一缓存命中时仍为 7。模型在数据构建之后初始化，因此需在模型初始化前恢复指定 seed，或严格固定并预生成共享缓存后再启动全部运行。

建议把增强 seed/缓存固定，改变模型初始化和训练随机性；不要把不同增强视图误称为仅改变训练 seed。还需聚合 mean/std 和同 seed 配对差值。工程上约半个工作日，可与组件开关实现合并。

### P0 core-only / raw / consolidated

当前只有处理后状态序列，未找到可证实为真实 raw 事件日志的输入。现有按目标行为过滤评测用户不是 core-only 历史实验；需要新的历史视图构造。

取得真实 raw 后，需要稳定事件/session/目标/candidate manifest、as-of 时间规则，以及三个历史视图共用的 split。报告同 token 预算下的历史覆盖，并另报同历史覆盖下的 token/FLOPs 成本；不能将最深状态反向展开当成 raw。

预计适配与核验 2–4 个工作日，不含取数等待。三组训练的运行时间要先统计序列长度和覆盖率后再估，不能保证等于三个普通 FixedSoft 运行。

### P1 Tmall 弱层级排序干预

需要额外开发独立的 identity→prior rank 映射。原因是 `_init_level_pair_bias()` 对 `num_behavior+1` 做 `arange`，lookup 的 `action_indices` 来自 router；训练 helper 按行为 token 的枚举顺序生成 `behavior_maps`。现有 prior 不直接使用 JSON 中可能并列的 `behavior_level` 数值。

例如 Tmall 的 collect/cart 即使在数据文件中并列，按当前默认枚举仍可能在 prior 中具有不同编号。只改 JSON 数值不构成可靠的模型层级干预，还可能改变增强、core 选择与缓存语义。

推荐仅修改 frozen prior table 初始化使用的 rank 映射，保留行为 token、Q/K/V identity embedding、FFN identity、core 目标、增强视图和候选。该实验检验固定 hierarchy prior 的方向/并列结构，不检验全部行为条件化机制。特殊编号 0 必须另行保持一致。预计开发与验证 0.5–1 个工作日。

### P1 lambda 网格 `{0, 0.01, 0.05, 0.1}`

模型代码已支持；提供四份独立配置并保存最终配置即可。CLI 未注册 `--th_relation_bias_soft_scale`，直接追加这个参数不会自动修改模型 config；应通过 `--base_model` 指向具有 tokenizer 资源的配置目录。

保持 frozen table、runtime multiplier=1、数据和 seed 不变。lambda=0 与组件消融的 zero-prior 重合，0.05 与 full 重合；条件完全一致时只需新增 0.01、0.1 两点。预计配置与核验 1–2 小时。选择或报告敏感性时仍须区分 validation 与 test。

### P1 稀疏用户离线分桶

具备所需主要产物：FixedSoft 和 FixedBias 的 cvr test 逐用户文件各有 8784 条、8 个指标；已逐项验证数组均值与对应 test summary 一致。

需要小型离线汇总工具：当前指标文件按字典序排序 uid 后仅保存数组，未保存 uid 列。因此必须从同一 split 恢复 cvr 用户集合并核对长度和顺序，不能按数字 uid 排序或直接按数组下标连接任意结果。无法证明 UID 顺序时，应重新导出带 UID 的指标。

现有 `analyze_sparse_behavior` 会重新加载两个 checkpoint 并推理，按整个 test 前缀计数，并由最佳命中 rank 重建 NDCG；这与现有多目标 NDCG 不完全等价。新离线工具应直接聚合已保存指标，分别统计模型实际可见窗口（如最后 100 项）和完整历史中的 core 数量，报告桶人数、两模型差值及配对区间。

预计开发/核验 0.5–1 个工作日，运行通常为 CPU 分钟级，GPU 训练需求为零。尚未执行正式分桶，因此没有分桶效果结论。若要与等容量 decoder 或 SASRecB/TIGER 比较，需先获得其可对齐逐用户结果。

## 时间成本与排期口径

用户提供的历史完整流程（包含周期性 validation，不含可分离的单次 test 计时）：

| 历史模型 | 硬件 | 流程时间 |
| --- | --- | --- |
| FixedSoft，factor 4 | 4×A800-SXM4-80GB | 22:34:26 = 22.574 h |
| 无 soft prior | 4×A800-SXM4-80GB | 21:32:27 = 21.541 h |
| 标准 Qwen3 decoder，factor 4 | 8×H800-SXM5-80GB | 08:37:50 = 8.631 h |
| MBGen | 8×A100-SXM4-80GB | 100:27:23 = 100.456 h |

记 `F=22.574 h`，`D` 为尚未测量的等容量普通 decoder 流程时间，`T_Tmall` 为尚未测量的 Tmall 时间。以下是排期算式，不是新增实验的实测或保证：

| 实验 | 工程时间预估 | 训练流程数量/成本 |
| --- | --- | --- |
| 六组组件比较，各一 seed | 1–2 工作日，含参数匹配及开关验证 | 6 次；包括 1 full、1 等容量及 4 消融，各变体时间待测 |
| full/等容量各三 seed | 约 0.5 工作日，部分与上一项重合 | 共 6 次，`3F+3D = 67.72h+3D`；若上一项已有各一 seed，则追加 4 次 |
| core-only/raw/consolidated | 2–4 工作日 + raw 获取等待 | 每 seed 三种视图；需先测各视图 token 长度，当前无可信总时长 |
| Tmall 弱层级映射 | 0.5–1 工作日 | `映射数×seed数×T_Tmall`，不能直接套用 ShortVideoAD 时长 |
| lambda 四点 | 1–2 小时 | 全新四次，按相近停止轮数参考约 89.26h；复用 full/zero 后新增两次，参考约 45.15h |
| 稀疏用户离线分桶 | 0.5–1 工作日 | 0 次训练，CPU 分钟级，正式计时待跑 |

先做 full/等容量各三 seed，再给其余四个组件各一 seed，总计 **10 次**训练流程；不与上表前两行重复相加。若额外 lambda 两点也复用 full/zero，则共 12 次。仅作统一时长的算术参照，10F=225.74h（约 9.4 天），12F=270.89h（约 11.3 天），均假设串行独占同一组 4×A800、所有运行与 full 一样长；真实等容量 dense FFN、提前停止和独立 test 成本尚未计入。不能用 8×H800 的 8.63h 替代 `D`。

## 已完成的轻量验证

- loader 修复：8 个 Toy eval 路径、默认/True/False 三种训练 loader 配置、flake8、compileall、diff check；已提交 `38a90d2`。
- CPU 验证 FixedSoft buffer 的数值、冻结状态与 runtime multiplier=1。
- meta device 上实际构造模型并计算上述参数量。
- 微型 full、zero-prior、共享 FFN、普通 decoder 均完成有限 loss、反向梯度和带 KV cache 的生成验证；不是完整训练结果。
- Toy 冷/热缓存实测了 seed 重置差异。
- FixedSoft/FixedBias 的 cvr test 逐用户均值与 summary 全部一致。

复用价值最高的后续动作是补齐最终实验产物、实现两个组件开关和 seed 恢复、固定运行清单，再开展关键三 seed；raw 数据语义与上游合并口径需要独立材料，不能由这些补实验替代。
