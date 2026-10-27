# Tmall baseline 结果来源与行为编码核查

英文对应：`2026-10-27-tmall-baseline-provenance.md`。核查日期：2026-10-27；实现 checkout：`6013055`。所给压缩包未记录历史训练代码版本。

**后续正确性核查：** `2026-10-27-sasrec-tiger-gap-audit.zh.md` 已验证当前 SASRec 路径存在 FFN 残差条件反向，以及训练样本／选模方式差异。下面的来源匹配不代表这些分数已经构成正确实现、条件对齐的 SASRec baseline；历史影响仍需原 checkout 和修正后训练核验。

## 与论文表格精确匹配的结果

论文三行均匹配用户提供的 `results_wzf.zip` 中的 **Behavior alipay**，不是 Merged Behavior。下列路径相对于包内 `results/Tmall-24-0.25/`。

| 论文行 | 结果路径 | HR@10 | NDCG@10 |
| --- | --- | ---: | ---: |
| SASRec | `smb_dis_sample/SASRec/result-smb_dis.json` | 0.07678938220887976 | 0.0533295220386152 |
| SASRec_B | `smb_dis_sample_diff/SASRec/result-smb_dis_diff.json` | 0.00647811660609891 | 0.004858649099901863 |
| TIGER | `smb/TIGER/results-smb-cid-shuffle-64.json` | 0.5809764575762364 | 0.4687695373957 |

保留四位小数后，六个数值均与截图一致。该包还保留 TIGER 的 6,329 条逐用户指标，所有指标均值都与汇总文件一致。

`results_ygk.zip` 中另有两份 TIGER 结果，需分开处理：

- `smb_explicit/TIGER/results-cid-shuffle-64.json`：alipay HR@10 为 **0.5732785849652559**，NDCG@10 为 **0.4711377563057159**。它是显式行为版本，不是论文 0.5810 / 0.4688 对应的那一行。
- `smb/TIGER/results-smb-cid-shuffle-64.json`：alipay HR@10 为 **0.5809659090909091**，NDCG@10 为 **0.46870820571939775**。其 NDCG 保留四位为 0.4687；wzf 包对应文件才是截图中的 0.4688。两包同一相对路径下的文件字节不同，不可互相覆盖。

## 已定位任务在当前实现中的含义

**SASRec，`smb_dis_sample`：** 使用 `SMBDisDataset(diff=False)`，将 `(behavior, item)` 映射为 `item + 1`。历史保留混合行为序列，但模型只使用原子 item embedding 和 position embedding，不显式区分行为。collator 也提供行为元数据，SASRec 的 forward／评分方法未消费这些字段。该任务不会将训练样本过滤为仅 alipay 行为。

**SASRec_B，`smb_dis_sample_diff`：** 使用同一 `SASRec` 类和源配置，数据改为 `SMBDisDataset(diff=True)`。令 `M = 原始最大 item ID + 1`，事件映射为 `behavior_index * M + item + 1`。Tmall 的枚举顺序为 click、collect、cart、alipay，因此同一 item 的四种行为对应独立原子 ID／embedding 行；collect/cart 即使层级并列，身份仍不同。这是联合的 behavior-item 词表，不是共享 item embedding 再加独立 behavior embedding；输出词表也扩展为四个 item-ID 区间。当前 `smb_dis_diff` 的 test 行为过滤通过 `item_range` 将评分限制在被请求的行为区间。

**TIGER，`smb`：** 使用 `SMBDataset`，每个事件仅表示为 item 的 token 序列，不增加行为 token。输入是混合历史，不是仅 alipay 历史。匹配的文件名明确标记 **CID、shuffle、chunk size 64**，脚本对应索引后缀为 `.index.cid.shuffle.chunk64.json`。压缩包没有历史索引内容／hash，因此不能证明其与当前某份索引文件完全相同，也不能把这一行描述为已确认采用 SID 的 TIGER。

**另一组 TIGER，`smb_explicit`：** 使用 `SMBExplicitDataset`，事件表示为 `<behavior_click>`／`<behavior_collect>`／`<behavior_cart>`／`<behavior_alipay>` 加上该 item 的 token 序列。T5 通过标准共享 token embedding 学习这些行为 token；encoder-decoder 评测时，decoder prompt 为起始 token 加请求的行为 token。TIGER 没有 TH prior 或 TH 专用的 Q/K/V 行为机制。

实现位置：`SeqRec/datasets/loaders/session_behavior_discriminative.py`、`SeqRec/datasets/discriminative/session_behavior.py:369`、`SeqRec/models/discriminative/SASRec/model.py:67`、`SeqRec/modules/model_base/seq_model.py:101`、`SeqRec/datasets/session_behavior/explicit.py`、`SeqRec/tasks/evaluation/helpers.py:93`。

## 采样、选模及仍缺失的证据

- 两个 `smb_dis_sample` 任务的训练仍使用 `SMBDisDataset`，源配置采用全词表 CE。`sample` 选择的是采样 **validation** 数据，不是采样训练目标。
- validation 默认先抽取 1,000 个负例，排除 test 前已观察的 item，并将负例行为设为 target behavior。collator 将正例放在前面，再将拼接候选截断为 **1,000 个总候选**。该元数据下 validation 过滤为 alipay；trainer 按最后一个请求指标选模，默认 NDCG@10。
- test 数据由独立的 `test_task` 控制。已找回文件名为 `result-smb_dis.json` 和 `result-smb_dis_diff.json`，与全量排序 test 路径对应。训练脚本不会自动将 `test_task` 与 `tasks` 同步，独立确认历史执行仍需原命令。历史 test split 的证据是任务命名及当前 loader 语义，没有保留 `.run.json` manifest。
- 当前源默认配置：两个 SASRec 均使用 `config/dis-models/SASRec/config.json`，2 层、2 heads、hidden 128、FFN 256、dropout 0.5、CE；TIGER 使用 `config/s2s-models/TIGER/config.json`，4 层 encoder／4 层 decoder、hidden 128、6 heads、FFN 1024、dropout 0.1。包中没有这些 baseline 的 checkpoint、config、训练参数、调参日志或运行 manifest，不能据此认定源默认值就是历史最终值。
- 当前判别式 SMB loader 要求最高层行为唯一。`Tmall-24-0.25` 的 alipay=2；V2 元数据有三个最高层行为，不能直接替换后沿用该 loader。

轻量 CPU 检查已验证 item-ID 编码、SASRec 不使用额外行为字段、diff 行为区间评分 mask、TIGER 事件字符串编码，以及 validation 的 1,000 总候选拼接。本次未修改源码，也未启动训练或评测任务。

## 本地保留证据

已将所选文件逐字节复制到 `results/rebuttal/2026-10-27-tmall-baseline-provenance/`，保留独立的 `results_wzf/` 与 `results_ygk/` 子目录。`audit.json` 记录包／文件 SHA256、精确匹配及核验范围。运行证据保留本地，不纳入 Git。
