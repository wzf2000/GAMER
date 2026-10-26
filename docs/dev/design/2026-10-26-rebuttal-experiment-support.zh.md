# TH-FixedSoft 补实验开发支持与运行说明

日期：2026-10-26。基础核查版本：`38a90d2`；本文对应此后的工作区实现。英文对应文件为同名 `.md`。实现顺序遵循用户给出的 1→6；没有运行真实数据完整训练。

四项问题的可转交说明见 `../notes/2026-10-26-review-agent-handoff.zh.md`。代码支持不能证明历史投稿实验实际采用了相同设置。

## 1. P0 组件消融与等容量对照

在 `Qwen3TemporalHierarchicalAttention` 中增加两个 config 选项，默认均为 true：

- `th_use_behavior_qkv`：关闭后删除行为 Q/K/V embedding 的参数与加法，保留 prior、gate、FFN。
- `th_use_output_gate`：关闭后删除 attention output gate 的参数与乘法，保留 Q/K/V 条件化、prior、FFN。

默认路径保留原有参数名与计算行为。开关经 checkpoint config 保存，训练、生成和重新加载一致。这里关闭的是 attention output gate，不是 SwiGLU FFN 的 gate projection。

配置生成器 `scripts/prepare_rebuttal_experiments.py` 提供 `components` suite，按顺序生成 full、plain_matched、no_qkv、no_gate、shared_ffn、zero_prior：

- 以现有 FixedSoft 为 full；所有组保持数据、索引、训练任务和默认 dropout 一致。
- 普通 decoder 清空 TH attention、token-position expert 和 FFN behavior-injection 的 layer 列表；保留同一代码路径中的 causal mask 与 dropout，再按实际词表/模型参数量自动选择最接近的整数 FFN 宽度。
- `shared_ffn` 只清空 `sparse_layers_decoder`，保留 FFN 行为条件化；不是使用另一个 backbone 的 dense 配置。
- `zero_prior` 只将 frozen soft prior scale 设为 0；仍有 Q/K/V 条件化、gate 和位置专家。

当前 ShortVideoAD 索引下 full=29,935,104，等容量普通 decoder=29,935,616（FFN width=3501）；差 512。参数匹配不代表 FLOPs/速度匹配。默认参数容差为 0.1%；微型测试模型的整数步长较大时可显式放宽。

## 2. P0 多 seed、执行清单与汇总

数据准备后、模型初始化前恢复用户指定 seed，修复冷缓存增强构建调用 `set_seed(42)` 影响模型初始化的问题。数据增强仍保持其既有 seed；命中缓存/重新构建缓存不再改变同一训练 seed 的模型初始化。

在 GAMER Python 环境中，从仓库根目录准备一个优先级清单：

```bash
python scripts/prepare_rebuttal_experiments.py \
  --suite components --seeds 42 43 44 --key_seeds_only \
  --dataset ShortVideoAD --data_path ./data --index_file .index.json \
  --tasks smb_explicit_decoder_4 --max_his_len 100 --model_max_length 1024 \
  --gpu 0,1,2,3 --precision fp32 \
  --per_device_train_batch_size 128 --per_device_eval_batch_size 128 \
  --gradient_accumulation_steps 4 --num_train_epochs 200 \
  --learning_rate 5e-4 --warmup_ratio 0.04 --patience 20 \
  --output_dir ./results/rebuttal/2026-10-26-p0
```

这只生成配置、JSON manifest 和 shell 命令，**不启动训练**。目录已存在时拒绝覆盖。上述超参是明确的新实验计划，不是找回的投稿最终命令；应在实际启动前与最终设置对齐。

产物：`configs/`、`plan.json`、`run_train_key.sh`（full/等容量各三 seed）、`run_train_remaining.sh`（四个消融各首个 seed）、`run_train.sh`（全部十次）、`run_validation.sh`、`run_test.sh`。三个阶段的输出文件分别命名，生成脚本检查已有 checkpoint/结果并拒绝覆盖。

脚本固定为生成时的 Python 解释器；多 GPU 使用该解释器的 `torch.distributed.run`。`batch` 参数均为每卡值；例中有效 batch 为 `128×4×4=2048`。随机增强仍为原视图加最多四个视图，不能称 no augmentation。

只准备关键两组可用 `--suite seeds`，默认 seeds=42/43/44；`components` 默认单 seed=42。

实际训练由用户确认后执行相应 `bash .../run_train_key.sh`。新计划自动传 `--record_run_manifest`：训练开始前记录有效 model/HF training config、参数量、index SHA256、seed、实际 attention backend、参数 dtype、设备信息和代码版本。AMP 设置单独记录，参数 dtype 不能代替 autocast 精度。训练开始记录不等于训练成功。

索引和行为层级元数据使用 SHA256；大型原始 JSON 与缓存记录路径、大小、mtime，缓存另记录实际样本数。汇总器检查各运行与计划的数据快照，以及运行之间的缓存一致性；这些轻量快照不等于对大文件做完整内容哈希。

SMB ranking 另存 `.run.json`，记录实际 split、任务、候选索引指纹、checkpoint、训练记录指纹和结果指纹。汇总器会检查这些绑定，不能把 validation 文件当 test 汇总。

```bash
python scripts/summarize_rebuttal_results.py \
  --plan ./results/rebuttal/2026-10-26-p0/plan.json \
  --split test --behavior cvr --reference full \
  --output_file ./results/rebuttal/2026-10-26-p0/summary-cvr-test.json
```

默认要求清单内结果齐全；阶段性核查需显式加 `--allow_partial`，报告会列出缺项。报告每组 mean、样本标准差（ddof=1）、seed 集合，并只对相同 seed 计算 variant−reference 的配对差值。其余消融仅一个 seed 时不会伪造三 seed 方差。

## 3. P0 core-only / raw / consolidated

提供 `scripts/prepare_rebuttal_history_views.py`。它需要真实 raw 与用户给定的 consolidation provenance，不能从最深状态重建 raw。

输入格式：raw 和 consolidated 均为 JSONL，每行事件包含：

```json
{"uid":"u1","event_id":"e1","item_id":0,"behavior":"click","session_id":0,"timestamp":"2026-01-01 00:00:00","available_at":"2026-01-01 00:00:00"}
```

UID/event ID 为字符串，item/session ID 为整数，时间为无时区、秒精度的规范格式。`timestamp` 是适配后事件发生时间，`available_at` 是状态可用于历史的时间；字段业务语义必须由真实上游解释，适配器不替用户推断。

consolidated 每行另有非空 `source_event_ids`，指向同一用户、item、原 session 的实际 raw 事件。每个 raw event 必须且只能被一个 consolidated event 覆盖，禁止伪造来源或将状态回溯到来源证据之前。

共同 split manifest：

```json
{
  "raw_provenance":"实际原始数据来源与快照版本",
  "core_behavior":"cvr",
  "users":{
    "u1":{"valid_session":2,"test_session":3,
          "valid_target_event_ids":["e5"],"test_target_event_ids":["e7"]}
  }
}
```

要求末两个原始 session 是共同 validation/test，held-out core target event ID 与顺序一致，候选索引完全共享。core-only 从 raw 过滤 core 事件，保留共同 held-out core 目标。无 core 训练历史的用户仍保留在评测中；相应空历史 collator 已支持。

当前静态 SMB 格式不能准确表达“先前事件在预测后才回填，但此前仍要作为标签”的所有情况。若 `available_at` 跨越后续预测 cutoff，适配器拒绝导出，需另做 query-time 视图适配；不会默认将迟到状态放入历史。

真实输入到位后执行（`inputs/rebuttal/` 中的文件需由上游提供）：

```bash
python scripts/prepare_rebuttal_history_views.py \
  --raw_file inputs/rebuttal/raw.jsonl \
  --consolidated_file inputs/rebuttal/consolidated.jsonl \
  --split_manifest inputs/rebuttal/split.json \
  --behavior_levels data/ShortVideoAD/ShortVideoAD.behavior_level.json \
  --index_file data/ShortVideoAD/ShortVideoAD.index.json --index_kind SID \
  --dataset ShortVideoADViews --max_his_len 100 \
  --output_dir ./results/rebuttal/2026-10-26-history-inputs
```

产物为 `core_only/`、`raw/`、`consolidated/` 三个 data_path，数据集名相同；索引统一输出为 `<dataset>.index.json`，manifest 另记真实 SID/CID 类型与源指纹。`coverage.json` 逐用户报告 validation/test 历史与可见事件数、history token 数、raw 来源事件覆盖、core 数和时间跨度；history token 数不含 prompt/target。

给每组相同超参，并使用 `--validation_behavior cvr --eval_behavior cvr` 准备训练与评测，避免 validation loss 的目标行为集合随视图改变。若需比较同历史覆盖下成本，应根据 coverage 再指定各组窗口；当前工具不会声称相同事件窗口等于相同 raw 覆盖。生产 raw 尚缺，因此只完成合成数据验证。

## 4. P1 弱层级排序干预

新增 `th_prior_behavior_ranks`，按 `behavior_order` 为各行为身份指定 prior rank。它只改变 frozen soft table 中行为与行为的偏置；编号 0 的原有行/列、行为 token/embedding/FFN、core 目标、增强数据全部保持一致。仅允许 frozen table + soft init，拒绝错误长度、负数、非有限数或其他 relation 类型。

不要修改原数据 `behavior_level.json` 来实现该干预。提供示例 `config/rebuttal/tmall-hierarchy-maps.json`：并列辅助层级、反向辅助排序、扁平辅助层级；core alipay 保持目标身份。这检验 fixed prior 的排序/并列结构，不等于去掉全部行为条件化。

```bash
python scripts/prepare_rebuttal_experiments.py \
  --suite hierarchy --dataset Tmall-24-0.25-V2 \
  --index_file .index.cid.shuffle.chunk64.json \
  --hierarchy_maps config/rebuttal/tmall-hierarchy-maps.json \
  --seeds 42 --output_dir ./results/rebuttal/2026-10-26-tmall-hierarchy
```

生成的 full 是原 identity 顺序 prior；其余组修改 table 的初始化，复用同一数据/cache。旧历史 Tmall weak-level 文件并不证明历史模型 prior 已使用并列 rank。

## 5. P1 lambda 网格

使用相同生成器，保持 table 冻结、runtime multiplier=1；lambda 是 `th_relation_bias_soft_scale`，不是 loss weight。不要直接传未注册的同名 CLI 参数给 `main.py`。

```bash
python scripts/prepare_rebuttal_experiments.py \
  --suite lambda --lambdas 0 0.01 0.05 0.1 --seeds 42 \
  --output_dir ./results/rebuttal/2026-10-26-lambda
```

如 P0 的 full/zero 已在完全相同条件下完成，改为 `--lambdas 0.01 0.1` 只准备新增两点。lambda=0.05 对照选择 `--reference lambda_0.05`；选择配置应使用 validation，test 仅做最终报告。

## 6. P1 稀疏用户离线分桶

新 SMB test 逐用户结果保持旧 metric 数组格式，同时新增 `.uids.json` sidecar，保存精确 UID 顺序、split、历史窗口并通过 SHA256 绑定 metric 文件。离线脚本检查这些字段，直接聚合保存的多目标 NDCG/Recall，不用最佳 rank 重算，也不重新加载模型。

```bash
python scripts/analyze_sparse_metrics.py \
  --ours ./results/ShortVideoAD/smb_explicit_decoder_4/Qwen3TemporalHierarchicalFixedSoft/results-smb_explicit-original/user_level_metrics_cvr.json \
  --baseline ./results/ShortVideoAD/smb_explicit_decoder_4/Qwen3TemporalHierarchicalFixedBias/results-smb_explicit-original/user_level_metrics_cvr.json \
  --behavior_file ./data/ShortVideoAD/ShortVideoAD.SMB.behavior.json \
  --session_file ./data/ShortVideoAD/ShortVideoAD.SMB.session.json \
  --behavior cvr --split test --max_his_len 100 --count_scope visible \
  --thresholds 1 3 6 --bootstrap_samples 1000 --seed 42 \
  --legacy_lexicographic_order \
  --output_file ./results/rebuttal/2026-10-26-sparse-cvr.json
```

旧文件没有 UID sidecar；上述命令显式采用“按当前 writer 字典序排序、且用户集合与恢复 split 相同”的假设，输出记录限制。没有此选项或匹配的 UID sidecar 时拒绝连接；无法证明旧顺序时需重新导出。新文件不需要 legacy 选项。

默认桶为 0、1–2、3–5、6+。`visible` 按模型最后 N 项历史中的 core 数量，`full` 按完整预测前缀；两者不可混称。输出桶人数、两模型原指标均值、配对差值和可复现 bootstrap 95% 区间。该脚本不提供多次训练 seed 的不确定性估计。

## 验证与交付范围

已有测试覆盖默认兼容、组件梯度与 shifted loss、KV cache、保存重载、seed 冷/热缓存、参数匹配、同 session provenance、目标/候选一致性、空 core 历史、迟到拒绝、UID/hash 对齐、可见历史分桶与配对汇总。另进行 Toy 微型训练到评测的实际链路验证。

Python 使用项目规定的 flake8 参数，执行 compileall；生成的 shell 脚本执行 bash -n。经用户授权，在当时空闲的 GPU 7（A100 80GB）完成真实 CLI 的 Toy 单 epoch 训练、validation/test ranking 和带 manifest 的汇总；在 GPU 5/7 完成六组变体的双卡 DDP 两轮前向/反向及 cached/uncached beam generation 一致性检查。可复用检查入口为 `tests/smoke_rebuttal_ddp.py`，运行前必须重新检查空闲卡。

生成的实际 P0 清单位于 `results/rebuttal/2026-10-26-p0/`，共十次；hierarchy 清单四次，lambda 补点清单两次，均未启动。它们显式设置 GPU 5/7、每卡 batch=128、梯度累积=8，有效 batch=2048；这是本次候选计划，不是历史最终设置。生产数据完整训练和大规模 ranking 尚未运行。原始数据业务定义与历史最终实验产物仍需外部补齐。
