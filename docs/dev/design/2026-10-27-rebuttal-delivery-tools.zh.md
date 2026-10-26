# Rebuttal 交付工具：目标、参数、CSV 与队列

日期：2026-10-27。基础提交 `38a90d2`，保留工作区已有修改。英文对应同名 `.md`。本文记录 2026-10-27 交接审计中四项缺口的实现。本轮未 commit、完整训练、安装依赖或生产数据评测。

## 兼容性与运行任务

本轮没有修改模型、训练入口、collator、dataset builder、缓存或已有实验配置。正在运行的训练已在本轮修改前加载其代码，Full（GPU 1/5）、plain_matched（2/6）和 no-QKV（4/7）均保持运行。新探针均在 CPU 上使用一个 Torch 线程；GPU 已被占用，因此本轮没有执行 GPU 验证。

评测新增路径由默认 false 的 `--export_user_targets` 控制。其他已有 plan suites 保持其启动脚本行为；handoff 是新的显式 suite，其阶段脚本默认仅打印待执行命令。旧结果汇总字段保留，CSV 为可选产物。

## 真实目标与 UID 对齐

`SeqRec/evaluation/smb_targets.py` 提供可选的内存 dataset 副本。补齐旧 validation UID 前，会将缓存中完整目标表示／行为、历史与按顺序的来源事件逐项核对。不兼容时拒绝，而不改写缓存。支持 `test` 和 `valid_test` ranking；loss-only validation 不导出目标。

在原有 `main.py test_SMB_decoder` 命令后添加 `--export_user_targets` 即可，新 handoff ranking 命令自动包含它。旧 metric 数组及 `.uids.json` 保留；每个行为增加 `user_level_metrics_<behavior>.targets.json`，包含 UID、真实 target item ID、事件／session 下标、SID／行为表示、逐用户指标及共享 SID 的候选 ID。重复事件按原顺序保留，item ID 来自 held-out 来源事件，不依靠反查 SID。指标、UID sidecar、索引及评测 receipt 使用 SHA256 绑定；大型来源文件仅记录路径／大小／mtime 快照，不宣称内容哈希。

开启选项不改变候选、prompt、历史和 labels。此选项下旧 validation 补齐 UID 后，DDP 指标使用已有 UID 去重路径；默认路径保持兼容。

## 独立参数审计

`scripts/audit_rebuttal_parameters.py` 实例化独立 resolved TH 模型，也可读取显式给定的已完成 checkpoint。它不使用 optimizer，不修改训练产物；默认 CPU。默认探针明确标为覆盖各行为的合成输入，也支持通过 `--batch_file` 读取真实 collated JSON batch，包括非 tensor metadata。`--temperature` 默认采用本轮配方的 0.7，并写入报告。

```bash
python scripts/audit_rebuttal_parameters.py \
  --config results/rebuttal/2026-10-26-p0/configs/full/config.json \
  --device cpu --seed 42 --temperature 0.7 \
  --output_file results/rebuttal/full-parameters-new.json
```

报告区分去重的注册／可训练参数、buffer、有非空输入的执行、与 loss 连通的 tensor、非零梯度 tensor，以及观察到的零梯度／未执行项。`active_participating` 定义为非空调用与 loss 连通的交集，包含梯度为零的 tensor。所有数值按整个 tensor 计数，embedding 不等于访问行数；零梯度或未观察到都不能证明全局结构不可达。保留被审计实例的 mode 和 gradients，不从单个 batch 推断全局无效容量。

七组生产词表（23,675 token、随机 seed 42、temperature 0.7、与训练一致的 masked padding、eval 模式启用梯度）的探针位于 `results/rebuttal/2026-10-27-p0-supported/parameters-final/`。注册参数仍为 Full 29,935,104；no-QKV 29,922,816；no-gate 29,672,960；shared-FFN 12,240,384；各 λ 点 29,935,104。该探针中所有注册 tensor 都被调用并与 loss 连通，但 Full 有 27 个零梯度 tensor；报告另记录 batch tensor 摘要。报告明确区分这些口径，不宣称其全局不参与。这些是合成诊断，不能当成已训练模型效果。

## JSON 与 CSV 结果汇总

`scripts/summarize_rebuttal_results.py` 新增 `--csv_file`。长表 CSV 提供每次运行实际指标、相同 seed 的 reference 值、绝对差和相对百分比；缺失项有明确状态，数值留空；reference 为零时相对百分比留空。原 JSON 均值／样本标准差及配对差值保留，单 seed 标准差为 null。

```bash
python scripts/summarize_rebuttal_results.py \
  --plan results/rebuttal/2026-10-27-p0-supported/plan.json \
  --split test --behavior cvr --reference full --allow_partial \
  --output_file results/rebuttal/cvr-test-summary-new.json \
  --csv_file results/rebuttal/cvr-test-summary-new.csv
```

`--allow_partial` 显式允许全部缺失的报告。已生成七组的 `summary-cvr-test-pending.*`，其中没有真实生产评测分数；没有使用 test 做选择。

## Handoff suite、复用与状态快照

`prepare_rebuttal_experiments.py --suite handoff --seeds 42` 恰好生成七组：Full、no-QKV、no-gate、shared-FFN、zero-prior、λ=0.01 和 0.1。Full 对应 λ=0.05，zero-prior 对应 λ=0；记录阶段、resolved config 差异、model/data seed 和复用依据。data seed 42 指既有 random factor4 builder，不是 policy augmentation 的 seed 开关。

用 `--reuse_plans <plan.json> ...` 复用既有 output/config/result 路径。复用检查代码 revision、resolved config SHA、数据快照、配方和设备数量；有 start manifest 时进一步检查实际 seed、参数量、dataset/script/HF 设置、精度及缓存。dirty workspace 的历史精确实现仍需结合保留的源码快照，不能只凭相同 Git HEAD 宣称任意工作区改动等价。

实际清单是 `results/rebuttal/2026-10-27-p0-supported/plan.json`，复用了此前 P0/lambda plans 的七组路径，其中 Full 和 no-QKV 已在跑。阶段脚本调用 manager 打印下一条命令，不直接启动。

```bash
python scripts/manage_rebuttal_queue.py \
  --plan results/rebuttal/2026-10-27-p0-supported/plan.json \
  --parameter_audits results/rebuttal/2026-10-27-p0-supported/parameters-final/*.json \
  --output_file results/rebuttal/status-new.json
```

可加 `--output_script <new.sh>` 保存单条下一项命令，供之后主动执行；`--gpu` 可以改卡号，但不能改变设备数量。工具不推断设备空闲。快照中下一项是 no-gate，但 GPU 4/7 已被 no-QKV 占用；本轮没有执行生成的训练命令。

`refresh_status` 区分运行 PID、已启动但存活状态未确认的目录、最终根目录权重加包含 `train_runtime` 的 trainer state、缺失结果和已校验的 result receipt。只有 epoch checkpoint 或 start manifest 不足以判为完成；进程命名空间不可见时保留不确定状态。B 阶段等待 A 阶段训练完成；ranking 队列还要求对应的 A 阶段结果齐全。配置／数据／产物异常阻止推进。已有启动目录不会被静默重启或覆盖。

## 验证与范围

- 23 项相关测试通过，覆盖默认选项／模型路径、seed／shifted loss、目标 collision／重复、旧缓存拒绝、参数图／零梯度口径、复用、阶段约束、handoff 错误训练任务拒绝，以及 CSV 缺项／实际差值。
- 实际 Toy validation/test ranking 在 CPU 上完成，开启／关闭导出后的解析指标完全一致，复制缓存的校验值未变。最新产物位于 `/tmp/gamer-four-support-cpu-n9oj87up/`，真实 batch 审计位于 `/tmp/gamer-four-support-cpu-fh4tgk9d/`；仅测试中注入 CPU device 和零 DataLoader worker，没有改变生产 device 路径。
- 使用真实 Toy causal collator batch 和已完成 Toy checkpoint，验证可选参数审计路径；另完成七组最终词表的独立 CPU 合成探针。
- 按规定使用 `--max-line-length=500 --ignore=F401,E203,W503,F841` 执行 flake8，并运行 compileall 和生成 shell 的 `bash -n`；日志和源码证据随新 plan 保留。
- 未运行完整生产 ranking，也未在 GPU 验证新增交付路径。本轮保持模型／训练实现及生产数据／缓存／配置指纹不变；三组用户训练持续推进。
