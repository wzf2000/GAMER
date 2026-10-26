# Rebuttal 交接就绪核查与下一项运行

日期：2026-10-27，Asia/Shanghai。工作区：`dev`，`38a90d2` 加已有未提交实验支持。英文对应同名 `.md`。

后续更新：下述四项实现缺口现已补齐，见[交付工具说明](../design/2026-10-27-rebuttal-delivery-tools.zh.md)。下文保留当时的审计快照。

本次按所给 2026-10-27 交接文档核查。新队列替代此前优先多 seed／等容量的队列：先 P0-A Full/no-QKV/no-gate/shared-FFN，再 P0-B λ=0/0.01/0.05/0.1；Full 共用为 λ=0.05。已有训练保持运行，没有启动新的完整训练或大规模评测。本次补充审计、注册表和命令，没有修改生产 Python 或已有配置。

## 已具备的能力

七组均已有真实模型和配置入口。组件组相对 Full 各只改变一个声明的配置键：

| 变体 | 配置变更 | 总参数 | 可训练参数 |
| --- | --- | ---: | ---: |
| Full | 无 | 29,935,104 | 29,935,104 |
| no-QKV | `th_use_behavior_qkv=false` | 29,922,816 | 29,922,816 |
| no-gate | `th_use_output_gate=false` | 29,672,960 | 29,672,960 |
| shared-FFN | `sparse_layers_decoder=[]` | 12,240,384 | 12,240,384 |
| λ=0 | `th_relation_bias_soft_scale=0` | 29,935,104 | 29,935,104 |
| λ=0.01 | `th_relation_bias_soft_scale=0.01` | 29,935,104 | 29,935,104 |
| λ=0.1 | `th_relation_bias_soft_scale=0.1` | 29,935,104 | 29,935,104 |

参数量来自扩展至 23,675 词表后的实例化模型。冻结 prior 是 buffer，不是注册参数；这些数值不能直接证明给定 loss 下的活跃参数数量。

no-QKV 保留普通投影、normalization、output gate、FFN、prior 和 TH 第 2–5 层。no-gate 只取消 attention output 乘法。shared-FFN 保留 `behavior_injection_decoder=[0,1,2,3,4,5]` 和已有 injection 实现，不需要重新解耦模型。三组组件消融的 prior 均冻结，λ=0.05，额外运行倍率为 1。

seed/cache 隔离、开始运行 manifest、validation/test 分离的 ranking 命令、test 逐用户指标的 UID sidecar，以及含 Full 差值的 JSON 汇总均已有实现。旧 FixedBias 仍只是待核验复用候选，本清单未计入任何已完成 λ=0 运行。交接文档确认的“0.05 为经验预设”属于作者提供的来源说明，不能由可执行代码独立证明。

## 仍需实现或补证的内容

1. **逐用户导出真实目标 item ID。** `SeqRec/tasks/evaluation/base.py::_save_user_metrics` 已保存指标数组和 UID/校验 sidecar，但没有原始 target item ID。应从实际 held-out 事件导出 UID、行为、split、原始 item ID 和 SID 字符串，保留重复目标并明确 SID collision。这可以在最终评测前补，不需要重训。此外 `BaseSMBDataset._process_valid_test_data` 的 validation ranking 记录没有 UID；只改 builder 不会更新旧缓存，需兼容处理缓存或独立导出。
2. **实际参与计算的参数统计。** 现有 helper/manifest 记录注册的可训练参数量。还需为最终词表和训练 loss 定义执行／梯度审计，区分不可达权重与单个 batch 的零梯度，并单列 buffer。注册表的 `active_participating=null` 明确表示未测量。
3. **结果 CSV。** `scripts/summarize_rebuttal_results.py` 已生成各组指标及相对 Full 配对差值的 JSON，但没有指标 CSV。需补实际值、差值、缺项的 CSV 导出；本次注册表 CSV 仅包含配置、状态和参数量。
4. **可复用的阶段队列／状态工具。** 本次已经生成统一七组注册表和分阶段脚本，但通用、受版本控制的 planner 仍沿用此前 components/seeds/lambda suites。之后可加入专用 handoff suite／状态刷新，自动维护阶段顺序、Full/0.05 去重和完成检测。目前注册表是带时间的审计快照，不是持续监控。

这些缺口不阻塞 no-QKV 训练。历史投稿训练来源、最终 test 分数和收敛情况不能由当前中间日志推断。本轮采用明确记录的新统一配方，不与未经核验的历史 Full/zero-prior 拼成受控对照。

## 已生成产物与当前任务

产物目录：`results/rebuttal/2026-10-27-handoff-audit/`：

- `registry.json`：七组配置、完整 resolved config／差异、代码快照、数据／缓存指纹、model/data seed、选择协议、预算、输出路径和观察状态；也兼容现有结果汇总器。
- `registry.csv`：配置／状态／参数量摘要。
- `P0-A-{train,validation,test}.sh` 和 `P0-B-{train,validation,test}.sh`：真实分阶段命令，GPU 4/7、端口 2316。训练脚本跳过已在跑的 Full，各命令均检查输出不存在。ranking 应在对应训练完成并保存最终 best model 后执行。
- `run_next_no_qkv.sh`：只启动下一项，不连带启动后续队列。
- `gamer_handoff_audit_20261027.py` 和验证日志：生成器及本次核查证据。

核查时 Full 在 GPU 1/5，plain_matched 在 2/6。最新保存状态为 Full 第 32 epoch、validation loss 2.052139，plain_matched 第 25 epoch、validation loss 2.062975；日志持续推进，未观察到训练错误。这是不同训练阶段，不能作为最终性能对比。GPU 4/7 空闲，端口 2316 未被占用。

下一项为 no-QKV、seed 42，保留相同 ShortVideoAD original SID `.index.json`、`smb_explicit_decoder_4`、训练缓存 238,270 样本和验证缓存 193,316 样本。历史长度 100；模型长度 1024；dropout 0.2；FP32/SDPA；每卡 batch 128；双卡；梯度累积 8；有效 batch 2048；学习率 5e-4；warmup 0.04；weight decay 0.01；temperature 0.7；最多 200 epoch；early-stopping patience 20。单次训练仍按全行为 validation loss 选 checkpoint，最终 core ranking 使用 `smb_explicit` test，以 cvr HR@10/NDCG@10 为主。

从仓库根目录执行：

```bash
bash results/rebuttal/2026-10-27-handoff-audit/run_next_no_qkv.sh
```

输出为 `results/rebuttal/2026-10-26-p0/no_qkv/seed_42/checkpoint/`。保留此前日期以保持同一受控实验系列。之后依次 no-gate、shared-FFN、λ=0、0.01、0.1。额外 seed／层级／core-only／等容量新训练按交接文档后置；已有 plain_matched 保持运行。

## 验证

- CPU 上已有组件／seed／plan 共 10 个测试通过。
- 七组微型模型通过未来 token 不影响过去、右侧 padding 不变、position/behavior 分组边界及保存重载输出完全一致的检查；shared-FFN 的 injection 梯度非零。这些补充检查以内联方式执行，日志随审计保存。
- 在实际空闲的 A100 80GB GPU 4/7 上，`tests/smoke_rebuttal_ddp.py` 完成 Full、no-QKV、no-gate、shared-FFN、zero-prior、plain 六组的双卡两步前后向及 cached/uncached beam 一致性检查；测试进程已结束并释放设备。
- 此前代码快照的 23 个源文件哈希仍与当前工作区一致；本次没有改变生产源码／配置。
- 全部 22 个改动／新增项目 Python 文件通过 `--max-line-length=500 --ignore=F401,E203,W503,F841` 的 flake8；七个生成脚本通过 `bash -n`。本次未执行真实完整训练或 ranking 评测。
