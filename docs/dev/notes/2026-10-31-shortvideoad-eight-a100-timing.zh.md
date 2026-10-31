# 八张 A100 上 ShortVideoAD 完整 GAMER 的单 epoch 计时

英文对应：`2026-10-31-shortvideoad-eight-a100-timing.md`。日期：2026-10-31，Asia/Shanghai。

## 采用的测量与 94 轮估算

作者明确接受已完成的 **首轮训练及其 Evaluation** 作为计时结果，排除之后的收尾等待，并指定按 **总计94个 epoch** 外推。没有再次运行完整数据 epoch。

| 环节 | 首轮测量 | 按94轮条件估算 |
| --- | ---: | ---: |
| Training，含 optimizer／DDP 和 batch 准备 | 约228秒 | 约5小时57分12秒 |
| 全行为 validation-loss Evaluation | 136.9415秒 | 约3小时34分33秒 |
| Training＋Evaluation | 约365秒（6分05秒） | **约9.53小时（9小时32分）** |

计算为 `(228 + 136.9415) × 94 / 3600 = 9.529028...小时`，八卡共约 **76.23 A100 GPU小时**。这是首轮测量的线性外推，不是实测94轮或稳定阶段多轮均值。训练耗时来自原生 tqdm 的整数秒计时，不能因计算保留小数而声称亚秒级测量精度。此口径不计初始化、checkpoint 保存／加载、收尾及独立 ranking。

## 实际配方

原命令指定 ShortVideoAD、`Qwen3TemporalHierarchical`／`config/s2s-models/Qwen3TemporalHierarchicalFixedSoft`、`smb_explicit_decoder_4`、original `.index.json`、history100、lr5e-4、最多200轮、每卡batch128、累积8、warmup0.04、patience20、temperature0.7。根据作者此前资源记录，原运行使用四卡，即有效batch `4×128×8=4096`。

本次使用本机 GPU0–7，即 **8×NVIDIA A100-SXM4-80GB**；每卡128不变，累积改为 **4**，有效batch仍为 **4096**。Training 与 Evaluation 的每卡batch均为128。保留seed42、最大长度1024、FP32（`fp16=False、bf16=False`）、SDPA、AdamW、cosine、weight decay0.01、全行为validation、完整Q/K/V条件化／gate／位置专家，以及冻结prior λ0.05。实际可训练参数 **29,935,104**。训练缓存 **238,270** 条、validation缓存 **193,316** 条；original SID的SHA256为 `0e71e2064bf2c0f95e639e511f4c81a00ef839ec12cd1e2e7ad43e485ab6e62d`。

可选 `--benchmark_epochs 1` 保留 `--num_train_epochs 200`，在首轮训练及按计划Evaluation／保存之后停止。实际optimizer更新 **59次**；保留的scheduler预算 **11,600步**、warmup **464步**。用户指定的94轮只用于时长外推，没有将计时运行的学习率调度改成计划1轮或94轮。

环境为 PyTorch2.7.1+cu118、编译CUDA11.8、cuDNN90100、Transformers4.51.0、Accelerate1.8.1、DeepSpeed0.16.5、NVIDIA driver550.144.03。W&B为offline，OMP／MKL／OpenBLAS线程数均为1，未设置 `CUDA_LAUNCH_BLOCKING`。硬件／互联拓扑、准确环境及命令均有运行记录。

## 证据与收尾问题

运行根目录：`results/rebuttal/2026-10-31-shortvideoad-8xa100-epoch/`。

- Evaluation前原生进度为 `59/11600 [03:48<...]`。
- **21:42:36** 的原生Evaluation记录为 `eval_runtime=136.9415、epoch=1.0`；之后累计进度为 `59/11600 [06:05<...]`。
- `checkpoint/checkpoint-59/trainer_state.json` 确认epoch1、global_step59、计划max_steps11600，以及完整validation-loss记录。
- `accepted-epoch-summary.json` 与 `accepted-94-epoch-estimate.csv` 保存本次采用的阶段耗时、94轮条件估算、命令／配置／cache／索引绑定，以及精度和口径限制。
- 原始证据为 `pipeline.log`、`checkpoint/run_manifest.start.json`、`hardware-during-training.json`、`library-runtime.json`。

Training、Evaluation和首次checkpoint已完成，但原进程在checkpoint后收尾阶段等待，随后被中断。原 `job.finish.json` 保留非零退出码，不能将其改写为完整pipeline正常结束。采用的计时排除了这段等待，也不依赖最终root权重成功导出。

八卡tiny模型的受控验证复现了Transformers4.51首次checkpoint竞态：同步前rank0已有新best checkpoint路径，而rank1–7为 `None`；首轮立即停止时，可能因此进入不同的load-best barrier／store-FLOPs路径。可选计时callback现已同步checkpoint完成和best路径，再进行最终加载。强制延迟保存的八rank回归已正常结束，见 `checkpoint-race-verification.json` 和 `smoke-checkpoint-sync.log`。四项CPU callback测试（包括保留200轮调度的真实Trainer）通过，规定flake8和compileall通过；默认不传 `--benchmark_epochs` 时不会创建该callback。

Tiny回归用于验证修复，不参与完整模型耗时外推。按用户要求，没有重复完整数据测量。首次完整运行的计时功能版本为 `ccc09b2`；后续收尾同步修复不改变首轮原始计时日志。
