# 在其他机器运行 FixedSoft rebuttal 实验

日期：2026-10-27。英文对应同名 `.md`。发布分支：`codex/rebuttal-fixedsoft-experiments-20261027`。

按优先级补齐剩余实验、一次完成训练和 validation/test 的最新指令，见[剩余实验命令清单](2026-10-27-remaining-experiment-commands.zh.md)。该清单默认复用本机已启动组，λ=0 先核验旧 FixedBias；ranking batch 统一为每卡 8。

远端 `dev` 包含另一项 V3 行为层级改动 `eadc431`。本实验分支保留当前 ShortVideoAD 任务对应、已验证的 FixedSoft 实现，新受控实验系列应统一使用此分支。本轮没有切换活动工作区分支、改写远端历史或合入 V3。

功能提交为 `aaf63c2`。从该提交干净导出的目录，仅使用导出的源码／配置／Toy 数据，通过了 23 项相关 CPU 测试、30 个 Python 文件的规定 flake8 和 compileall，不依赖本机忽略的运行文件。此检查不代表已实测目标机的生产 GPU 训练。

## 获取代码与输入

从新 checkout 的父目录执行：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --single-branch \
  --branch codex/rebuttal-fixedsoft-experiments-20261027 \
  git@github.com:wzf2000/GAMER.git GAMER
cd GAMER
git lfs pull --include='data/ShortVideoAD/**' --exclude=''
```

最后一条命令要求目标机已有 Git LFS。ShortVideoAD 的 inter、behavior、session、time 及 SID index JSON 使用 LFS；只有 pointer 的 checkout 无法运行。behavior-level/item 元数据和 FixedSoft tokenizer 资源已受版本控制；tokenizer 采用仓库内部相对符号链接，需使用支持符号链接的 checkout。无法访问 LFS 时，从有权限的来源复制对应、完全相同的已处理输入文件；不要替换成重建的 raw 或另一套 SID。

本次发布不含 `results/`、checkpoint、日志或 `.pkl` 缓存。评测已完成权重时需另行传输；新建缓存沿用原有 factor4/data seed，模型初始化前会恢复 model seed。严格比较跨机器结果时应记录输入的内容 SHA256；传输缓存时也应记录实际缓存文件 SHA256。现有轻量数据快照包含绝对路径／mtime，本身不能证明不同机器的数据内容相同。

依赖安装沿用仓库现有说明，生成 plan 不涉及安装。本机已测试环境为 Python 3.12.13、PyTorch 2.7.1+cu118、Transformers 4.51.0、Accelerate 1.8.1、DeepSpeed 0.16.5。效率对比前需对齐目标机 CUDA/PyTorch 环境；硬件与 backend 默认值可能不同。

## 在目标机重新生成清单

在目标机 GAMER Python 环境中，从仓库根目录执行；此命令只生成配置／命令：

```bash
python scripts/prepare_rebuttal_experiments.py \
  --suite handoff --seeds 42 \
  --dataset ShortVideoAD --data_path ./data --index_file .index.json \
  --base_model ./config/s2s-models/Qwen3TemporalHierarchicalFixedSoft \
  --tasks smb_explicit_decoder_4 --max_his_len 100 --model_max_length 1024 \
  --gpu 0,1 --port 2314 --precision fp32 \
  --per_device_train_batch_size 128 --per_device_eval_batch_size 128 \
  --gradient_accumulation_steps 8 --num_train_epochs 200 \
  --learning_rate 5e-4 --weight_decay 0.01 --warmup_ratio 0.04 \
  --patience 20 --temperature 0.7 --test_batch_size 64 --num_beams 20 \
  --eval_behavior cvr \
  --output_dir ./results/rebuttal/2026-10-27-p0-local
```

GPU 0/1、端口 2314 是示例，应改为目标机空闲资源。配方有效 batch 为 `128 × 2 × 8 = 2048`，FP32，按全行为 validation loss 选 checkpoint，与当前受控系列一致。最终 ranking 命令区分 validation/test，以 cvr 指标为主，不等于转化概率预测。这是明确记录的新统一基准，不是找回的投稿命令。

生成的 plan 属于目标机，记录本地 Python、checkout/data/config 路径和指纹；不要直接执行从本服务器复制且未重新生成的 plan／shell。`--reuse_plans` 只用于本地路径／revision 兼容的清单，不是自动跨机器导入器。复用本机已在跑的 Full/no-QKV，仍需另行转移并核验实际产物；新七组清单不会将其标为在目标机已完成。跨机器分配不同配置，避免重复运行。

## 查看状态并主动启动单项

```bash
python scripts/manage_rebuttal_queue.py \
  --plan ./results/rebuttal/2026-10-27-p0-local/plan.json \
  --gpu 0,1 --port 2314 \
  --output_file ./results/rebuttal/2026-10-27-p0-local/status-001.json \
  --output_script ./results/rebuttal/2026-10-27-p0-local/next-001.sh
```

此命令刷新状态、打印并写出单条运行命令，不启动训练。检查所选变体和 GPU 空闲后，准备好时执行生成的 `next-001.sh`。后续状态／脚本采用新文件名，已有产物不会被覆盖；阶段脚本也默认仅打印命令。单有 start manifest 或 epoch checkpoint 不会被判为成功完成。

Handoff 共七组，Full 共用为 λ=0.05，zero-prior 共用为 λ=0。先完成 P0-A（Full/no-QKV/no-gate/shared-FFN），再 P0-B（0/0.01/0.1）；已在跑的等容量对照仍属于独立可选实验。完整训练和大规模 ranking 使用目标机已获授权的资源范围。

参数探针、UID/目标导出、CSV 汇总及适用范围见[交付工具说明](../design/2026-10-27-rebuttal-delivery-tools.zh.md)。
