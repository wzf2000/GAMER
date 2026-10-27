# 剩余实验的一次性运行命令

日期：2026-10-27。英文对应同名 `.md`。使用分支 `codex/rebuttal-fixedsoft-experiments-20261027` 的最新提交。本文件所有模型训练命令均供用户在目标机主动执行；本轮只验证命令生成，没有启动完整训练。

## 状态与顺序

当前服务器已启动 Full、no-QKV、等容量 plain_matched 的 seed 42，尚未保存最终根目录权重；不要在另一台机器默认重复训练这三组。新机器首先运行 no-gate、shared-FFN，然后 λ=0.01、0.1。

λ=0 在完整架构中就是 without prior bias。旧 `Qwen3TemporalHierarchicalFixedBias` 的 table 冻结且采用 `zero` 初始化，配置中的 `soft_scale=0.1` 在 zero 初始化分支不生效，计算上与 FixedSoft 的 soft_scale=0 等价。已有 `results/ShortVideoAD/smb_explicit_decoder_4/Qwen3TemporalHierarchicalFixedBias/results-smb_explicit-original.json`，其 cvr HR@10=0.1974043716、NDCG@10=0.0872569724。该旧文件没有 `.run.json` 或最终训练 manifest/checkpoint，文件名本身不能独立证明当时实际 split。当前 FixedSoft 对应旧文件的值是 0.1980874317 / 0.0899807344，同样不是本轮在跑的新 Full。**先把 λ=0 标为待核验复用，不放进默认重跑队列**；若核验旧数据/SID/增强/训练预算/seed/选模失败，再执行下文备用命令。

优先级：P0-A 组件 → P0-B λ 新点 → Tmall prior-order → 关键 seed → 条件具备的 core-only → 可选等容量额外 seed。更多组件 seed 只在需要对应结论时展开；稀疏分桶保留 future work。

## 0. 目标机准备一次

在目标机 Bash 中激活 GAMER Python 环境，从仓库根目录执行。新 checkout 可按[换机说明](2026-10-27-rebuttal-other-machine.zh.md)获取代码，先取得 ShortVideoAD LFS 输入：

```bash
git lfs pull --include='data/ShortVideoAD/**' --exclude=''
```

**推荐的快速准备**：新机器不需要先有 `results/`，用下面一条命令在当前机器生成 P0 和 seed 清单，并单进程准备缓存；此命令不启动训练。它代替本节后面的手动生成／预热步骤，两种方式选择其一：

```bash
export WANDB_MODE=offline
python scripts/prepare_remaining_rebuttal.py \
  --gpu 2,5 --port 24567 --prepare_cache
```

生成 `results/rebuttal/remaining-p0/plan.json` 和 `results/rebuttal/remaining-seeds/plan.json`，绑定目标机 Python、配置／数据路径及指纹。默认有效 batch=2048，每卡 train batch=128；双卡累积自动为 8，四卡为 4，单卡为 16。不能整除的卡数／batch 组合会报错。可用 `--data_path`、`--output_root` 指定目录，已有清单不会覆盖；省略 `--prepare_cache` 则只生成清单。输入缺失或仍是 LFS pointer 时先报错，不下载数据、不生成假清单。`WANDB_MODE=offline` 用于后续运行，避免未登录的新机器出现交互提示。

Tmall 输入与 CID 索引到位后，可单独执行 `python scripts/prepare_remaining_rebuttal.py --groups tmall --gpu 2,5 --port 24567 --prepare_cache`，生成 `remaining-tmall/plan.json`。不要直接复制服务器上的旧 plan；其中绝对路径和解释器属于原机器。以下保留可自行调整配方的手动准备命令。

生成本地配置；此处不启动训练。示例按两张 80GB GPU，卡号为 `0,1`，端口 `2314`；可替换卡号/端口。训练 per-device batch=128、累积=8，有效 batch=2048；FP32、history=100、model length=1024、random factor4、seed=42。小显存机器需另行调整 microbatch/累积并保持有效 batch。Ranking 统一每卡 batch=8、beam=20，以降低推理显存压力；它不是 GPU 吞吐实测保证。原服务器后续补测也用相同 ranking batch。

```bash
export WANDB_MODE=offline
rebuttal_common=(
  --base_model ./config/s2s-models/Qwen3TemporalHierarchicalFixedSoft
  --data_path ./data --tasks smb_explicit_decoder_4
  --max_his_len 100 --model_max_length 1024
  --gpu 0,1 --port 2314 --precision fp32
  --per_device_train_batch_size 128 --per_device_eval_batch_size 128
  --gradient_accumulation_steps 8 --num_train_epochs 200
  --learning_rate 5e-4 --weight_decay 0.01 --warmup_ratio 0.04
  --patience 20 --temperature 0.7 --test_batch_size 8 --num_beams 20
)
python scripts/prepare_rebuttal_experiments.py \
  "${rebuttal_common[@]}" --suite handoff --seeds 42 \
  --dataset ShortVideoAD --index_file .index.json --eval_behavior cvr \
  --output_dir ./results/rebuttal/remaining-p0
python scripts/prepare_rebuttal_experiments.py \
  "${rebuttal_common[@]}" --suite components --seeds 43 44 --export_user_targets \
  --dataset ShortVideoAD --index_file .index.json --eval_behavior cvr \
  --output_dir ./results/rebuttal/remaining-seeds
```

上述离线 W&B 设置避免新机器登录交互，不改变模型训练目标；若使用已配置的在线 W&B 可自行改为 online。后续指令假定同一 Bash 会话，换终端时需重新设置该环境变量。

新机器冷缓存先单进程预热一次，避免首次多进程同时创建缓存。它不训练模型，但会读取数据并生成数 GB 级缓存；足够的内存、磁盘和时间属于输入准备开销：

```bash
python - <<'PY'
from SeqRec.datasets.loaders.session_behavior import load_SMB_datasets, load_SMB_test_dataset
settings = dict(dataset='ShortVideoAD', data_path='./data', max_his_len=100, index_file='.index.json')
load_SMB_datasets(**settings, tasks='smb_explicit_decoder_4', train_session=True)
for task in ('smb_explicit_valid', 'smb_explicit'):
    load_SMB_test_dataset(**settings, test_task=task)
PY
```

两个 plan 均包含未执行的候选配置，不代表会自动训练其中所有组。以下 runner 只执行明确指定的变体；去掉 `--execute` 即为 dry-run。每组依次执行 train → validation ranking → test ranking，训练内部按 validation loss 保存 best model；失败即停止，已有输出即拒绝，不自动覆盖或续训。它不会推断本服务器或其他机器的任务状态，阶段顺序由用户按下列清单安排。

下列运行命令均显式列出 `--gpu 0,1 --port 2314`；直接换成 `--gpu 2,5 --port 24567` 即可在指定卡上完成训练和评测，无需修改同卡数的 plan。`--gpus` 是 `--gpu` 的别名；工具不会自动选择或占用其他卡，也不保证所选卡空闲。

单卡要重新生成 plan 为 `--gpu 0 --gradient_accumulation_steps 16`；四卡为 `--gpu 0,1,2,3 --gradient_accumulation_steps 4`，保持有效 batch=2048。不要只在 runner 中改变训练卡数。同一优先级可用不同两卡组合/不同端口并行，但先完成缓存预热，且同一配置仅分配一台机器。

## 1. P0-A：no-gate 和 shared-FFN

```bash
python scripts/run_rebuttal_experiment.py \
  --gpu 0,1 --port 2314 \
  --plan ./results/rebuttal/remaining-p0/plan.json \
  --variants no_gate shared_ffn --seeds 42 --execute
```

此命令只补两次训练，各自自动生成 validation/test JSON。Full 和 no-QKV 使用服务器现有任务，不重复运行。

## 2. P0-B：λ=0.01、0.1

组件阶段完成后执行：

```bash
python scripts/run_rebuttal_experiment.py \
  --gpu 0,1 --port 2314 \
  --plan ./results/rebuttal/remaining-p0/plan.json \
  --variants lambda_0.01 lambda_0.1 --seeds 42 --execute
```

λ=0.05 共用服务器 Full。**只有旧 FixedBias 无法作为同配方 λ=0 复用时**，补以下一项：

```bash
python scripts/run_rebuttal_experiment.py \
  --gpu 0,1 --port 2314 \
  --plan ./results/rebuttal/remaining-p0/plan.json \
  --variants zero_prior --seeds 42 --execute
```

## 3. P1：Tmall prior-order 三组

沿用原先审计的 `Tmall-24-0.25-V2` 和同一 CID 索引，在三组内部固定数据、身份、目标、增强和候选。该 CID 索引被 Git 忽略，**需先从授权来源复制到目标机**：

`data/Tmall-24-0.25-V2/Tmall-24-0.25-V2.index.cid.shuffle.chunk64.json`

其当前源内容 SHA256 为 `1263e3b160f89e4f912feadd5d57e6aa1c9f37fe582ec670d7b5152d4bd6cbe2`。`V2` 的来源 JSON 是指向原 Tmall-24-0.25 的仓库内相对链接，需获取原文件：

```bash
git lfs pull --include='data/Tmall-24-0.25/**' --exclude=''
python scripts/prepare_rebuttal_experiments.py \
  --suite hierarchy --seeds 42 \
  --base_model ./config/s2s-models/Qwen3TemporalHierarchicalFixedSoft \
  --dataset Tmall-24-0.25-V2 --data_path ./data \
  --index_file .index.cid.shuffle.chunk64.json \
  --tasks smb_explicit_decoder_4 --max_his_len 100 --model_max_length 1024 \
  --gpu 0,1 --port 2314 --precision fp32 \
  --per_device_train_batch_size 128 --per_device_eval_batch_size 128 \
  --gradient_accumulation_steps 8 --num_train_epochs 200 \
  --learning_rate 5e-4 --weight_decay 0.01 --warmup_ratio 0.04 \
  --patience 20 --temperature 0.7 --test_batch_size 8 --num_beams 20 \
  --eval_behavior alipay --export_user_targets \
  --hierarchy_maps ./config/rebuttal/tmall-hierarchy-screening.json \
  --output_dir ./results/rebuttal/remaining-tmall
python - <<'PY'
from SeqRec.datasets.loaders.session_behavior import load_SMB_datasets, load_SMB_test_dataset
settings = dict(dataset='Tmall-24-0.25-V2', data_path='./data', max_his_len=100, index_file='.index.cid.shuffle.chunk64.json')
load_SMB_datasets(**settings, tasks='smb_explicit_decoder_4', train_session=True)
for task in ('smb_explicit_valid', 'smb_explicit'):
    load_SMB_test_dataset(**settings, test_task=task)
PY
python scripts/run_rebuttal_experiment.py \
  --gpu 0,1 --port 2314 \
  --plan ./results/rebuttal/remaining-tmall/plan.json \
  --variants full weak_auxiliary random_auxiliary_seed42 --seeds 42 --execute
```

三组是 legacy identity-order prior、辅助层级并列映射、预设非恒等随机辅助顺序。随机映射来自 `random.Random(42).sample([0,1,2],3)=[2,0,1]`，不是只互换 collect/cart，core alipay 的身份不变。只改 prior table 的行为间子表，保留 special ID 0 的既有行列，不改全局 behavior-level 文件；并列映射还改变 rank 间距，随机映射则保留 rank 集合。V2 元数据让 collect/cart/alipay 同为 max level，legacy target 是最后的 alipay，增强会保留这些 max-level 事件；三组完全共享这一事实。本实验只解释该数据与 CID 下的 prior-order sensitivity，不外推为全部层级机制的消融。

## 4. P1：关键对照补 seed 43、44

先预设 Full 对 no-QKV 为关键比较，避免看 test 后才选 seed。每个变体补两次；服务器已有各自 seed 42：

```bash
python scripts/run_rebuttal_experiment.py \
  --gpu 0,1 --port 2314 \
  --plan ./results/rebuttal/remaining-seeds/plan.json \
  --variants full no_qkv --seeds 43 44 --execute
```

若最终需要其他组件的跨训练稳定性，再补对应组；以下三项全部展开是额外六次训练，不属于首轮必须项：

```bash
python scripts/run_rebuttal_experiment.py \
  --gpu 0,1 --port 2314 \
  --plan ./results/rebuttal/remaining-seeds/plan.json \
  --variants no_gate shared_ffn zero_prior --seeds 43 44 --execute
```

## 5. 可选：等容量 decoder 的另外两个 seed

服务器已有 plain_matched seed 42；如仍需完整三 seed 容量比较：

```bash
python scripts/run_rebuttal_experiment.py \
  --gpu 0,1 --port 2314 \
  --plan ./results/rebuttal/remaining-seeds/plan.json \
  --variants plain_matched --seeds 43 44 --execute
```

## 已启动三组的后续评测与产物

本服务器 Full/no-QKV/plain_matched 在训练完成后仍需 validation/test。若要在另一台机器评测，先复制**最终 checkpoint 根目录**到目标机（包括 tokenizer/config/weights/start manifest），而不是只复制某个中间 checkpoint；此操作不能在训练尚未完成时冒充 final。

从服务器复制到 `./results/rebuttal/transferred/<variant>/checkpoint/` 后，在目标机执行以下 evaluation-only 命令，不重训：

```bash
for variant in full no_qkv plain_matched; do
  for split in validation test; do
    task=smb_explicit
    if [[ "$split" == validation ]]; then task=smb_explicit_valid; fi
    result="./results/rebuttal/transferred-eval/${variant}/${split}-ranking.json"
    test ! -e "$result" || exit 1
    CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python -m torch.distributed.run \
      --nproc_per_node=2 --master_port=2314 main.py test_SMB_decoder \
      --backbone Qwen3TemporalHierarchical \
      --ckpt_path "./results/rebuttal/transferred/${variant}/checkpoint" \
      --dataset ShortVideoAD --data_path ./data --index_file .index.json \
      --seed 42 --max_his_len 100 --test_batch_size 8 --num_beams 20 \
      --test_task "$task" --behaviors cvr --export_user_targets \
      --results_file "$result" || exit 1
  done
done
```

Runner 的一般产物是 `<plan目录>/<variant>/seed_<seed>/checkpoint/`、`validation-ranking.json`、`test-ranking.json`；对应目录还包含逐用户 metric、UID、真实 target ID、split/hash receipt，以及逐阶段 pipeline start/finish 记录。默认不合并 validation/test，不将 test 用于 checkpoint 或 λ 选择。

如只汇总新机器已完成的 P0 结果（没有 Full 时差值留空）：

```bash
python scripts/summarize_rebuttal_results.py \
  --plan ./results/rebuttal/remaining-p0/plan.json \
  --split test --behavior cvr --reference full --allow_partial \
  --output_file ./results/rebuttal/remaining-p0/summary-cvr-test.json \
  --csv_file ./results/rebuttal/remaining-p0/summary-cvr-test.csv
```

Tmall 三组齐全后可去掉 `--allow_partial`，替换 plan 和 `--behavior alipay` 得到相对同机 Full 的差值。跨机器合并需收集双方 plan/manifest/results 并按数据内容、SID、配方和真实 split 核验；轻量路径/mtime 指纹不具备自动跨机器迁移能力，不能靠改 JSON 路径冒充兼容。旧 λ=0 结果也不自动导入受控表。

## 尚不能给出无条件开跑命令的项目

- strict core-only/raw/consolidated：还缺真实 raw、consolidation provenance 和共同 split/target manifest。已有适配工具，但不得从最深状态反向展开 raw。输入到位后再固定可比目标/候选和验证选择口径。
- 同设置 runtime/MPC：前者还需确定计时协议，后者尚缺独立 baseline 实现与评测适配，均不在本次 P0 自动队列中。
- 稀疏／popularity／重复目标诊断及 Q/K/V 分拆，按最新计划保留后置或 future work，不在上述默认训练指令中恢复。

## 命令验证

相关 CPU 测试共 27 项通过，包括 FixedBias zero init 与 FixedSoft λ=0 的权重和 logits 完全一致。另在临时 Toy 数据上实际执行一次 train → validation → test 的完整 runner 链路，生成 JSON、UID/目标和阶段 receipt；仅测试时注入 CPU device，不改生产训练路径。本轮未启动生产训练或 GPU 评测，已有三组训练继续运行。双语命令块保持一致，并执行 Bash 语法、嵌入 Python 语法及规定 flake8 检查。

新增本地准备入口另通过 16 项相关检查（包含重叠回归），验证无 results 目录生成、单进程冷缓存、GPU/累积匹配、LFS pointer 拒绝、缺失 Tmall 输入和已有输出保护。
