#!/bin/bash
: ${dataset:=ShortVideoAD}
: ${tasks:=smb_explicit}
: ${test_task:=smb_explicit}
: ${gpu:=0}
: ${backbone:=Qwen3Multi}
: ${baseline_backbone:=$backbone}
: ${num_beams:=20}
: ${batch_size:=16}
: ${metrics:=hit@10,ndcg@10}
: ${bucket_thresholds:=3,6}
: ${max_sparse_count:=2}
: ${max_interesting_users:=20}
: ${interesting_top_k:=10}

export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/lib/s2s_backbone.sh"
source "${script_dir}/lib/args.sh"
source "${script_dir}/lib/tokenization.sh"
source "${script_dir}/lib/paths.sh"
source "${script_dir}/lib/runtime.sh"

data_path=./data
gpu_num=$(count_gpus "${gpu}")
per_device_batch_size=$(compute_per_device_batch_size "${batch_size}" "${gpu_num}")
backbone_arg=$(resolve_s2s_backbone_arg "${backbone}")
baseline_backbone_arg=$(resolve_s2s_backbone_arg "${baseline_backbone}")

: ${suffix:=}
task_dir=$(build_task_dir "${dataset}" "${tasks}" "${backbone}" "${suffix}")

# ---------- Our model checkpoint ----------
resolve_tokenization
ckpt_path=$(build_checkpoint_path "SMB-decoder" "${task_dir}" "${token_tag}")
ckpt_tag=${token_tag}
echo "Our model: ${tokenization_desc}."

: ${ckpt_num:=best}
if [[ "$ckpt_num" != "best" ]]; then
    ckpt_path=${ckpt_path}checkpoint-${ckpt_num}/
    echo "Our model: using checkpoint from step ${ckpt_num}."
else
    echo "Our model: using the best checkpoint."
fi

# ---------- Baseline checkpoint ----------
# The baseline checkpoint must be provided explicitly via baseline_ckpt_path.
# Optionally set baseline_index_file if the baseline uses a different index.
: ${baseline_ckpt_path:=}
if [ -z "${baseline_ckpt_path}" ]; then
    echo "ERROR: baseline_ckpt_path is not set. Please set it before calling this script."
    exit 1
fi
echo "Baseline model: ${baseline_backbone_arg} from ${baseline_ckpt_path}."

# ---------- Results file ----------
results_file=$(build_result_path "${task_dir}" "sparse_behavior-${test_task}-${ckpt_tag}-vs-baseline.json")

: ${target_behavior:=}
: ${extra_args:=}
extra_args_out=$(parse_extra_args "${extra_args}")
echo "Extra arguments: ${extra_args_out}"

: ${extra_flags:=}
extra_flags_out=$(parse_extra_flags "${extra_flags}")
echo "Extra flags: ${extra_flags_out}"

target_behavior_arg=""
if [ "${target_behavior}" != "" ]; then
    target_behavior_arg="--target_behavior ${target_behavior}"
fi

echo "Running sparse-behavior analysis (batch_size=${per_device_batch_size}, num_beams=${num_beams})."
echo "  Buckets: ${bucket_thresholds} | max_sparse_count: ${max_sparse_count} | interesting_top_k: ${interesting_top_k}"
python main.py analyze_sparse_behavior \
    --backbone ${backbone_arg} \
    --ckpt_path ${ckpt_path} \
    --baseline_backbone ${baseline_backbone_arg} \
    --baseline_ckpt_path ${baseline_ckpt_path} \
    --dataset ${dataset} \
    --data_path ${data_path} \
    --results_file ${results_file} \
    --test_batch_size ${per_device_batch_size} \
    --num_beams ${num_beams} \
    --index_file ${index_file} \
    --test_task ${test_task} \
    --metrics ${metrics} \
    --bucket_thresholds ${bucket_thresholds} \
    --max_sparse_count ${max_sparse_count} \
    --max_interesting_users ${max_interesting_users} \
    --interesting_top_k ${interesting_top_k} \
    ${target_behavior_arg} \
    ${extra_args_out} \
    ${extra_flags_out}
