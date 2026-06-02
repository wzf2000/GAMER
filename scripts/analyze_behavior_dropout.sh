#!/bin/bash
: ${dataset:=ShortVideoAD}
: ${tasks:=smb_explicit}
: ${test_task:=smb_explicit}
: ${gpu:=0}
: ${backbone:=Qwen3Multi}
: ${num_beams:=20}
: ${max_users:=20}
: ${batch_size:=16}

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

: ${suffix:=}
task_dir=$(build_task_dir "${dataset}" "${tasks}" "${backbone}" "${suffix}")

resolve_tokenization
results_file=$(build_result_path "${task_dir}" "behavior_dropout-${test_task}-${token_tag}.json")
ckpt_path=$(build_checkpoint_path "SMB-decoder" "${task_dir}" "${token_tag}")
echo "Behavior dropout analysis on ${dataset} using ${tokenization_desc}."

: ${ckpt_num:=best}
if [[ "$ckpt_num" != "best" ]]; then
    ckpt_path=${ckpt_path}checkpoint-${ckpt_num}/
    echo "Using checkpoint from step ${ckpt_num}."
else
    echo "Using the best checkpoint."
fi

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

# This task is single-process only (post-hoc analysis on a small user subset)
echo "Running behavior dropout analysis (max_users=${max_users}, num_beams=${num_beams})."
python main.py analyze_behavior_dropout \
    --backbone ${backbone_arg} \
    --ckpt_path ${ckpt_path} \
    --dataset ${dataset} \
    --data_path ${data_path} \
    --results_file ${results_file} \
    --test_batch_size ${per_device_batch_size} \
    --num_beams ${num_beams} \
    --max_users ${max_users} \
    --index_file ${index_file} \
    --test_task ${test_task} \
    ${target_behavior_arg} \
    ${extra_args_out} \
    ${extra_flags_out}
