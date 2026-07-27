#!/bin/bash
: ${dataset:=ShortVideoAD}
: ${data_path:=/home/zhouman/guoyunhe/workspace/full/GAMER-rank/data}
: ${original:=1}
: ${rq_kmeans:=0}
: ${batch_size:=204800}
: ${tasks:=smb_ranking_decoder}
: ${test_task:=smb_ranking_decoder}
: ${metrics:=auc,logloss,gauc}
: ${gpu:=4,5,6}
: ${port:=2316}
: ${backbone:=Qwen3TemporalHierarchicalFactorized}

export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/lib/s2s_backbone.sh"
source "${script_dir}/lib/args.sh"
source "${script_dir}/lib/tokenization.sh"
source "${script_dir}/lib/paths.sh"
source "${script_dir}/lib/runtime.sh"

gpu_num=$(count_gpus "${gpu}")
per_device_batch_size=$(compute_per_device_batch_size "${batch_size}" "${gpu_num}")
backbone_arg=$(resolve_s2s_backbone_arg "${backbone}")

has_model_file() {
    [[ -f "$1/model.safetensors" || -f "$1/pytorch_model.bin" ]]
}

latest_checkpoint() {
    find "$1" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
}

best_checkpoint() {
    local root="$1"
    if has_model_file "${root%/}"; then
        echo "$root"
        return
    fi

    local latest
    latest=$(latest_checkpoint "$root")
    if [[ -z "$latest" ]]; then
        echo "$root"
        return
    fi

    local best=""
    if [[ -f "$latest/trainer_state.json" ]]; then
        best=$("${PYTHON:-python}" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("best_model_checkpoint", ""))' "$latest/trainer_state.json")
    fi
    if [[ -n "$best" && -d "$best" ]]; then
        echo "${best%/}/"
    else
        echo "${latest%/}/"
    fi
}

: ${suffix:=new}
parse_script_path_args "$@"
task_dir=$(build_task_dir "${dataset}" "${tasks}" "${backbone}" "${suffix}")

resolve_tokenization
results_file=$(build_result_path "${task_dir}" "results-${test_task}-${token_tag}.json")
ckpt_path=$(build_checkpoint_path "SMB-ranking-decoder" "${task_dir}" "${token_tag}")
echo "Testing SMB ranking decoder on ${dataset} using ${tokenization_desc} with GPU ${gpu}."

: ${ckpt_num:=475}
if [[ "$ckpt_num" == "best" ]]; then
    ckpt_path=$(best_checkpoint "$ckpt_path")
    echo "Using the best checkpoint."
else
    ckpt_path=${ckpt_path}checkpoint-${ckpt_num}/
    echo "Using checkpoint from step ${ckpt_num}."
fi
echo "Checkpoint path: ${ckpt_path}"

build_extra_cli_args "${SCRIPT_CLI_ARGS[@]}"
print_extra_cli_args

run_main_distributed "${gpu_num}" "${port}" test_SMB_ranking_decoder \
    --backbone ${backbone_arg} \
    --ckpt_path ${ckpt_path} \
    --dataset ${dataset} \
    --data_path ${data_path} \
    --results_file ${results_file} \
    --test_batch_size ${per_device_batch_size} \
    --metrics ${metrics} \
    --index_file ${index_file} \
    --test_task ${test_task} \
    "${EXTRA_CLI_ARGS[@]}"
