#!/bin/bash
: ${dataset:=ShortVideoADSmall}
: ${data_path:=/home/zhouman/guoyunhe/workspace/full/GAMER/data}
: ${batch_size:=1024}
: ${tasks:=smb_din}
: ${max_his_len:=50}
: ${test_task:=smb_din}
: ${gpu:=0}
: ${backbone:=DIN}
: ${metrics:=auc}

export CUDA_VISIBLE_DEVICES=$gpu

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/lib/args.sh"
source "${script_dir}/lib/paths.sh"

base_model=./config/dis-models/${backbone}

: ${suffix:=}
parse_script_path_args "$@"
task_dir=$(build_task_dir "${dataset}" "${tasks}" "${backbone}" "${suffix}")

output_dir=./checkpoint/SMB-DIN/${task_dir}/
result_dir=$(build_result_path "${task_dir}" "")
run_name=${task_dir}

build_extra_cli_args "${SCRIPT_CLI_ARGS[@]}"
print_extra_cli_args

python main.py train_SMB_rec \
    --backbone ${backbone} \
    --base_model ${base_model} \
    --output_dir ${output_dir} \
    --result_dir ${result_dir} \
    --wandb_run_name ${run_name} \
    --dataset ${dataset} \
    --data_path ${data_path} \
    --batch_size ${batch_size} \
    --tasks ${tasks} \
    --test_task ${test_task} \
    --max_his_len ${max_his_len} \
    --metrics ${metrics} \
    "${EXTRA_CLI_ARGS[@]}"
