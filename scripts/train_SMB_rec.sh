#!/bin/bash
: ${dataset:=Retail}
: ${batch_size:=4096}
: ${learning_rate:=1e-3}
: ${tasks:=smb_dis}
: ${test_task:=smb_dis}
: ${gpu:=0}
: ${epochs:=200}
: ${backbone:=GRU4Rec}

export CUDA_VISIBLE_DEVICES=$gpu

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/lib/args.sh"
source "${script_dir}/lib/paths.sh"

base_model=./config/dis-models/${backbone}

: ${suffix:=}
task_dir=$(build_task_dir "${dataset}" "${tasks}" "${backbone}" "${suffix}")

output_dir=./checkpoint/smb_dis/${task_dir}/
result_dir=$(build_result_path "${task_dir}" "")
run_name=${task_dir}

build_extra_cli_args "$@"
print_extra_cli_args

python main.py train_SMB_rec \
    --backbone ${backbone} \
    --base_model ${base_model} \
    --output_dir ${output_dir} \
    --result_dir ${result_dir} \
    --wandb_run_name ${run_name} \
    --dataset ${dataset} \
    --batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --tasks ${tasks} \
    --test_task ${test_task} \
    --epochs ${epochs} \
    "${EXTRA_CLI_ARGS[@]}"
