#!/bin/bash
set -e

: ${dataset:=ShortVideoAD}
: ${data_path:=/home/zhouman/guoyunhe/workspace/full/GAMER-rank/data}
: ${batch_size:=1024}
: ${max_his_len:=100}
: ${gpu:=0}
: ${backbone:=DIN}
: ${seed:=42}
: ${optim:=adamw}
: ${learning_rate:=1e-3}
: ${weight_decay:=0.01}
: ${logging_step:=30}
: ${patience:=2}
: ${metrics:=auc,prauc,logloss,accuracy,precision,recall,f1,gauc_macro,gauc_pair,gauc}

case "${backbone}" in
    MeanPooling) output_scope=SMB-MeanPooling ;;
    DIN) output_scope=SMB-DIN ;;
    SASRecCVR) output_scope=SMB-SASRecCVR ;;
    DIENCVR) output_scope=SMB-DIENCVR ;;
    BSTCVR) output_scope=SMB-BSTCVR ;;
    HSTUCVR) output_scope=SMB-HSTUCVR ;;
    DSIN) output_scope=SMB-DSIN ;;
    *)
        echo "Unsupported binary baseline: ${backbone}."
        exit 1
        ;;
esac

if [[ "${backbone}" == "DSIN" ]]; then
    default_task=smb_ctr_dsin
else
    default_task=smb_ctr_din
fi
: ${tasks:=${default_task}}
: ${test_task:=${tasks}}

if [[ "${backbone}" == "BSTCVR" ]]; then
    : ${suffix:=item_id_new}
else
    : ${suffix:=item_id}
fi
: ${epochs:=3}
: ${save_epoch_limit:=3}
: ${ckpt_num:=best}

export CUDA_VISIBLE_DEVICES="${gpu}"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/lib/args.sh"
source "${script_dir}/lib/paths.sh"

base_model=./config/dis-models/${backbone}

parse_script_path_args "$@"
task_dir=$(build_task_dir "${dataset}" "${tasks}" "${backbone}" "${suffix}")
output_dir=./checkpoint/${output_scope}/${task_dir}/
result_dir=$(build_result_path "${task_dir}" "")

build_extra_cli_args "${SCRIPT_CLI_ARGS[@]}"
print_extra_cli_args

python main.py train_SMB_rec \
    --seed "${seed}" \
    --backbone "${backbone}" \
    --base_model "${base_model}" \
    --output_dir "${output_dir}" \
    --result_dir "${result_dir}" \
    --wandb_run_name "${task_dir}" \
    --dataset "${dataset}" \
    --data_path "${data_path}" \
    --batch_size "${batch_size}" \
    --tasks "${tasks}" \
    --test_task "${test_task}" \
    --max_his_len "${max_his_len}" \
    --optim "${optim}" \
    --epochs "${epochs}" \
    --learning_rate "${learning_rate}" \
    --logging_step "${logging_step}" \
    --weight_decay "${weight_decay}" \
    --patience "${patience}" \
    --save_epoch_limit "${save_epoch_limit}" \
    --ckpt_num "${ckpt_num}" \
    --metrics "${metrics}" \
    "${EXTRA_CLI_ARGS[@]}"
