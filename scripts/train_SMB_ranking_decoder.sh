#!/bin/bash
: ${dataset:=ShortVideoADSmall}
: ${data_path:=/home/zhouman/guoyunhe/workspace/full/GAMER/data}
: ${original:=1}
: ${rq_kmeans:=0}
: ${batch_size:=1024}
: ${tasks:=smb_ranking_decoder}
: ${max_his_len:=50}
: ${gpu:=0,1,2,3,4,5,6,7}
: ${port:=2314}
: ${backbone:=Qwen3TemporalHierarchicalFactorized}
: ${train_auc_samples:=1024}
: ${train_auc_batch_size:=256}
: ${eval_epochs:=10}

export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export SMB_RANKING_TRAIN_AUC_SAMPLES=${train_auc_samples}
export SMB_RANKING_TRAIN_AUC_BATCH_SIZE=${train_auc_batch_size}
export SMB_RANKING_EVAL_EPOCHS=${eval_epochs}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/lib/s2s_backbone.sh"
source "${script_dir}/lib/args.sh"
source "${script_dir}/lib/tokenization.sh"
source "${script_dir}/lib/paths.sh"
source "${script_dir}/lib/runtime.sh"

gpu_num=$(count_gpus "${gpu}")
per_device_batch_size=$(compute_per_device_batch_size "${batch_size}" "${gpu_num}")
backbone_arg=$(resolve_s2s_backbone_arg "${backbone}")
if ! base_model=$(resolve_s2s_base_model "${backbone}"); then
    echo "Unsupported backbone model: ${backbone}."
    exit 1
fi

: ${suffix:=}
parse_script_path_args "$@"
task_dir=$(build_task_dir "${dataset}" "${tasks}" "${backbone}" "${suffix}")

resolve_tokenization
output_dir=$(build_checkpoint_path "SMB-ranking-decoder" "${task_dir}" "${token_tag}")
run_name=${task_dir}/${token_tag}/
echo "Training SMB Ranking Decoder on ${dataset} using ${tokenization_desc} with GPUs ${gpu}."
echo "Sampled train-time CVR AUC: samples=${train_auc_samples}, batch=${train_auc_batch_size}."
echo "Eval/save interval: every ${eval_epochs} epoch(s)."

build_extra_cli_args "${SCRIPT_CLI_ARGS[@]}"
print_extra_cli_args

run_main_distributed "${gpu_num}" "${port}" train_SMB_ranking_decoder \
    --backbone ${backbone_arg} \
    --base_model ${base_model} \
    --output_dir ${output_dir} \
    --wandb_run_name ${run_name} \
    --dataset ${dataset} \
    --data_path ${data_path} \
    --per_device_batch_size ${per_device_batch_size} \
    --tasks ${tasks} \
    --max_his_len ${max_his_len} \
    --index_file ${index_file} \
    "${EXTRA_CLI_ARGS[@]}"
