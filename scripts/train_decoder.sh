#!/bin/bash
: ${dataset:=Beauty}
: ${rq_kmeans:=0}
: ${batch_size:=512}
: ${learning_rate:=5e-4}
: ${tasks:=seqrec}
: ${gpu:=0,1,2,3}
: ${epochs:=200}
: ${port:=2314}
: ${backbone:=TIGER}

export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/lib/args.sh"
source "${script_dir}/lib/tokenization.sh"
source "${script_dir}/lib/paths.sh"
source "${script_dir}/lib/runtime.sh"

gpu_num=$(count_gpus "${gpu}")
per_device_batch_size=$(compute_per_device_batch_size "${batch_size}" "${gpu_num}")
if [ "${backbone}" = "TIGER" ]; then
    base_model=./config/s2s-models/TIGER
elif [ "${backbone}" = "PBATransformer" ]; then
    base_model=./config/s2s-models/PBATransformer
elif [ "${backbone}" = "Qwen3" ]; then
    base_model=./config/s2s-models/Qwen3-Light
else
    echo "Unsupported backbone model: ${backbone}."
    exit 1
fi

task_dir=$(build_task_dir "${dataset}" "${tasks}" "${backbone}")

resolve_tokenization
output_dir=$(build_checkpoint_path "decoder" "${task_dir}" "${token_tag}")
run_name=${task_dir}/${token_tag}/
echo "Training Decoder on ${dataset} using ${tokenization_desc} with GPUs ${gpu}."

build_extra_cli_args "$@"
print_extra_cli_args

run_main_distributed "${gpu_num}" "${port}" train_decoder \
    --backbone ${backbone} \
    --base_model ${base_model} \
    --output_dir ${output_dir} \
    --wandb_run_name ${run_name} \
    --dataset ${dataset} \
    --per_device_batch_size ${per_device_batch_size} \
    --learning_rate ${learning_rate} \
    --tasks ${tasks} \
    --epochs ${epochs} \
    --index_file ${index_file} \
    --temperature 0.7 \
    "${EXTRA_CLI_ARGS[@]}"
