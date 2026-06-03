#!/bin/bash
: ${dataset:=Retail}
: ${rq_kmeans:=0}
: ${batch_size:=1024}
: ${tasks=mb_explicit}
: ${test_task:=mb_explicit}
: ${gpu:=0,1,2,3}
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

data_path=./data
gpu_num=$(count_gpus "${gpu}")
per_device_batch_size=$(compute_per_device_batch_size "${batch_size}" "${gpu_num}")

: ${suffix:=}
task_dir=$(build_task_dir "${dataset}" "${tasks}" "${backbone}" "${suffix}")

resolve_tokenization
results_file=$(build_result_path "${task_dir}" "results-${token_tag}.json")
ckpt_path=$(build_checkpoint_path "MB-decoder" "${task_dir}" "${token_tag}")
echo "Testing MB decoder on ${dataset} using ${tokenization_desc} with GPU ${gpu}."

: ${ckpt_num:=best}
if [[ "$ckpt_num" == "best" ]]; then
    # no changes needed
    echo "Using the best checkpoint."
else
    ckpt_path=${ckpt_path}checkpoint-${ckpt_num}/
    echo "Using checkpoint from step ${ckpt_num}."
fi

build_extra_cli_args "$@"
print_extra_cli_args

run_main_distributed "${gpu_num}" "${port}" test_MB_decoder \
    --backbone ${backbone} \
    --ckpt_path ${ckpt_path} \
    --dataset ${dataset} \
    --data_path ${data_path} \
    --results_file ${results_file} \
    --test_batch_size ${per_device_batch_size} \
    --num_beams 20 \
    --index_file ${index_file} \
    --test_task ${test_task} \
    "${EXTRA_CLI_ARGS[@]}"
