#!/bin/bash
: ${dataset:=Retail}
: ${rq_kmeans:=0}
: ${batch_size:=512}
: ${learning_rate:=5e-4}
: ${tasks:=smb_explicit}
: ${gpu:=0,1,2,3}
: ${epochs:=200}
: ${port:=2314}
: ${backbone:=TIGER}

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
if ! base_model=$(resolve_s2s_base_model "${backbone}"); then
    echo "Unsupported backbone model: ${backbone}."
    exit 1
fi

: ${suffix:=}
task_dir=$(build_task_dir "${dataset}" "${tasks}" "${backbone}" "${suffix}")


resolve_tokenization
output_dir=$(build_checkpoint_path "SMB-decoder" "${task_dir}" "${token_tag}")
run_name=${task_dir}/${token_tag}/
echo "Training SMB Decoder on ${dataset} using ${tokenization_desc} with GPUs ${gpu}."

: ${extra_args:=}
extra_args_out=$(parse_extra_args "${extra_args}")
echo "Extra arguments: ${extra_args_out}"

: ${extra_flags:=}
extra_flags_out=$(parse_extra_flags "${extra_flags}")
echo "Extra flags: ${extra_flags_out}"

run_main_distributed "${gpu_num}" "${port}" train_SMB_decoder \
    --backbone ${backbone_arg} \
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
    ${extra_args_out} \
    ${extra_flags_out}
