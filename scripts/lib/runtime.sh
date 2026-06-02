#!/bin/bash

count_gpus() {
    local gpu="$1"
    echo "${gpu}" | awk -F, '{print NF}'
}

compute_per_device_batch_size() {
    local batch_size="$1"
    local gpu_num="$2"
    echo $((batch_size / gpu_num))
}

run_main_distributed() {
    local gpu_num="$1"
    local port="$2"
    shift 2
    if [ "${gpu_num}" -eq 1 ]; then
        echo "Using single GPU: ${CUDA_VISIBLE_DEVICES}"
        python main.py "$@"
    else
        echo "Using multiple GPUs: ${CUDA_VISIBLE_DEVICES}"
        torchrun --nproc_per_node="${gpu_num}" --master_port="${port}" ./main.py "$@"
    fi
}

