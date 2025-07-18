##!/bin/bash
# This script trains the RQ-VAE model on a specified dataset with given hyperparameters.
: ${dataset:=Beauty}
: ${alpha:=0.02}
: ${beta:=0.0001}
: ${batch_size:=1024}
: ${gpu:=0,1,2,3}
: ${semantic_model:=llama-3.1}
: ${cf_model:=sasrec}

export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_LAUNCH_BLOCKING=1

gpu_num=$(echo $gpu | awk -F, '{print NF}')
per_device_batch_size=$(($batch_size / $gpu_num))

: ${extra_args:=}
# transform the format of "X=a,Y=b" into "-X a -Y b"
extra_args_out=$(echo "$extra_args" | awk -F, '{
  for(i=1; i<=NF; i++) {
    split($i, arr, "=")
    printf "--%s %s ", arr[1], arr[2]
  }
}')
echo "Extra arguments: ${extra_args_out}"

echo "Training RQ-VAE on ${dataset} with alpha=${alpha} and beta=${beta} using GPU ${gpu}, semantic model ${semantic_model}, and cf model ${cf_model}."

# check if gpu_num is greater than 1
if [ $gpu_num -gt 1 ]; then
  export OMP_NUM_THREADS=1
  torchrun --nproc_per_node=${gpu_num} ./main.py RQVAE \
    --data_path ./data/${dataset}/${dataset}.emb-${semantic_model}-td.npy \
    --alpha ${alpha} \
    --beta ${beta} \
    --cf_emb ./ckpt/cf-embs/ckpt/${dataset}-32d-${cf_model}.pt \
    --ckpt_dir ./checkpoint/RQ-VAE/${dataset} \
    --batch_size ${per_device_batch_size} \
    ${extra_args_out}
else
  # if gpu_num is 1, use single GPU training
  python main.py RQVAE \
    --device cuda:${gpu} \
    --data_path ./data/${dataset}/${dataset}.emb-${semantic_model}-td.npy \
    --alpha ${alpha} \
    --beta ${beta} \
    --cf_emb ./ckpt/cf-embs/ckpt/${dataset}-32d-${cf_model}.pt \
    --ckpt_dir ./checkpoint/RQ-VAE/${dataset} \
    --batch_size ${per_device_batch_size} \
    ${extra_args_out}
fi
