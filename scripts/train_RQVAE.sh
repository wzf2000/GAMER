##!/bin/bash
# This script trains the RQ-VAE model on a specified dataset with given hyperparameters.
: ${dataset:=Beauty}
: ${alpha:=0.02}
: ${beta:=0.0001}
: ${gpu:=0}
: ${semantic_model:=llama-3.1}
: ${cf_model:=sasrec}

echo "Training RQ-VAE on ${dataset} with alpha=${alpha} and beta=${beta} using GPU ${gpu}, semantic model ${semantic_model}, and cf model ${cf_model}."

python main.py RQVAE \
  --device cuda:${gpu} \
  --data_path ./data/${dataset}/${dataset}.emb-${semantic_model}-td.npy\
  --alpha ${alpha} \
  --beta ${beta} \
  --cf_emb ./ckpt/cf-embs/ckpt/${dataset}-32d-${cf_model}.pt\
  --ckpt_dir ./checkpoint/RQ-VAE/${dataset}
