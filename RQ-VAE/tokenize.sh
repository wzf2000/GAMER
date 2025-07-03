#!#!/bin/bash
# This script generates indices for the RQ-VAE model using a specified dataset and hyperparameters.
: ${dataset:=Yelp}
: ${alpha:=0.02}
: ${beta:=0.0001}
: ${gpu:=0}

echo "Generating indices for ${dataset} with alpha=${alpha} and beta=${beta} using GPU ${gpu}."

python ./RQ-VAE/generate_indices.py\
  --device cuda:${gpu} \
  --dataset ${dataset} \
  --root_path ./checkpoint/${dataset}/ \
  --alpha ${alpha} \
  --beta ${beta} \
  --epoch 20000 \
  --checkpoint best_collision_model.pth
