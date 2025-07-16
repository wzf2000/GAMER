#!#!/bin/bash
# This script generates indices for the RQ-VAE model using a specified dataset and hyperparameters.
: ${dataset:=Beauty}
: ${rq_kmeans:=0}
: ${gpu:=0}
: ${semantic_model:=llama-3.1}

data_path="./data/${dataset}/${dataset}.emb-${semantic_model}-td.npy"
output_dir="./data/${dataset}/"

if [ $rq_kmeans -eq 0 ]; then
  : ${cid:=0}
  if [ $cid -eq 0 ]; then
    : ${rid:=0}
    if [ $rid -eq 0 ]; then
      echo "Using RQ-VAE for index generation."
      : ${alpha:=0.02}
      : ${beta:=0.0001}
      : ${epoch:=20000}
      : ${checkpoint:=best_collision_model.pth}
      echo "Generating indices for ${dataset} with alpha=${alpha} and beta=${beta} using GPU ${gpu}."
      python main.py tokenize \
        --device cuda:${gpu} \
        --dataset ${dataset} \
        --data_path ${data_path} \
        --output_dir ${output_dir} \
        --alpha ${alpha} \
        --beta ${beta} \
        --epoch ${epoch} \
        --checkpoint ${checkpoint}
    else
      echo "Using Random ID tokenization for index generation."
      echo "Generating indices for ${dataset} with random ID tokenization."
      python main.py tokenize \
        --dataset ${dataset} \
        --data_path ${data_path} \
        --output_dir ${output_dir} \
        --rid
    fi
  else
    echo "Using Chunked ID tokenization for index generation."
    : ${chunk_size:=64}
    echo "Generating indices for ${dataset} with chunk size ${chunk_size}."
    python main.py tokenize \
      --dataset ${dataset} \
      --data_path ${data_path} \
      --output_dir ${output_dir} \
      --cid \
      --chunk_size ${chunk_size}
  fi
else
  echo "Using RQ-Kmeans for index generation."
  : ${cf_emb:=None}
  if [ "${cf_emb}" == "None" ]; then
    echo "Generating indices for ${dataset} using RQ-Kmeans without CF embeddings."
    python main.py tokenize \
      --device cuda:${gpu} \
      --dataset ${dataset} \
      --data_path ${data_path} \
      --output_dir ${output_dir} \
      --rq_kmeans
  else
      : ${reduce:=0}
      if [ $reduce -eq 0 ]; then
        echo "Generating indices for ${dataset} using RQ-Kmeans with CF embeddings."
        python main.py tokenize \
          --device cuda:${gpu} \
          --dataset ${dataset} \
          --data_path ${data_path} \
          --output_dir ${output_dir} \
          --rq_kmeans \
          --cf_emb ${cf_emb}
      else
        echo "Generating indices for ${dataset} using RQ-Kmeans with CF embeddings and reduced semantic embeddings."
        python main.py tokenize \
          --device cuda:${gpu} \
          --dataset ${dataset} \
          --data_path ${data_path} \
          --output_dir ${output_dir} \
          --rq_kmeans \
          --cf_emb ${cf_emb} \
          --reduce
      fi
  fi
fi
