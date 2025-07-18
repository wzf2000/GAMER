#!#!/bin/bash
# This script generates indices for the RQ-VAE model using a specified dataset and hyperparameters.
: ${dataset:=Beauty}
: ${rq_kmeans:=0}
: ${gpu:=0}
: ${semantic_model:=llama-3.1}

data_path="./data/${dataset}/${dataset}.emb-${semantic_model}-td.npy"
output_dir="./data/${dataset}/"

: ${extra_args:=}
# transform the format of "X=a,Y=b" into "-X a -Y b"
extra_args_out=$(echo "$extra_args" | awk -F, '{
  for(i=1; i<=NF; i++) {
    split($i, arr, "=")
    printf "--%s %s ", arr[1], arr[2]
  }
}')
echo "Extra arguments: ${extra_args_out}"

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
        --checkpoint ${checkpoint} \
        ${extra_args_out}
    else
      echo "Using Random ID tokenization for index generation."
      echo "Generating indices for ${dataset} with random ID tokenization."
      python main.py tokenize \
        --dataset ${dataset} \
        --data_path ${data_path} \
        --output_dir ${output_dir} \
        --rid \
        ${extra_args_out}
    fi
  else
    echo "Using Chunked ID tokenization for index generation."
    : ${chunk_size:=64}
    : ${shuffle:=0}
    if [ $shuffle -eq 1 ]; then
      shuffle_option="--shuffle"
      echo "Generating indices for ${dataset} with chunk size ${chunk_size} and shuffling."
    else
      shuffle_option=""
      echo "Generating indices for ${dataset} with chunk size ${chunk_size}."
    fi
    python main.py tokenize \
      --dataset ${dataset} \
      --data_path ${data_path} \
      --output_dir ${output_dir} \
      --cid \
      --chunk_size ${chunk_size} \
      ${shuffle_option} \
      ${extra_args_out}
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
      --rq_kmeans \
      ${extra_args_out}
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
          --cf_emb ${cf_emb} \
          ${extra_args_out}
      else
        echo "Generating indices for ${dataset} using RQ-Kmeans with CF embeddings and reduced semantic embeddings."
        python main.py tokenize \
          --device cuda:${gpu} \
          --dataset ${dataset} \
          --data_path ${data_path} \
          --output_dir ${output_dir} \
          --rq_kmeans \
          --cf_emb ${cf_emb} \
          --reduce \
          ${extra_args_out}
      fi
  fi
fi
