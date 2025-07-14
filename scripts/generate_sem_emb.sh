#!/bin/bash
# This script generates item semantic embeddings for a specified dataset using the main.py script.
: ${dataset:=Beauty}
: ${gpu:=0}
: ${semantic_model:=llama-3.1}
: ${checkpoint:=Meta-llama/Meta-Llama-3.1-8B}
: ${max_sent_len:=2048}

python main.py SemEmb \
  --dataset ${dataset} \
  --root ./data \
  --gpu_id ${gpu} \
  --plm_name ${semantic_model} \
  --plm_checkpoint ${checkpoint} \
  --max_sent_len ${max_sent_len} \
