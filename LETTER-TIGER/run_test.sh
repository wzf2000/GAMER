: ${dataset:=Yelp}
: ${alpha:=0.02}
: ${beta:=0.0001}
: ${epoch:=20000}
: ${gpu:=0}
: ${port:=2314}
data_path=../data
results_file=./results/${dataset}/results-alpha${alpha}-beta${beta}.json
ckpt_path=./ckpt/${dataset}/alpha${alpha}-beta${beta}/

export CUDA_LAUNCH_BLOCKING=1

python test.py \
    --gpu_id ${gpu} \
    --ckpt_path ${ckpt_path} \
    --dataset ${dataset} \
    --data_path ${data_path} \
    --results_file ${results_file} \
    --test_batch_size 1024 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.epoch${epoch}.alpha${alpha}-beta${beta}.json \
    --temperature 0.7 \
