: ${dataset:=Yelp}
: ${alpha:=0.02}
: ${beta:=0.0001}
: ${epoch:=20000}
: ${gpu:=0,1,2,3}
: ${port:=2314}
output_dir=./ckpt/${dataset}/alpha${alpha}-beta${beta}/
run_name=${dataset}/alpha${alpha}-beta${beta}/
gpu_num=$(echo $gpu | awk -F, '{print NF}')

export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_LAUNCH_BLOCKING=1
export WANDB_PROJECT=LETTER-TIGER

echo "Training LETTER-TIGER on ${dataset} with alpha=${alpha}, beta=${beta}, epoch=${epoch} using GPUs ${gpu}."

torchrun --nproc_per_node=${gpu_num} --master_port=${port} ./finetune.py \
    --output_dir ${output_dir} \
    --wandb_run_name ${run_name} \
    --dataset ${dataset} \
    --per_device_batch_size 128 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file .index.epoch${epoch}.alpha${alpha}-beta${beta}.json \
    --temperature 0.7
