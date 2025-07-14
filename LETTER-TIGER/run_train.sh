: ${dataset:=Yelp}
: ${rq_kmeans:=0}
: ${batch_size:=512}
: ${learning_rate:=5e-4}
: ${gpu:=0,1,2,3}
: ${epochs:=200}
: ${port:=2314}

export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_LAUNCH_BLOCKING=1
export WANDB_PROJECT=LETTER-TIGER

gpu_num=$(echo $gpu | awk -F, '{print NF}')
per_device_batch_size=$(($batch_size / $gpu_num))

if [ $rq_kmeans -eq 0 ]; then
    : ${original:=0}
    if [ $original -eq 0 ]; then
        : ${alpha:=0.02}
        : ${beta:=0.0001}
        : ${epoch:=20000}
        output_dir=./ckpt/${dataset}/alpha${alpha}-beta${beta}/
        run_name=${dataset}/alpha${alpha}-beta${beta}/
        index_file=.index.epoch${epoch}.alpha${alpha}-beta${beta}.json
        echo "Training LETTER-TIGER on ${dataset} with alpha=${alpha}, beta=${beta}, epoch=${epoch} using GPUs ${gpu}."
    else
        output_dir=./ckpt/${dataset}/original/
        run_name=${dataset}/original/
        index_file=.index.json
        echo "Training LETTER-TIGER on ${dataset} using original index file from LETTER repository."
    fi
else
    : ${cf_emb:=0}
    if [ $cf_emb -eq 0 ]; then
        output_dir=./ckpt/${dataset}/rq-kmeans/
        run_name=${dataset}/rq-kmeans/
        index_file=.index.rq-kmeans.json
        echo "Training LETTER-TIGER on ${dataset} using RQ-Kmeans without CF embeddings."
    else
        : ${reduce:=0}
        if [ $reduce -eq 0 ]; then
            output_dir=./ckpt/${dataset}/rq-kmeans-cf/
            run_name=${dataset}/rq-kmeans-cf/
            index_file=.index.rq-kmeans-cf.json
            echo "Training LETTER-TIGER on ${dataset} using RQ-Kmeans with CF embeddings."
        else
            output_dir=./ckpt/${dataset}/rq-kmeans-cf-reduce/
            run_name=${dataset}/rq-kmeans-cf-reduce/
            index_file=.index.rq-kmeans-cf-reduce.json
            echo "Training LETTER-TIGER on ${dataset} using RQ-Kmeans with CF embeddings and reduced semantic embeddings."
        fi
    fi
fi

torchrun --nproc_per_node=${gpu_num} --master_port=${port} ./finetune.py \
    --output_dir ${output_dir} \
    --wandb_run_name ${run_name} \
    --dataset ${dataset} \
    --per_device_batch_size ${per_device_batch_size} \
    --learning_rate ${learning_rate} \
    --epochs ${epochs} \
    --index_file ${index_file} \
    --temperature 0.7
