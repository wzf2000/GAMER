#!/bin/bash
: ${dataset:=Retail}
: ${rq_kmeans:=0}
: ${batch_size:=512}
: ${learning_rate:=5e-4}
: ${tasks:=smb_explicit}
: ${gpu:=0,1,2,3}
: ${epochs:=200}
: ${port:=2314}
: ${backbone:=TIGER}

export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

gpu_num=$(echo $gpu | awk -F, '{print NF}')
per_device_batch_size=$(($batch_size / $gpu_num))
task_dir=${tasks//,/-}
backbone_arg=${backbone}

if [ "${backbone}" = "TIGER" ]; then
    base_model=./ckpt/s2s-models/TIGER
elif [ "${backbone}" = "PBATransformer" ]; then
    base_model=./ckpt/s2s-models/PBATransformer
elif [ "${backbone}" = "Qwen3" ]; then
    base_model=./ckpt/s2s-models/Qwen3-Light
elif [ "${backbone}" = "Qwen3Moe" ]; then
    base_model=./ckpt/s2s-models/Qwen3Moe
elif [ "${backbone}" = "Qwen3ActionMoe" ]; then
    base_model=./ckpt/s2s-models/Qwen3ActionMoe
elif [ "${backbone}" = "Qwen3Session" ]; then
    base_model=./ckpt/s2s-models/Qwen3-Light
elif [ "${backbone}" = "Qwen3Session2" ]; then
    base_model=./ckpt/s2s-models/Qwen3-Light-2
    backbone_arg=Qwen3Session
elif [ "${backbone}" = "Qwen3SessionMoe" ]; then
    base_model=./ckpt/s2s-models/Qwen3SessionMoe
elif [ "${backbone}" = "Qwen3Multi" ]; then
    base_model=./ckpt/s2s-models/Qwen3Multi
elif [ "${backbone}" = "Qwen3SessionMulti" ]; then
    base_model=./ckpt/s2s-models/Qwen3SessionMulti
else
    echo "Unsupported backbone model: ${backbone}."
    exit 1
fi

task_dir=${dataset}/${task_dir}/${backbone}

: ${suffix:=}
if [ "${suffix}" != "" ]; then
    task_dir=${task_dir}_${suffix}
fi


if [ $rq_kmeans -eq 0 ]; then
    : ${cid:=0}
    if [ $cid -eq 0 ]; then
        : ${rid:=0}
        if [ $rid -eq 0 ]; then
            : ${original:=0}
            if [ $original -eq 0 ]; then
                : ${alpha:=0.02}
                : ${beta:=0.0001}
                : ${epoch:=20000}
                output_dir=./checkpoint/SMB-decoder/${task_dir}/alpha${alpha}-beta${beta}/
                run_name=${task_dir}/alpha${alpha}-beta${beta}/
                index_file=.index.epoch${epoch}.alpha${alpha}-beta${beta}.json
                echo "Training SMB Decoder on ${dataset} with alpha=${alpha}, beta=${beta}, epoch=${epoch} using GPUs ${gpu}."
            else
                output_dir=./checkpoint/SMB-decoder/${task_dir}/original/
                run_name=${task_dir}/original/
                index_file=.index.json
                echo "Training SMB Decoder on ${dataset} using original index file from LETTER repository."
            fi
        else
            output_dir=./checkpoint/SMB-decoder/${task_dir}/rid/
            run_name=${task_dir}/rid/
            index_file=.index.rid.json
            echo "Training SMB Decoder on ${dataset} using random ID tokenization."
        fi
    else
        : ${chunk_size:=64}
        : ${shuffle:=0}
        if [ $shuffle -eq 1 ]; then
            output_dir=./checkpoint/SMB-decoder/${task_dir}/cid-shuffle-${chunk_size}/
            run_name=${task_dir}/cid-shuffle-${chunk_size}/
            index_file=.index.cid.shuffle.chunk${chunk_size}.json
            echo "Training SMB Decoder on ${dataset} using chunked ID tokenization with chunk size ${chunk_size} and shuffling."
        else
            output_dir=./checkpoint/SMB-decoder/${task_dir}/cid-${chunk_size}/
            run_name=${task_dir}/cid-${chunk_size}/
            index_file=.index.cid.chunk${chunk_size}.json
            echo "Training SMB Decoder on ${dataset} using chunked ID tokenization with chunk size ${chunk_size}."
        fi
    fi
else
    : ${cf_emb:=0}
    if [ $cf_emb -eq 0 ]; then
        output_dir=./checkpoint/SMB-decoder/${task_dir}/rq-kmeans/
        run_name=${task_dir}/rq-kmeans/
        index_file=.index.rq-kmeans.json
        echo "Training SMB Decoder on ${dataset} using RQ-Kmeans without CF embeddings."
    else
        : ${reduce:=0}
        if [ $reduce -eq 0 ]; then
            output_dir=./checkpoint/SMB-decoder/${task_dir}/rq-kmeans-cf/
            run_name=${task_dir}/rq-kmeans-cf/
            index_file=.index.rq-kmeans-cf.json
            echo "Training SMB Decoder on ${dataset} using RQ-Kmeans with CF embeddings."
        else
            output_dir=./checkpoint/SMB-decoder/${task_dir}/rq-kmeans-cf-reduce/
            run_name=${task_dir}/rq-kmeans-cf-reduce/
            index_file=.index.rq-kmeans-cf-reduce.json
            echo "Training SMB Decoder on ${dataset} using RQ-Kmeans with CF embeddings and reduced semantic embeddings."
        fi
    fi
fi

: ${extra_args:=}
# transform the format of "X=a,Y=b" into "-X a -Y b"
extra_args_out=$(echo "$extra_args" | awk -F, '{
    for(i=1; i<=NF; i++) {
        split($i, arr, "=")
        printf "--%s %s ", arr[1], arr[2]
    }
}')
echo "Extra arguments: ${extra_args_out}"

if [ $gpu_num -eq 1 ]; then
    echo "Using single GPU: ${gpu}"
    python main.py train_SMB_decoder \
        --backbone ${backbone_arg} \
        --base_model ${base_model} \
        --output_dir ${output_dir} \
        --wandb_run_name ${run_name} \
        --dataset ${dataset} \
        --per_device_batch_size ${per_device_batch_size} \
        --learning_rate ${learning_rate} \
        --tasks ${tasks} \
        --epochs ${epochs} \
        --index_file ${index_file} \
        --temperature 0.7 \
        ${extra_args_out}
else
    echo "Using multiple GPUs: ${gpu}"
    torchrun --nproc_per_node=${gpu_num} --master_port=${port} ./main.py train_SMB_decoder \
        --backbone ${backbone_arg} \
        --base_model ${base_model} \
        --output_dir ${output_dir} \
        --wandb_run_name ${run_name} \
        --dataset ${dataset} \
        --per_device_batch_size ${per_device_batch_size} \
        --learning_rate ${learning_rate} \
        --tasks ${tasks} \
        --epochs ${epochs} \
        --index_file ${index_file} \
        --temperature 0.7 \
        ${extra_args_out}
fi
