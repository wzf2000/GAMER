#!/bin/bash
: ${dataset:=Retail}
: ${rq_kmeans:=0}
: ${batch_size:=1024}
: ${tasks=smb_explicit}
: ${test_task:=smb_explicit}
: ${gpu:=0,1,2,3}
: ${port:=2314}
: ${backbone:=TIGER}

export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

data_path=./data
gpu_num=$(echo $gpu | awk -F, '{print NF}')
per_device_batch_size=$(($batch_size / $gpu_num))

task_dir=${tasks//,/-}
task_dir=${dataset}/${task_dir}/${backbone}

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
                results_file=./results/${task_dir}/results-alpha${alpha}-beta${beta}.json
                ckpt_path=./checkpoint/SMB-decoder/${task_dir}/alpha${alpha}-beta${beta}/
                index_file=.index.epoch${epoch}.alpha${alpha}-beta${beta}.json
                echo "Testing SMB decoder on ${dataset} using RQ-VAE with alpha=${alpha}, beta=${beta}, epoch=${epoch} using GPU ${gpu}."
            else
                results_file=./results/${task_dir}/results-original.json
                ckpt_path=./checkpoint/SMB-decoder/${task_dir}/original/
                index_file=.index.json
                echo "Testing SMB decoder on ${dataset} using original index file from LETTER repository."
            fi
        else
            results_file=./results/${task_dir}/results-rid.json
            ckpt_path=./checkpoint/SMB-decoder/${task_dir}/rid/
            index_file=.index.rid.json
            echo "Testing SMB decoder on ${dataset} using random ID tokenization."
        fi
    else
        : ${chunk_size:=64}
        : ${shuffle:=0}
        if [ $shuffle -eq 1 ]; then
            results_file=./results/${task_dir}/results-cid-shuffle-${chunk_size}.json
            ckpt_path=./checkpoint/SMB-decoder/${task_dir}/cid-shuffle-${chunk_size}/
            index_file=.index.cid.shuffle.chunk${chunk_size}.json
            echo "Testing SMB decoder on ${dataset} using chunked ID tokenization with chunk size ${chunk_size} and shuffling."
        else
            results_file=./results/${task_dir}/results-cid-${chunk_size}.json
            ckpt_path=./checkpoint/SMB-decoder/${task_dir}/cid-${chunk_size}/
            index_file=.index.cid.chunk${chunk_size}.json
            echo "Testing SMB decoder on ${dataset} using chunked ID tokenization with chunk size ${chunk_size}."
        fi
    fi
else
    : ${cf_emb:=0}
    if [ $cf_emb -eq 0 ]; then
        results_file=./results/${task_dir}/results-rq-kmeans.json
        ckpt_path=./checkpoint/SMB-decoder/${task_dir}/rq-kmeans/
        index_file=.index.rq-kmeans.json
        echo "Testing SMB decoder on ${dataset} using RQ-Kmeans without CF embeddings."
    else
        : ${reduce:=0}
        if [ $reduce -eq 0 ]; then
            results_file=./results/${task_dir}/results-rq-kmeans-cf.json
            ckpt_path=./checkpoint/SMB-decoder/${task_dir}/rq-kmeans-cf/
            index_file=.index.rq-kmeans-cf.json
            echo "Testing SMB decoder on ${dataset} using RQ-Kmeans with CF embeddings."
        else
            results_file=./results/${task_dir}/results-rq-kmeans-cf-reduce.json
            ckpt_path=./checkpoint/SMB-decoder/${task_dir}/rq-kmeans-cf-reduce/
            index_file=.index.rq-kmeans-cf-reduce.json
            echo "Testing SMB decoder on ${dataset} using RQ-Kmeans with CF embeddings and reduced semantic embeddings."
        fi
    fi
fi

: ${ckpt_num:=best}
if [[ "$ckpt_num" == "best" ]]; then
    # no changes needed
    echo "Using the best checkpoint."
else
    ckpt_path=${ckpt_path}checkpoint-${ckpt_num}/
    echo "Using checkpoint from step ${ckpt_num}."
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

: ${extra_flags:=}
# transform the format of "X,Y" into "--X --Y"
extra_flags_out=$(echo "$extra_flags" | awk -F, '{for(i=1; i<=NF; i++) printf "--%s ", $i}')
echo "Extra flags: ${extra_flags_out}"

if [ $gpu_num -eq 1 ]; then
    echo "Using single GPU: ${gpu}"
    python main.py test_SMB_decoder \
        --backbone ${backbone} \
        --ckpt_path ${ckpt_path} \
        --dataset ${dataset} \
        --data_path ${data_path} \
        --results_file ${results_file} \
        --test_batch_size ${per_device_batch_size} \
        --num_beams 20 \
        --index_file ${index_file} \
        --test_task ${test_task} \
        ${extra_args_out} \
        ${extra_flags_out}
else
    echo "Using multiple GPUs: ${gpu}"
    torchrun --nproc_per_node=${gpu_num} --master_port=${port} ./main.py test_SMB_decoder \
        --backbone ${backbone} \
        --ckpt_path ${ckpt_path} \
        --dataset ${dataset} \
        --data_path ${data_path} \
        --results_file ${results_file} \
        --test_batch_size ${per_device_batch_size} \
        --num_beams 20 \
        --index_file ${index_file} \
        --test_task ${test_task} \
        ${extra_args_out} \
        ${extra_flags_out}
fi
