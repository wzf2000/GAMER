#!/bin/bash
: ${dataset:=ShortVideoAD}
: ${tasks:=smb_explicit}
: ${test_task:=smb_explicit}
: ${gpu:=0}
: ${backbone:=Qwen3Multi}
: ${num_beams:=20}
: ${max_users:=20}
: ${batch_size:=16}

export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

data_path=./data
gpu_num=$(echo $gpu | awk -F, '{print NF}')
per_device_batch_size=$(($batch_size / $gpu_num))
backbone_arg=${backbone}

if [ "${backbone}" = "Qwen3Session2" ]; then
    backbone_arg=Qwen3Session
elif [ "${backbone}" = "Llama" ]; then
    backbone_arg=LlamaMulti
elif [[ "${backbone}" == Qwen3Multi* ]]; then
    backbone_arg=Qwen3Multi
fi

task_dir=${tasks//,/-}
task_dir=${dataset}/${task_dir}/${backbone}

: ${suffix:=}
if [ "${suffix}" != "" ]; then
    task_dir=${task_dir}_${suffix}
fi

if [ $rq_kmeans -eq 0 ] 2>/dev/null || [ "${rq_kmeans:-0}" = "0" ]; then
    : ${cid:=0}
    if [ $cid -eq 0 ]; then
        : ${rid:=0}
        if [ $rid -eq 0 ]; then
            : ${original:=0}
            if [ $original -eq 0 ]; then
                : ${alpha:=0.02}
                : ${beta:=0.0001}
                : ${epoch:=20000}
                results_file=./results/${task_dir}/behavior_dropout-${test_task}-alpha${alpha}-beta${beta}.json
                ckpt_path=./checkpoint/SMB-decoder/${task_dir}/alpha${alpha}-beta${beta}/
                index_file=.index.epoch${epoch}.alpha${alpha}-beta${beta}.json
                echo "Behavior dropout analysis on ${dataset} (RQ-VAE alpha=${alpha}, beta=${beta}, epoch=${epoch}) using GPU ${gpu}."
            else
                results_file=./results/${task_dir}/behavior_dropout-${test_task}-original.json
                ckpt_path=./checkpoint/SMB-decoder/${task_dir}/original/
                index_file=.index.json
                echo "Behavior dropout analysis on ${dataset} using original index file."
            fi
        else
            results_file=./results/${task_dir}/behavior_dropout-${test_task}-rid.json
            ckpt_path=./checkpoint/SMB-decoder/${task_dir}/rid/
            index_file=.index.rid.json
            echo "Behavior dropout analysis on ${dataset} using random ID tokenization."
        fi
    else
        : ${chunk_size:=64}
        : ${shuffle:=0}
        if [ $shuffle -eq 1 ]; then
            results_file=./results/${task_dir}/behavior_dropout-${test_task}-cid-shuffle-${chunk_size}.json
            ckpt_path=./checkpoint/SMB-decoder/${task_dir}/cid-shuffle-${chunk_size}/
            index_file=.index.cid.shuffle.chunk${chunk_size}.json
            echo "Behavior dropout analysis on ${dataset} using chunked ID (shuffle, chunk=${chunk_size})."
        else
            results_file=./results/${task_dir}/behavior_dropout-${test_task}-cid-${chunk_size}.json
            ckpt_path=./checkpoint/SMB-decoder/${task_dir}/cid-${chunk_size}/
            index_file=.index.cid.chunk${chunk_size}.json
            echo "Behavior dropout analysis on ${dataset} using chunked ID (chunk=${chunk_size})."
        fi
    fi
else
    : ${cf_emb:=0}
    if [ $cf_emb -eq 0 ]; then
        results_file=./results/${task_dir}/behavior_dropout-${test_task}-rq-kmeans.json
        ckpt_path=./checkpoint/SMB-decoder/${task_dir}/rq-kmeans/
        index_file=.index.rq-kmeans.json
        echo "Behavior dropout analysis on ${dataset} using RQ-Kmeans."
    else
        : ${reduce:=0}
        if [ $reduce -eq 0 ]; then
            results_file=./results/${task_dir}/behavior_dropout-${test_task}-rq-kmeans-cf.json
            ckpt_path=./checkpoint/SMB-decoder/${task_dir}/rq-kmeans-cf/
            index_file=.index.rq-kmeans-cf.json
            echo "Behavior dropout analysis on ${dataset} using RQ-Kmeans with CF embeddings."
        else
            results_file=./results/${task_dir}/behavior_dropout-${test_task}-rq-kmeans-cf-reduce.json
            ckpt_path=./checkpoint/SMB-decoder/${task_dir}/rq-kmeans-cf-reduce/
            index_file=.index.rq-kmeans-cf-reduce.json
            echo "Behavior dropout analysis on ${dataset} using RQ-Kmeans with CF+reduce."
        fi
    fi
fi

: ${ckpt_num:=best}
if [[ "$ckpt_num" != "best" ]]; then
    ckpt_path=${ckpt_path}checkpoint-${ckpt_num}/
    echo "Using checkpoint from step ${ckpt_num}."
else
    echo "Using the best checkpoint."
fi

: ${target_behavior:=}

: ${extra_args:=}
extra_args_out=$(echo "$extra_args" | awk -F, '{
    for(i=1; i<=NF; i++) {
        split($i, arr, "=")
        printf "--%s %s ", arr[1], arr[2]
    }
}')
echo "Extra arguments: ${extra_args_out}"

: ${extra_flags:=}
extra_flags_out=$(echo "$extra_flags" | awk -F, '{for(i=1; i<=NF; i++) printf "--%s ", $i}')
echo "Extra flags: ${extra_flags_out}"

target_behavior_arg=""
if [ "${target_behavior}" != "" ]; then
    target_behavior_arg="--target_behavior ${target_behavior}"
fi

# This task is single-process only (post-hoc analysis on a small user subset)
echo "Running behavior dropout analysis (max_users=${max_users}, num_beams=${num_beams})."
python main.py analyze_behavior_dropout \
    --backbone ${backbone_arg} \
    --ckpt_path ${ckpt_path} \
    --dataset ${dataset} \
    --data_path ${data_path} \
    --results_file ${results_file} \
    --test_batch_size ${per_device_batch_size} \
    --num_beams ${num_beams} \
    --max_users ${max_users} \
    --index_file ${index_file} \
    --test_task ${test_task} \
    ${target_behavior_arg} \
    ${extra_args_out} \
    ${extra_flags_out}
