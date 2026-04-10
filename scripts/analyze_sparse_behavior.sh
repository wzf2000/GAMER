#!/bin/bash
: ${dataset:=ShortVideoAD}
: ${tasks:=smb_explicit}
: ${test_task:=smb_explicit}
: ${gpu:=0}
: ${backbone:=Qwen3Multi}
: ${baseline_backbone:=$backbone}
: ${num_beams:=20}
: ${batch_size:=16}
: ${metrics:=hit@10,ndcg@10}
: ${bucket_thresholds:=3,6}
: ${max_sparse_count:=2}
: ${max_interesting_users:=20}
: ${interesting_top_k:=10}

export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

data_path=./data
gpu_num=$(echo $gpu | awk -F, '{print NF}')
per_device_batch_size=$(($batch_size / $gpu_num))
backbone_arg=${backbone}
baseline_backbone_arg=${baseline_backbone}

if [ "${backbone}" = "Qwen3Session2" ]; then
    backbone_arg=Qwen3Session
elif [ "${backbone}" = "Llama" ]; then
    backbone_arg=LlamaMulti
elif [[ "${backbone}" == Qwen3Multi* ]]; then
    backbone_arg=Qwen3Multi
fi

if [ "${baseline_backbone}" = "Qwen3Session2" ]; then
    baseline_backbone_arg=Qwen3Session
elif [ "${baseline_backbone}" = "Llama" ]; then
    baseline_backbone_arg=LlamaMulti
elif [[ "${baseline_backbone}" == Qwen3Multi* ]]; then
    baseline_backbone_arg=Qwen3Multi
fi

task_dir=${tasks//,/-}
task_dir=${dataset}/${task_dir}/${backbone}

: ${suffix:=}
if [ "${suffix}" != "" ]; then
    task_dir=${task_dir}_${suffix}
fi

# ---------- Our model checkpoint ----------
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
                ckpt_path=./checkpoint/SMB-decoder/${task_dir}/alpha${alpha}-beta${beta}/
                index_file=.index.epoch${epoch}.alpha${alpha}-beta${beta}.json
                ckpt_tag=alpha${alpha}-beta${beta}
                echo "Our model: RQ-VAE alpha=${alpha}, beta=${beta}, epoch=${epoch}."
            else
                ckpt_path=./checkpoint/SMB-decoder/${task_dir}/original/
                index_file=.index.json
                ckpt_tag=original
                echo "Our model: original index."
            fi
        else
            ckpt_path=./checkpoint/SMB-decoder/${task_dir}/rid/
            index_file=.index.rid.json
            ckpt_tag=rid
            echo "Our model: random-ID tokenization."
        fi
    else
        : ${chunk_size:=64}
        : ${shuffle:=0}
        if [ $shuffle -eq 1 ]; then
            ckpt_path=./checkpoint/SMB-decoder/${task_dir}/cid-shuffle-${chunk_size}/
            index_file=.index.cid.shuffle.chunk${chunk_size}.json
            ckpt_tag=cid-shuffle-${chunk_size}
            echo "Our model: chunked ID (shuffle, chunk=${chunk_size})."
        else
            ckpt_path=./checkpoint/SMB-decoder/${task_dir}/cid-${chunk_size}/
            index_file=.index.cid.chunk${chunk_size}.json
            ckpt_tag=cid-${chunk_size}
            echo "Our model: chunked ID (chunk=${chunk_size})."
        fi
    fi
else
    : ${cf_emb:=0}
    if [ $cf_emb -eq 0 ]; then
        ckpt_path=./checkpoint/SMB-decoder/${task_dir}/rq-kmeans/
        index_file=.index.rq-kmeans.json
        ckpt_tag=rq-kmeans
        echo "Our model: RQ-Kmeans."
    else
        : ${reduce:=0}
        if [ $reduce -eq 0 ]; then
            ckpt_path=./checkpoint/SMB-decoder/${task_dir}/rq-kmeans-cf/
            index_file=.index.rq-kmeans-cf.json
            ckpt_tag=rq-kmeans-cf
            echo "Our model: RQ-Kmeans + CF embeddings."
        else
            ckpt_path=./checkpoint/SMB-decoder/${task_dir}/rq-kmeans-cf-reduce/
            index_file=.index.rq-kmeans-cf-reduce.json
            ckpt_tag=rq-kmeans-cf-reduce
            echo "Our model: RQ-Kmeans + CF + reduce."
        fi
    fi
fi

: ${ckpt_num:=best}
if [[ "$ckpt_num" != "best" ]]; then
    ckpt_path=${ckpt_path}checkpoint-${ckpt_num}/
    echo "Our model: using checkpoint from step ${ckpt_num}."
else
    echo "Our model: using the best checkpoint."
fi

# ---------- Baseline checkpoint ----------
# The baseline checkpoint must be provided explicitly via baseline_ckpt_path.
# Optionally set baseline_index_file if the baseline uses a different index.
: ${baseline_ckpt_path:=}
if [ -z "${baseline_ckpt_path}" ]; then
    echo "ERROR: baseline_ckpt_path is not set. Please set it before calling this script."
    exit 1
fi
echo "Baseline model: ${baseline_backbone_arg} from ${baseline_ckpt_path}."

# ---------- Results file ----------
results_file=./results/${task_dir}/sparse_behavior-${test_task}-${ckpt_tag}-vs-baseline.json

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

echo "Running sparse-behavior analysis (batch_size=${per_device_batch_size}, num_beams=${num_beams})."
echo "  Buckets: ${bucket_thresholds} | max_sparse_count: ${max_sparse_count} | interesting_top_k: ${interesting_top_k}"
python main.py analyze_sparse_behavior \
    --backbone ${backbone_arg} \
    --ckpt_path ${ckpt_path} \
    --baseline_backbone ${baseline_backbone_arg} \
    --baseline_ckpt_path ${baseline_ckpt_path} \
    --dataset ${dataset} \
    --data_path ${data_path} \
    --results_file ${results_file} \
    --test_batch_size ${per_device_batch_size} \
    --num_beams ${num_beams} \
    --index_file ${index_file} \
    --test_task ${test_task} \
    --metrics ${metrics} \
    --bucket_thresholds ${bucket_thresholds} \
    --max_sparse_count ${max_sparse_count} \
    --max_interesting_users ${max_interesting_users} \
    --interesting_top_k ${interesting_top_k} \
    ${target_behavior_arg} \
    ${extra_args_out} \
    ${extra_flags_out}
