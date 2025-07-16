: ${dataset:=Beauty}
: ${rq_kmeans:=0}
: ${batch_size:=1024}
: ${test_task:=seqrec}
: ${filter:=0}
: ${gpu:=0,1,2,3}
: ${port:=2314}

export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

data_path=./data
gpu_num=$(echo $gpu | awk -F, '{print NF}')
per_device_batch_size=$(($batch_size / $gpu_num))

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
                results_file=./results/${dataset}/results-alpha${alpha}-beta${beta}.json
                ckpt_path=./checkpoint/decoder/${dataset}/alpha${alpha}-beta${beta}/
                index_file=.index.epoch${epoch}.alpha${alpha}-beta${beta}.json
                echo "Testing decoder on ${dataset} using RQ-VAE with alpha=${alpha}, beta=${beta}, epoch=${epoch} using GPU ${gpu}."
            else
                results_file=./results/${dataset}/results-original.json
                ckpt_path=./checkpoint/decoder/${dataset}/original/
                index_file=.index.json
                echo "Testing decoder on ${dataset} using original index file from LETTER repository."
            fi
        else
            results_file=./results/${dataset}/results-rid.json
            ckpt_path=./checkpoint/decoder/${dataset}/rid/
            index_file=.index.rid.json
            echo "Testing decoder on ${dataset} using random ID tokenization."
        fi
    else
        : ${chunk_size:=64}
        results_file=./results/${dataset}/results-cid-${chunk_size}.json
        ckpt_path=./checkpoint/decoder/${dataset}/cid-${chunk_size}/
        index_file=.index.cid.chunk${chunk_size}.json
        echo "Testing decoder on ${dataset} using chunked ID tokenization with chunk size ${chunk_size}."
    fi
else
    : ${cf_emb:=0}
    if [ $cf_emb -eq 0 ]; then
        results_file=./results/${dataset}/results-rq-kmeans.json
        ckpt_path=./checkpoint/decoder/${dataset}/rq-kmeans/
        index_file=.index.rq-kmeans.json
        echo "Testing decoder on ${dataset} using RQ-Kmeans without CF embeddings."
    else
        : ${reduce:=0}
        if [ $reduce -eq 0 ]; then
            results_file=./results/${dataset}/results-rq-kmeans-cf.json
            ckpt_path=./checkpoint/decoder/${dataset}/rq-kmeans-cf/
            index_file=.index.rq-kmeans-cf.json
            echo "Testing decoder on ${dataset} using RQ-Kmeans with CF embeddings."
        else
            results_file=./results/${dataset}/results-rq-kmeans-cf-reduce.json
            ckpt_path=./checkpoint/decoder/${dataset}/rq-kmeans-cf-reduce/
            index_file=.index.rq-kmeans-cf-reduce.json
            echo "Testing decoder on ${dataset} using RQ-Kmeans with CF embeddings and reduced semantic embeddings."
        fi
    fi
fi

if [ $filter -eq 0 ]; then
    filter_flag=""
else
    filter_flag="--filter"
fi

torchrun --nproc_per_node=${gpu_num} --master_port=${port} ./main.py test_decoder \
    --ckpt_path ${ckpt_path} \
    --dataset ${dataset} \
    --data_path ${data_path} \
    --results_file ${results_file} \
    --test_batch_size ${per_device_batch_size} \
    --num_beams 20 \
    --index_file ${index_file} \
    --test_task ${test_task} \
    ${filter_flag}
