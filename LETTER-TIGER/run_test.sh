: ${dataset:=Yelp}
: ${rq_kmeans:=0}
: ${gpu:=0}
: ${port:=2314}

export CUDA_LAUNCH_BLOCKING=1

data_path=../data
results_file=./results/${dataset}/results-alpha${alpha}-beta${beta}.json
ckpt_path=./ckpt/${dataset}/alpha${alpha}-beta${beta}/

if [ $rq_kmeans -eq 0 ]; then
    : ${original:=0}
    if [ $original -eq 0 ]; then
        : ${alpha:=0.02}
        : ${beta:=0.0001}
        : ${epoch:=20000}
        results_file=./results/${dataset}/results-alpha${alpha}-beta${beta}.json
        ckpt_path=./ckpt/${dataset}/alpha${alpha}-beta${beta}/
        index_file=.index.epoch${epoch}.alpha${alpha}-beta${beta}.json
        echo "Testing LETTER-TIGER on ${dataset} with alpha=${alpha}, beta=${beta}, epoch=${epoch} using GPU ${gpu}."
    else
        results_file=./results/${dataset}/results-original.json
        ckpt_path=./ckpt/${dataset}/original/
        index_file=.index.json
        echo "Testing LETTER-TIGER on ${dataset} using original index file from LETTER repository."
    fi
else
    : ${cf_emb:=0}
    if [ $cf_emb -eq 0 ]; then
        results_file=./results/${dataset}/results-rq-kmeans.json
        ckpt_path=./ckpt/${dataset}/rq-kmeans/
        index_file=.index.rq-kmeans.json
        echo "Testing LETTER-TIGER on ${dataset} using RQ-Kmeans without CF embeddings."
    else
        : ${reduce:=0}
        if [ $reduce -eq 0 ]; then
            results_file=./results/${dataset}/results-rq-kmeans-cf.json
            ckpt_path=./ckpt/${dataset}/rq-kmeans-cf/
            index_file=.index.rq-kmeans-cf.json
            echo "Testing LETTER-TIGER on ${dataset} using RQ-Kmeans with CF embeddings."
        else
            results_file=./results/${dataset}/results-rq-kmeans-cf-reduce.json
            ckpt_path=./ckpt/${dataset}/rq-kmeans-cf-reduce/
            index_file=.index.rq-kmeans-cf-reduce.json
            echo "Testing LETTER-TIGER on ${dataset} using RQ-Kmeans with CF embeddings and reduced semantic embeddings."
        fi
    fi
fi

python test.py \
    --gpu_id ${gpu} \
    --ckpt_path ${ckpt_path} \
    --dataset ${dataset} \
    --data_path ${data_path} \
    --results_file ${results_file} \
    --test_batch_size 1024 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file ${index_file} \
