#!/bin/bash

resolve_tokenization() {
    : ${rq_kmeans:=0}
    if [ "${rq_kmeans}" -eq 0 ] 2>/dev/null; then
        : ${cid:=0}
        if [ "${cid}" -eq 0 ]; then
            : ${rid:=0}
            if [ "${rid}" -eq 0 ]; then
                : ${original:=0}
                if [ "${original}" -eq 0 ]; then
                    : ${alpha:=0.02}
                    : ${beta:=0.0001}
                    : ${epoch:=20000}
                    token_tag=alpha${alpha}-beta${beta}
                    index_file=.index.epoch${epoch}.alpha${alpha}-beta${beta}.json
                    tokenization_desc="RQ-VAE alpha=${alpha}, beta=${beta}, epoch=${epoch}"
                else
                    token_tag=original
                    index_file=.index.json
                    tokenization_desc="original index"
                fi
            else
                token_tag=rid
                index_file=.index.rid.json
                tokenization_desc="random ID tokenization"
            fi
        else
            : ${chunk_size:=64}
            : ${shuffle:=0}
            if [ "${shuffle}" -eq 1 ]; then
                token_tag=cid-shuffle-${chunk_size}
                index_file=.index.cid.shuffle.chunk${chunk_size}.json
                tokenization_desc="chunked ID tokenization with chunk size ${chunk_size} and shuffling"
            else
                token_tag=cid-${chunk_size}
                index_file=.index.cid.chunk${chunk_size}.json
                tokenization_desc="chunked ID tokenization with chunk size ${chunk_size}"
            fi
        fi
    else
        : ${cf_emb:=0}
        if [ "${cf_emb}" -eq 0 ] 2>/dev/null || [ "${cf_emb}" = "None" ]; then
            token_tag=rq-kmeans
            index_file=.index.rq-kmeans.json
            tokenization_desc="RQ-Kmeans without CF embeddings"
        else
            : ${reduce:=0}
            if [ "${reduce}" -eq 0 ]; then
                token_tag=rq-kmeans-cf
                index_file=.index.rq-kmeans-cf.json
                tokenization_desc="RQ-Kmeans with CF embeddings"
            else
                token_tag=rq-kmeans-cf-reduce
                index_file=.index.rq-kmeans-cf-reduce.json
                tokenization_desc="RQ-Kmeans with CF embeddings and reduced semantic embeddings"
            fi
        fi
    fi
}

