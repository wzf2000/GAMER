#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "${script_dir}/.." && pwd)
cd "${repo_dir}"

: ${task:=ctr}
: ${dataset:=ShortVideoAD}
: ${data_path:=${repo_dir}/data}
: ${gpu:=0,1,2,3}
: ${backbone:=Qwen3TemporalHierarchicalFixedSoft}
: ${suffix:=}
: ${conditions:=p3s,base}
: ${strategies:=frozen_probe,cold_start,full_finetune}
: ${max_his_len:=100}
: ${train_batch_size:=1024}
: ${test_batch_size:=1024}
: ${train_port:=2315}
: ${test_port:=2316}
: ${pretrained_model:=${repo_dir}/checkpoint/SMB-decoder/${dataset}/smb_explicit_decoder_4/${backbone}/original}

case "${task}" in
    ctr)
        ranking_task=smb_ctr_ranking_decoder
        ;;
    cvr)
        ranking_task=smb_ranking_decoder
        ;;
    *)
        echo "task must be ctr or cvr; received ${task}."
        exit 1
        ;;
esac

common_env=(
    "dataset=${dataset}"
    "data_path=${data_path}"
    "original=1"
    "rq_kmeans=0"
    "tasks=${ranking_task}"
    "test_task=${ranking_task}"
    "gpu=${gpu}"
    "backbone=${backbone}"
    "max_his_len=${max_his_len}"
)

IFS=',' read -r -a condition_list <<< "${conditions}"
for condition in "${condition_list[@]}"; do
    case "${condition}" in
        p3s)
            candidate_behavior=p3s
            ;;
        base)
            candidate_behavior=target
            ;;
        *)
            echo "Unknown condition ${condition}; expected p3s or base."
            exit 1
            ;;
    esac

    IFS=',' read -r -a strategy_list <<< "${strategies}"
    for strategy in "${strategy_list[@]}"; do
        case "${strategy}" in
            frozen_probe)
                freeze_backbone=1
                cold_start=0
                strategy_pretrained_model="${pretrained_model}"
                ;;
            cold_start)
                freeze_backbone=0
                cold_start=1
                strategy_pretrained_model=""
                ;;
            full_finetune)
                freeze_backbone=0
                cold_start=0
                strategy_pretrained_model="${pretrained_model}"
                ;;
            *)
                echo "Unknown strategy ${strategy}; expected frozen_probe, cold_start, or full_finetune."
                exit 1
                ;;
        esac

        strategy_suffix="${condition}_${strategy}"
        if [[ -n "${suffix}" ]]; then
            strategy_suffix="${suffix}_${strategy_suffix}"
        fi
        echo "Training and testing ${task} condition ${condition}, strategy ${strategy}."

        env "${common_env[@]}" \
            "SMB_RANKING_CANDIDATE_BEHAVIOR=${candidate_behavior}" \
            batch_size="${train_batch_size}" \
            port="${train_port}" \
            suffix="${strategy_suffix}" \
            pretrained_model="${strategy_pretrained_model}" \
            freeze_backbone="${freeze_backbone}" \
            cold_start="${cold_start}" \
            bash "${script_dir}/train_SMB_ranking_decoder.sh"

        env "${common_env[@]}" \
            "SMB_RANKING_CANDIDATE_BEHAVIOR=${candidate_behavior}" \
            batch_size="${test_batch_size}" \
            port="${test_port}" \
            suffix="${strategy_suffix}" \
            ckpt_num=best \
            bash "${script_dir}/test_SMB_ranking_decoder.sh"
    done
done
