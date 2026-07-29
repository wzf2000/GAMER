#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "${script_dir}/.." && pwd)
cd "${repo_dir}"

: ${checkpoint_root:=/home/zhouman/guoyunhe/workspace/full/GAMER-rank/checkpoint/SMB-decoder/ShortVideoAD/smb_explicit_decoder_4/Qwen3TemporalHierarchicalFixedSoft}
: ${gpu:=0,1,2,3}
: ${experiments:=all}

IFS=',' read -r -a gpu_ids <<< "${gpu}"
if [[ "${#gpu_ids[@]}" -lt 4 ]]; then
    echo "CTR all-experiment launcher requires four GPU IDs; received ${gpu}."
    exit 1
fi

if [[ -f "${checkpoint_root}/model.safetensors" || -f "${checkpoint_root}/pytorch_model.bin" ]]; then
    pretrained_model="${checkpoint_root}"
elif [[ -f "${checkpoint_root}/original/model.safetensors" || -f "${checkpoint_root}/original/pytorch_model.bin" ]]; then
    pretrained_model="${checkpoint_root}/original"
else
    echo "No model checkpoint found under ${checkpoint_root} or ${checkpoint_root}/original."
    exit 1
fi

run_gamer() {
    local mode="$1"
    local port="$2"
    local cold_start=0
    local freeze_backbone=1
    local mode_checkpoint="${pretrained_model}"

    case "${mode}" in
        frozen_probe) ;;
        cold_start)
            cold_start=1
            mode_checkpoint=""
            ;;
        full_finetune)
            freeze_backbone=0
            ;;
        *)
            echo "Unknown GAMER CTR mode: ${mode}."
            exit 1
            ;;
    esac

    echo "Starting GAMER CTR ${mode} on GPUs ${gpu}."
    env \
        dataset=ShortVideoAD \
        data_path=/home/zhouman/guoyunhe/workspace/full/GAMER-rank/data \
        original=1 \
        rq_kmeans=0 \
        batch_size=1024 \
        tasks=smb_ctr_ranking_decoder \
        max_his_len=100 \
        gpu="${gpu}" \
        port="${port}" \
        backbone=Qwen3TemporalHierarchicalFixedSoft \
        epochs=3 \
        learning_rate=1e-3 \
        weight_decay=0.01 \
        patience=2 \
        save_total_limit=5 \
        train_auc_samples=2048 \
        train_auc_batch_size=256 \
        eval_epochs=1 \
        suffix="${mode}" \
        pretrained_model="${mode_checkpoint}" \
        freeze_backbone="${freeze_backbone}" \
        cold_start="${cold_start}" \
        extra_args= \
        extra_flags= \
        bash "${script_dir}/train_SMB_ranking_decoder.sh" \
        --seed 42 \
        --optim adamw_torch \
        --gradient_accumulation_steps 2 \
        --logging_step 30 \
        --model_max_length 1024 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type cosine \
        --save_and_eval_strategy epoch \
        --save_and_eval_steps 1000 \
        --temperature 0.7
}

run_baseline() {
    local backbone="$1"
    local device="$2"
    local task=smb_ctr_din
    if [[ "${backbone}" == "DSIN" ]]; then
        task=smb_ctr_dsin
    fi
    local epochs=3
    local save_epoch_limit=3
    local ckpt_num=best
    local suffix=item_id
    if [[ "${backbone}" == "BSTCVR" ]]; then
        suffix=item_id_new
    fi
    echo "Starting ${backbone} CTR baseline on GPU ${device}."
    env \
        dataset=ShortVideoAD \
        data_path=/home/zhouman/guoyunhe/workspace/full/GAMER-rank/data \
        batch_size=1024 \
        max_his_len=100 \
        backbone="${backbone}" \
        gpu="${device}" \
        tasks="${task}" \
        test_task="${task}" \
        seed=42 \
        optim=adamw \
        epochs="${epochs}" \
        learning_rate=1e-3 \
        logging_step=30 \
        weight_decay=0.01 \
        patience=2 \
        save_epoch_limit="${save_epoch_limit}" \
        ckpt_num="${ckpt_num}" \
        suffix="${suffix}" \
        metrics=auc,prauc,logloss,accuracy,precision,recall,f1,gauc_macro,gauc_pair,gauc \
        extra_args= \
        extra_flags= \
        bash "${script_dir}/train_SMB_binary_baseline.sh"
}

wait_for_baselines() {
    local status=0
    local pid
    for pid in "$@"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done
    return "${status}"
}

run_all_baselines() {
    local pids=()

    # Build the shared DIN-style CTR caches before concurrent readers start.
    run_baseline MeanPooling "${gpu_ids[0]}"

    run_baseline DIN "${gpu_ids[0]}" & pids+=("$!")
    run_baseline DIENCVR "${gpu_ids[1]}" & pids+=("$!")
    run_baseline BSTCVR "${gpu_ids[2]}" & pids+=("$!")
    run_baseline HSTUCVR "${gpu_ids[3]}" & pids+=("$!")
    wait_for_baselines "${pids[@]}"

    pids=()
    run_baseline SASRecCVR "${gpu_ids[0]}" & pids+=("$!")
    run_baseline DSIN "${gpu_ids[1]}" & pids+=("$!")
    wait_for_baselines "${pids[@]}"
}

case "${experiments}" in
    all|gamer)
        run_gamer frozen_probe 2415
        run_gamer cold_start 2416
        run_gamer full_finetune 2417
        ;;
    baselines) ;;
    *)
        echo "experiments must be one of: all, gamer, baselines."
        exit 1
        ;;
esac

if [[ "${experiments}" == "all" || "${experiments}" == "baselines" ]]; then
    run_all_baselines
fi
