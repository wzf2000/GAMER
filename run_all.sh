#!/bin/bash

set -e

# ==========================
# General settings
# ==========================

dataset=ShortVideoAD
original=1
gpu=0,1,2,3,4,5,6,7
batch_size=512
tasks=smb_policy_decoder
backbone=Qwen3TemporalHierarchicalMultiViewSoft

max_his_len=100
num_train_epochs=200
learning_rate=5e-4
gradient_accumulation_steps=8
warmup_ratio=0.04
patience=20
temperature=0.7
augmentation_views=1


# ==========================
# Function
# ==========================

run_exp() {

    aug=$1
    suffix=$2

    echo "======================================"
    echo "Running augmentation: ${aug}"
    echo "Suffix: ${suffix}"
    echo "======================================"


    # Train
    CUDA_VISIBLE_DEVICES=${gpu} \
    dataset=${dataset} \
    original=${original} \
    gpu=${gpu} \
    batch_size=${batch_size} \
    tasks=${tasks} \
    backbone=${backbone} \
    suffix=${suffix} \
    bash scripts/train_SMB_decoder.sh \
        --sequence_augmentation ${aug} \
        --augmentation_views ${augmentation_views} \
        --max_his_len ${max_his_len} \
        --num_train_epochs ${num_train_epochs} \
        --learning_rate ${learning_rate} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --warmup_ratio ${warmup_ratio} \
        --patience ${patience} \
        --temperature ${temperature}


    # Test
    CUDA_VISIBLE_DEVICES=${gpu} \
    dataset=${dataset} \
    original=${original} \
    gpu=${gpu} \
    batch_size=${batch_size} \
    tasks=${tasks} \
    test_task=smb_explicit \
    backbone=${backbone} \
    suffix=${suffix} \
    bash scripts/test_SMB_decoder.sh \
        --max_his_len ${max_his_len}


    echo "Finished: ${aug}"
    echo

}



# ==========================
# Run all experiments
# ==========================


#run_exp \
#"time_decay" \
#"fullseq_time_decay"


run_exp \
"session" \
"fullseq_session"


run_exp \
"dataset_proportion" \
"fullseq_dataset_proportion"


run_exp \
"user_adaptive_ratio" \
"fullseq_user_adaptive"


run_exp \
"target_conditioned" \
"fullseq_target_conditioned"


run_exp \
"multi_view" \
"fullseq_multi_view"


echo "======================================"
echo "All experiments finished!"
echo "======================================"
