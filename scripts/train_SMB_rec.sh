#!/bin/bash
: ${dataset:=Retail}
: ${batch_size:=4096}
: ${learning_rate:=1e-3}
: ${tasks:=smb_dis}
: ${test_task:=smb_dis}
: ${gpu:=0}
: ${epochs:=200}
: ${backbone:=GRU4Rec}

export CUDA_VISIBLE_DEVICES=$gpu

base_model=./ckpt/seq-models/${backbone}
task_dir=${tasks//,/-}

task_dir=${dataset}/${task_dir}/${backbone}

: ${suffix:=}
if [ "${suffix}" != "" ]; then
    task_dir=${task_dir}_${suffix}
fi

output_dir=./checkpoint/smb_dis/${task_dir}/
result_dir=./results/${task_dir}/
run_name=${task_dir}

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

python main.py train_SMB_rec \
    --backbone ${backbone} \
    --base_model ${base_model} \
    --output_dir ${output_dir} \
    --result_dir ${result_dir} \
    --wandb_run_name ${run_name} \
    --dataset ${dataset} \
    --batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --tasks ${tasks} \
    --test_task ${test_task} \
    --epochs ${epochs} \
    ${extra_args_out} \
    ${extra_flags_out}
