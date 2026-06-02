#!/bin/bash

build_task_dir() {
    local dataset="$1"
    local tasks="$2"
    local backbone="$3"
    local suffix="${4:-}"
    local task_name="${tasks//,/-}"
    local task_dir="${dataset}/${task_name}/${backbone}"
    if [ "${suffix}" != "" ]; then
        task_dir=${task_dir}_${suffix}
    fi
    echo "${task_dir}"
}

build_checkpoint_path() {
    local scope="$1"
    local task_dir="$2"
    local tag="$3"
    echo "./checkpoint/${scope}/${task_dir}/${tag}/"
}

build_result_path() {
    local task_dir="$1"
    local file_name="$2"
    echo "./results/${task_dir}/${file_name}"
}

