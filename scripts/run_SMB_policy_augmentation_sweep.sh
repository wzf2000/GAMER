#!/bin/bash
set -u
set -o pipefail

dataset=${dataset:-ShortVideoAD}
original=${original:-1}
batch_size=${batch_size:-512}
test_batch_size=${test_batch_size:-128}
tasks=${tasks:-smb_policy_decoder}
test_task=${test_task:-smb_explicit}
gpu=${gpu:-0,1,2,3,4,5,6,7}
backbone=Qwen3TemporalHierarchicalMultiViewSoft

port_base=${port_base:-2330}
log_root=${log_root:-logs/augment_sweep}
run_stamp=${run_stamp:-$(date +%Y%m%d-%H%M%S)}
skip_done=${skip_done:-1}
skip_test_done=${skip_test_done:-0}
auto_resume=${auto_resume:-1}
continue_on_error=${continue_on_error:-0}
dry_run=${dry_run:-0}

log_dir="${log_root}/${run_stamp}"
failure_dir="${log_dir}/failures"
mkdir -p "${log_dir}" "${failure_dir}"
summary_file="${log_dir}/summary.tsv"
printf "time\tsuffix\tphase\tstatus\tgpu\tport\tcheckpoint_root\tdetail\n" > "${summary_file}"

common_train_args=(
  --max_his_len 100
  --num_train_epochs 200
  --learning_rate 5e-4
  --gradient_accumulation_steps 8
  --warmup_ratio 0.04
  --patience 20
  --temperature 0.7
  --eval_strategy steps
  --eval_steps 200
  --save_strategy steps
  --save_steps 200
)

checkpoint_root_for() {
  local suffix="$1"
  echo "./checkpoint/SMB-decoder/${dataset}/${tasks}/${backbone}_${suffix}/original"
}

result_file_for() {
  local suffix="$1"
  echo "./results/${dataset}/${tasks}/${backbone}_${suffix}/results-${test_task}-original.json"
}

latest_checkpoint_for() {
  local root="$1"
  if [ ! -d "${root}" ]; then
    return 0
  fi
  find "${root}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
}

checkpoint_num_from_path() {
  basename "$1" | sed 's/^checkpoint-//'
}

select_test_ckpt_num() {
  local root="$1"
  local latest_ckpt
  if [ -f "${root}/model.safetensors" ]; then
    echo "best"
    return 0
  fi
  latest_ckpt=$(latest_checkpoint_for "${root}")
  if [ -n "${latest_ckpt}" ]; then
    checkpoint_num_from_path "${latest_ckpt}"
    return 0
  fi
  echo "best"
}

log_summary() {
  local suffix="$1"
  local phase="$2"
  local status="$3"
  local port="$4"
  local root="$5"
  local detail="${6:-}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%F %T')" "${suffix}" "${phase}" "${status}" "${gpu}" "${port}" "${root}" "${detail}" \
    >> "${summary_file}"
}

run_train() {
  local index="$1"
  local suffix="$2"
  local train_port="$3"
  local log_file="$4"
  shift 4
  local root
  local latest_ckpt
  local -a resume_args=()

  root=$(checkpoint_root_for "${suffix}")

  if [ "${skip_done}" = "1" ] && [ -f "${root}/model.safetensors" ]; then
    echo "[$(date '+%F %T')] Skip train ${suffix}: final model exists." | tee -a "${log_file}"
    log_summary "${suffix}" "train" "skipped" "${train_port}" "${root}" "final model exists"
    return 0
  fi

  latest_ckpt=$(latest_checkpoint_for "${root}")
  if [ "${auto_resume}" = "1" ] && [ -n "${latest_ckpt}" ] && [ ! -f "${root}/model.safetensors" ]; then
    resume_args=(--resume_from_checkpoint "${latest_ckpt}")
    echo "[$(date '+%F %T')] Resume train from ${latest_ckpt}" | tee -a "${log_file}"
  fi

  if [ "${dry_run}" = "1" ]; then
    echo "[$(date '+%F %T')] Dry run train command:" | tee -a "${log_file}"
    printf "env dataset=%q original=%q batch_size=%q tasks=%q gpu=%q port=%q backbone=%q bash scripts/train_SMB_decoder.sh" \
      "${dataset}" "${original}" "${batch_size}" "${tasks}" "${gpu}" "${train_port}" "${backbone}" | tee -a "${log_file}"
    printf " %q" --suffix "${suffix}" "${common_train_args[@]}" "${resume_args[@]}" "$@" | tee -a "${log_file}"
    printf "\n" | tee -a "${log_file}"
    log_summary "${suffix}" "train" "dry_run" "${train_port}" "${root}" ""
    return 0
  fi

  env \
    dataset="${dataset}" \
    original="${original}" \
    batch_size="${batch_size}" \
    tasks="${tasks}" \
    gpu="${gpu}" \
    port="${train_port}" \
    backbone="${backbone}" \
    bash scripts/train_SMB_decoder.sh \
      --suffix "${suffix}" \
      "${common_train_args[@]}" \
      "${resume_args[@]}" \
      "$@" 2>&1 | tee -a "${log_file}"

  local status=${PIPESTATUS[0]}
  if [ "${status}" -ne 0 ]; then
    echo "[$(date '+%F %T')] FAILED train ${suffix} with status ${status}" | tee -a "${log_file}"
    log_summary "${suffix}" "train" "failed:${status}" "${train_port}" "${root}" ""
    touch "${failure_dir}/${suffix}.train"
    return "${status}"
  fi

  echo "[$(date '+%F %T')] Finished train ${suffix}" | tee -a "${log_file}"
  log_summary "${suffix}" "train" "finished" "${train_port}" "${root}" ""
  return 0
}

run_test() {
  local suffix="$1"
  local test_port="$2"
  local log_file="$3"
  local root
  local result_file
  local test_ckpt_num

  root=$(checkpoint_root_for "${suffix}")
  result_file=$(result_file_for "${suffix}")
  test_ckpt_num=$(select_test_ckpt_num "${root}")

  if [ "${skip_test_done}" = "1" ] && [ -f "${result_file}" ]; then
    echo "[$(date '+%F %T')] Skip test ${suffix}: result file exists." | tee -a "${log_file}"
    log_summary "${suffix}" "test" "skipped" "${test_port}" "${root}" "${result_file}"
    return 0
  fi

  if [ "${dry_run}" = "1" ]; then
    echo "[$(date '+%F %T')] Dry run test command:" | tee -a "${log_file}"
    printf "env ckpt_num=%q dataset=%q original=%q batch_size=%q tasks=%q test_task=%q gpu=%q port=%q backbone=%q bash scripts/test_SMB_decoder.sh" \
      "${test_ckpt_num}" "${dataset}" "${original}" "${test_batch_size}" "${tasks}" "${test_task}" "${gpu}" "${test_port}" "${backbone}" | tee -a "${log_file}"
    printf " %q" --suffix "${suffix}" --max_his_len 100 | tee -a "${log_file}"
    printf "\n" | tee -a "${log_file}"
    log_summary "${suffix}" "test" "dry_run" "${test_port}" "${root}" "ckpt_num=${test_ckpt_num}"
    return 0
  fi

  env \
    ckpt_num="${test_ckpt_num}" \
    dataset="${dataset}" \
    original="${original}" \
    batch_size="${test_batch_size}" \
    tasks="${tasks}" \
    test_task="${test_task}" \
    gpu="${gpu}" \
    port="${test_port}" \
    backbone="${backbone}" \
    bash scripts/test_SMB_decoder.sh \
      --suffix "${suffix}" \
      --max_his_len 100 2>&1 | tee -a "${log_file}"

  local status=${PIPESTATUS[0]}
  if [ "${status}" -ne 0 ]; then
    echo "[$(date '+%F %T')] FAILED test ${suffix} with status ${status}" | tee -a "${log_file}"
    log_summary "${suffix}" "test" "failed:${status}" "${test_port}" "${root}" "ckpt_num=${test_ckpt_num}"
    touch "${failure_dir}/${suffix}.test"
    return "${status}"
  fi

  echo "[$(date '+%F %T')] Finished test ${suffix}" | tee -a "${log_file}"
  log_summary "${suffix}" "test" "finished" "${test_port}" "${root}" "${result_file}"
  return 0
}

run_experiment() {
  local index="$1"
  local suffix="$2"
  shift 2
  local train_port=$((port_base + index * 2))
  local test_port=$((train_port + 1))
  local root
  local log_file

  root=$(checkpoint_root_for "${suffix}")
  log_file="${log_dir}/${index}_${suffix}.log"

  echo "============================================================" | tee -a "${log_file}"
  echo "[$(date '+%F %T')] Start ${suffix}" | tee -a "${log_file}"
  echo "Backbone: ${backbone}" | tee -a "${log_file}"
  echo "GPUs: ${gpu} | train_port: ${train_port} | test_port: ${test_port}" | tee -a "${log_file}"
  echo "Checkpoint root: ${root}" | tee -a "${log_file}"

  if ! run_train "${index}" "${suffix}" "${train_port}" "${log_file}" "$@"; then
    return 1
  fi

  run_test "${suffix}" "${test_port}" "${log_file}"
}

run_or_stop() {
  local index="$1"
  local suffix="$2"
  shift 2
  if ! run_experiment "${index}" "${suffix}" "$@"; then
    if [ "${continue_on_error}" != "1" ]; then
      echo "Stopping after failure. Set continue_on_error=1 to keep running."
      exit 1
    fi
  fi
}

echo "Logs: ${log_dir}"
echo "Summary: ${summary_file}"
echo "Fixed backbone: ${backbone}"
echo "Serial GPUs: ${gpu}"
echo "Train batch_size=${batch_size}; test_batch_size=${test_batch_size}"

run_or_stop 0 aug_time_decay \
  --sequence_augmentation time_decay \
  --time_decay_type exponential \
  --time_decay_tau 30 \
  --time_decay_severity 0.5 \
  --time_decay_max_drop 0.7 \
  --time_decay_min_recent_items 5

run_or_stop 1 aug_session \
  --sequence_augmentation session \
  --recent_session_count 2 \
  --session_keep_probability 0.5 \
  --session_time_decay_tau 3 \
  --session_high_level_bonus 0.2

run_or_stop 2 aug_dataset_proportion \
  --sequence_augmentation dataset_proportion

run_or_stop 3 aug_user_adaptive \
  --sequence_augmentation user_adaptive_ratio \
  --user_adaptive_smoothing 5 \
  --user_adaptive_confidence_scale 20 \
  --user_adaptive_min_ratio 0.25 \
  --user_adaptive_max_ratio 20 \
  --user_adaptive_tolerance 1.0

run_or_stop 4 aug_target_conditioned \
  --sequence_augmentation target_conditioned \
  --target_conditioned_base_policy time_decay \
  --target_conditioned_same_level_restore 0.8 \
  --target_conditioned_precursor_restore 0.8

run_or_stop 5 aug_multi_view \
  --sequence_augmentation multi_view

if compgen -G "${failure_dir}/*" > /dev/null; then
  echo "Sweep completed with failures. See ${summary_file} and ${failure_dir}."
  exit 1
fi

echo "Sweep completed successfully."
