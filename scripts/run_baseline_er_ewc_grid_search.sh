#!/bin/bash
# Grid search for ER and EWC baselines on TRACE-5000
#
# GPUs: 0,1,2,3,4,5,7
#
# Grid:
#   baseline      in {er, ewc}
#   learning_rate in {1e-4, 5e-4, 1e-3}
#   num_epochs    in {3, 5}
#
# Total runs:
#   2 baselines x 3 learning rates x 2 epochs = 12 runs

set -euo pipefail

GPUS=(0 1 2 3 4 5 7)

DATA_STREAM="TRACE-5000"
STREAM_TYPE="Batched"

BATCH_SIZE=16

SEED=0
OUTPUT_ROOT="review_outputs"
LOG_ROOT="grid_search_logs"

BASELINES=(er ewc)
LEARNING_RATES=(1e-4 5e-4 1e-3)
EPOCHS=(3 5)

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

JOBS=()

for baseline in "${BASELINES[@]}"; do
  for lr in "${LEARNING_RATES[@]}"; do
    for epochs in "${EPOCHS[@]}"; do
      JOBS+=("${baseline} ${lr} ${epochs}")
    done
  done
done

TOTAL_RUNS=${#JOBS[@]}
NUM_GPUS=${#GPUS[@]}

echo "=========================================="
echo "Baseline grid search"
echo "Baselines: ${BASELINES[*]}"
echo "Learning rates: ${LEARNING_RATES[*]}"
echo "Epochs: ${EPOCHS[*]}"
echo "GPUs: ${GPUS[*]}"
echo "Total runs: ${TOTAL_RUNS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Log root: ${LOG_ROOT}"
echo "=========================================="

run_job() {
  local gpu_id="$1"
  local job_index="$2"
  local baseline="$3"
  local lr="$4"
  local epochs="$5"

  local run_number=$((job_index + 1))
  local run_name="${baseline}_lr${lr}_ep${epochs}_bs${BATCH_SIZE}"
  local log_dir="${LOG_ROOT}/gpu${gpu_id}"
  local log_file="${log_dir}/${run_name}.log"
  local run_output_dir="${OUTPUT_ROOT}/${run_name}"

  mkdir -p "$log_dir" "$run_output_dir"

  echo "[$run_number/$TOTAL_RUNS] START: $run_name" | tee "$log_file"
  echo "GPU: ${gpu_id}" | tee -a "$log_file"
  echo "Baseline: ${baseline}" | tee -a "$log_file"
  echo "Learning rate: ${lr}" | tee -a "$log_file"
  echo "Epochs: ${epochs}" | tee -a "$log_file"
  echo "Batch size: ${BATCH_SIZE}" | tee -a "$log_file"
  echo "Output dir: ${run_output_dir}" | tee -a "$log_file"
  echo "Log file: ${log_file}" | tee -a "$log_file"
  echo "------------------------------------------" | tee -a "$log_file"

  set +e

  CUDA_VISIBLE_DEVICES="$gpu_id" python main.py \
    --baseline "$baseline" \
    --data_stream "$DATA_STREAM" \
    --stream_type "$STREAM_TYPE" \
    --learning_rate "$lr" \
    --num_epochs "$epochs" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --output_dir "$run_output_dir" \
    --use_gt_metrics \
    2>&1 | tee -a "$log_file"

  status=${PIPESTATUS[0]}

  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "[$run_number/$TOTAL_RUNS] FAILED: $run_name with exit code $status" | tee -a "$log_file"
    return "$status"
  fi

  echo "[$run_number/$TOTAL_RUNS] DONE: $run_name" | tee -a "$log_file"
  echo "==========================================" | tee -a "$log_file"
}

worker() {
  local worker_id="$1"
  local gpu_id="$2"

  local local_count=0

  for job_index in "${!JOBS[@]}"; do
    if (( job_index % NUM_GPUS == worker_id )); then
      local_count=$((local_count + 1))

      read -r baseline lr epochs <<< "${JOBS[$job_index]}"

      echo "GPU ${gpu_id}: assigned job $((job_index + 1))/${TOTAL_RUNS}: baseline=${baseline}, lr=${lr}, epochs=${epochs}"

      run_job "$gpu_id" "$job_index" "$baseline" "$lr" "$epochs"
    fi
  done

  echo "GPU ${gpu_id}: finished ${local_count} assigned runs."
}

pids=()

cleanup() {
  echo "Stopping all workers..."
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}

trap cleanup INT TERM

for worker_id in "${!GPUS[@]}"; do
  gpu_id="${GPUS[$worker_id]}"

  worker "$worker_id" "$gpu_id" &
  pids+=("$!")
done

failed=0

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "At least one GPU worker failed."
  exit 1
fi

echo "All ${TOTAL_RUNS} baseline runs finished successfully."