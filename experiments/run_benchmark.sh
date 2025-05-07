#!/bin/bash

# Define available options
ALL_MODELS=(
  "meta-llama/Llama-3.2-1B"
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "Qwen/QwQ-32B"
  "meta-llama/Llama-3.1-70B-Instruct"
  "bigscience/bloom"
)
ALL_MODEL_SIZES=(1b 8b 32b 70b 176b)
ALL_BATCH_SIZES=(8 16 32 64 128 256 512 1024 2048 4096)
ALL_NUM_GPUS=(1 2 4 8)

# Associate model name to its size
declare -A MODEL_SIZE_MAP=(
  ["meta-llama/Llama-3.2-1B"]=1b
  ["meta-llama/Meta-Llama-3.1-8B-Instruct"]=8b
  ["Qwen/QwQ-32B"]=32b
  ["meta-llama/Llama-3.1-70B-Instruct"]=70b
  ["bigscience/bloom"]=176b
)

declare -A MODEL_SHORTCUT_MAP=(
  [1b]="meta-llama/Llama-3.2-1B"
  [8b]="meta-llama/Meta-Llama-3.1-8B-Instruct"
  [32b]="Qwen/QwQ-32B"
  [70b]="meta-llama/Llama-3.1-70B-Instruct"
  [176b]="bigscience/bloom"
)

# Default selections
SELECTED_MODELS=("meta-llama/Llama-3.1-70B-Instruct")
SELECTED_BATCH_SIZES=(8 16 32 64 128 256 512 1024 2048 4096)
SELECTED_NUM_GPUS=(1 2 4 8)

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --models)
        shift
        SELECTED_MODELS=()
        while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
          model_key="$1"
          if [[ -n "${MODEL_SHORTCUT_MAP[$model_key]}" ]]; then
            SELECTED_MODELS+=("${MODEL_SHORTCUT_MAP[$model_key]}")
          else
            SELECTED_MODELS+=("$model_key")
          fi
          shift
        done
        ;;
      --batch_sizes)
        shift
        SELECTED_BATCH_SIZES=()
        while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
          SELECTED_BATCH_SIZES+=("$1")
          shift
        done
        ;;
      --gpus)
        shift
        SELECTED_NUM_GPUS=()
        while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
          SELECTED_NUM_GPUS+=("$1")
          shift
        done
        ;;
      *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
  done
}

kill_gpu_processes() {
  echo "Killing GPU processes..."
  pids=$(pgrep -f "vllm/bin/python")
  if [ -n "$pids" ]; then
    kill -9 $pids
    echo "Killed processes: $pids"
  else
    echo "No processes found."
  fi
  sleep 1
}

launch_vllm() {
  model=$1
  batch_size=$2
  max_tokens=$3
  model_size=$4
  tp=${5:-1}
  pp=${6:-1}

  echo "Launching: $model batch_size: $batch_size max_tokens: $max_tokens tp: $tp pp: $pp"
  result_filename="experiments/results/result_${model_size}_tp${tp}_pp${pp}_r${batch_size}_t${max_tokens}.json"
  rm -f "$result_filename"

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  VLLM_USE_V1=1 \
  python benchmark.py \
    --model "$model" \
    -tp "$tp" \
    -pp "$pp" \
    --max_tokens "$max_tokens" \
    --num_prompts "$batch_size" \
    --max_model_len 1000 \
    --result_filename "$result_filename" \
    --disable_log_requests

  sleep 2
  kill_gpu_processes
}

main() {
  parse_args "$@"
  max_tokens=1000

  echo "==================== Benchmark Configuration ===================="
  echo "Selected Models:        ${SELECTED_MODELS[*]}"
  echo "Selected Batch Sizes:   ${SELECTED_BATCH_SIZES[*]}"
  echo "Selected # of GPUs:     ${SELECTED_NUM_GPUS[*]}"
  echo "==============================================================="

  for model in "${SELECTED_MODELS[@]}"; do
    model_size=${MODEL_SIZE_MAP[$model]}

    for batch_size in "${SELECTED_BATCH_SIZES[@]}"; do
      for num_gpus in "${SELECTED_NUM_GPUS[@]}"; do

        if (( num_gpus < 8 )) && [[ "$model_size" == "176b" ]]; then
          echo "Skipping 176B with #GPUs < 8"
          continue
        fi

        if (( num_gpus == 1 )) && [[ "$model_size" == "70b" ]]; then
          echo "Skipping 70B on 1 GPU"
          continue
        fi

        if [[ "$model_size" == "70b" ]]; then
          if (( num_gpus == 2 )); then
            echo "Skipping 70B on 2 GPUs"
            continue
          elif (( num_gpus == 4 && batch_size > 512 )); then
            echo "Skipping 70B on 4 GPUs with batch_size > 512"
            continue
          fi
        fi

        launch_vllm "$model" "$batch_size" "$max_tokens" "$model_size" 1 "$num_gpus"
        launch_vllm "$model" "$batch_size" "$max_tokens" "$model_size" "$num_gpus" 1

      done
    done
  done
}

main "$@"