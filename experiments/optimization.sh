models=("meta-llama/Llama-3.2-1B" 
  "meta-llama/Meta-Llama-3.1-8B-Instruct")
model_sizes=(1b 8b)
batch_sizes=(8 16 32 64 128 256 512 1024)
num_gpus_list=(1)

# max batch size 4096

# 1b: 4096

# 8b:
#  - 2 gpus: 1024
#  - 4 gpus: 2048
#  - 8 gpus: 4096

# 70b:
#  - 2 gpus: 64
#  - 4 gpus: 512
#  - 8 gpus: 2048

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
  num_reqs=$2
  max_tokens=$3
  model_size=$4
  tp="${5:-1}"
  pp="${6:-1}"
  echo $model num_reqs: ${num_reqs}, max_tokens: ${max_tokens}, tp: ${tp}, pp: ${pp}.
  result_filename="experiments/results/result_${model_size}_tp${tp}_pp${pp}_r${num_reqs}_t${max_tokens}.json"
  rm -f $result_filename
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  VLLM_USE_V1=1 \
  python benchmark.py \
  --model $model \
  -tp $tp \
  -pp $pp \
  --max_tokens $max_tokens \
  --num_prompts $num_reqs \
  --result_filename $result_filename \
  --disable_log_requests
  sleep 2
  kill_gpu_processes
}

main() {

  max_tokens=1000

  ITER=0
  for model in ${models[@]}; do
    model_size=${model_sizes[$ITER]}
    for batch_size in ${batch_sizes[@]}; do
      for num_gpus in ${num_gpus_list[@]}; do
        if (( num_gpus == 1 )); then
          if [[ "$model_size" != "70b" ]]; then
            launch_vllm $model $batch_size $max_tokens $model_size 1 1
          fi
          continue
        fi
        if [[ "$model_size" == "70b" ]]; then
          if (( num_gpus == 2 )); then
            continue
          elif (( num_gpus == 4 && batch_size > 512 )); then
            continue
          fi
        fi
        # run pp, tp
        launch_vllm $model $batch_size $max_tokens $model_size 1 $num_gpus
        launch_vllm $model $batch_size $max_tokens $model_size $num_gpus 1
      done
    done
    ITER=$(expr $ITER + 1)
  done
}

main "$@"