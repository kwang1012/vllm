
kill_gpu_processes() {
  # kill all processes on GPU.
  pgrep pt_main_thread | xargs -r kill -9
  pgrep python | xargs -r kill -9
  for port in 8000 8100 8200; do lsof -t -i:$port | xargs -r kill -9; done
  sleep 1
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


launch_vllm() {
    local model=$1
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    VLLM_USE_V1=1 \
    VLLM_LOGGING_FILENAME=result-pp4-r64-t1000-sharegpt.log \
    vllm serve $1 \
    -pp 4 \
    --disable_log_requests
    wait_for_server 8100
    sleep 1
}


benchmark() {
    results_filename="benchmark-pp4-r64-t1000-sharegpt.json"
    local model=$1
    dataset_name="sharegpt"
    dataset_path="ShareGPT_V3.json"
    num_prompts=64

    python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model $1 \
    --dataset-name $dataset_name \
    --dataset-path $dataset_path \
    --num-prompts $num_prompts \
    --result-filename $results_filename

    sleep 2
}


main() {
  launch_disagg_prefill
  benchmark
  done
  kill_gpu_processes
}


main "$@"