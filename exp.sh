model_name=meta-llama/Llama-2-7b-hf
backend=ray
tp=1
pp=1
batchSizes=(16 32 64 128 256)
maxTokens=(100 200 300 400 500 1000)

for batchSize in "${batchSizes[@]}"
do
    for maxToken in "${maxTokens[@]}"
    do
        HF_TOKEN=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 VLLM_LOGGING_FILENAME=result-$tp-$pp-$batchSize-$maxToken python experiment.py \
        --model $model_name \
        --distributed_executor_backend $backend \
        --max_num_seqs $batchSize \
        --tensor_parallel_size $tp \
        --pipeline_parallel_size $pp \
        --max_tokens $maxToken
    done
done