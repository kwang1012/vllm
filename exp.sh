model_name=meta-llama/Llama-2-7b-hf
backend=ray
batchSizes=(16 32 64 128 256)
maxTokens=(100 200 300 400 500 1000)

for i in $(seq 2 2); 
do
    for j in $(seq 2 2);
    do
        if [ $i -eq 1 ]
        then
            tp=$j
            pp=1
        else
            tp=1
            pp=$j
        fi
        if [ $tp -eq 3 ] || [ $pp -eq 3 ] # not divisible by 3
        then
            continue
        fi
        for batchSize in "${batchSizes[@]}"
        do
            for maxToken in "${maxTokens[@]}"
            do
                rm results/profiling/result-$tp-$pp-$batchSize-$maxToken-wonvlink.log
                echo Running experiment, model: $model_name, batch: $batchSize, tp: $tp, pp: $pp, maxTokens: $maxToken
                CUDA_VISIBLE_DEVICES=1,2 HF_TOKEN=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 VLLM_LOGGING_FILENAME=results/profiling/result-$tp-$pp-$batchSize-$maxToken-wonvlink.log python experiment.py \
                --model $model_name \
                --distributed_executor_backend $backend \
                --max_num_seqs $batchSize \
                --tensor_parallel_size $tp \
                --pipeline_parallel_size $pp \
                --max_tokens $maxToken
            done
        done
    done
done