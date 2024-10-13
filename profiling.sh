model_name=meta-llama/Llama-3.2-1B
batchSizes=(8 16 32 64 128 256 512)

for i in $(seq 1 2); 
do
    for j in $(seq 1 2);
    do
        if [ $i -eq 2 ] && [ $j -eq 1 ]
        then
            continue
        fi
        if [ $i -eq 1 ]
        then
            tp=$j
            pp=1
        else
            tp=1
            pp=$j
        fi
        for batchSize in "${batchSizes[@]}"
        do
            rm results/1b/result-$tp-$pp-$batchSize.log
            echo Running experiment, model: $model_name, batch: $batchSize, tp: $tp, pp: $pp
            CUDA_VISIBLE_DEVICES=2,3 VLLM_LOGGING_FILENAME=results/1b/result-$tp-$pp-$batchSize.log python experiment.py \
            --model $model_name \
            --max_num_seqs $batchSize \
            -tp $tp \
            -pp $pp \
            --max_tokens 200 \
            --num_prompts 1000
        done
    done
done