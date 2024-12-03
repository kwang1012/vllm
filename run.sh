CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_LOGGING_FILENAME=result-comm.log \
python experiment.py \
--model meta-llama/Llama-2-7b-hf \
-pp 2 \
--max_num_seqs 146 \
--max_tokens 1000 \
--num_prompts 1000