from argparse import ArgumentParser
import asyncio
import json
import os
import random
from typing import List, Optional, Tuple

from tqdm import tqdm

from vllm import SamplingParams, envs
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import FlexibleArgumentParser, random_uuid
from transformers import PreTrainedTokenizerBase, AutoTokenizer

def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int, None]]:
    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or (fixed_output_len is None and output_len < 4):
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len, None))

    return filtered_dataset


async def main(args):
    
    # Create an LLM.
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    
    # Sample prompts.
    if not os.path.exists("ShareGPT_V3.json"):
        import urllib.request
        urllib.request.urlretrieve("https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json", "ShareGPT_V3.json")

    tokenizer = AutoTokenizer.from_pretrained(engine_args.model)
    requests = sample_sharegpt_requests("ShareGPT_V3.json", args.num_prompts, tokenizer)

    prompts = [request[0] for request in requests]

    prompts = [
        "How is the weather in Champaign?",
    ] * args.num_prompts
    
    pbar = tqdm(
        total=len(prompts),
        desc="Processed prompts",
        dynamic_ncols=True,
    )

    async def run(prompt: str):
        sampling_params = SamplingParams(max_tokens=args.max_tokens)

        request_id = random_uuid()
        async for output in engine.generate(prompt,
                                            sampling_params,
                                            request_id=request_id):
            if output.finished:
                final_output = output
                pbar.update(1)
        return final_output

    async def generate():
        return await asyncio.gather(
            *[run(prompt) for prompt in prompts]
        )
    
    if envs.VLLM_TORCH_PROFILER_DIR:
        await engine.start_profile()
    outputs = await generate()
    if envs.VLLM_TORCH_PROFILER_DIR:
        await engine.stop_profile()
    pbar.close()

    avg_generated_text_len = []
    for output in outputs:
        generated_text = output.outputs[0].text
        # print(generated_text)
        avg_generated_text_len.append(len(generated_text))

    destroy_model_parallel()
    print("Average generated text length:", sum(avg_generated_text_len) / len(avg_generated_text_len))

if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--num-prompts", default=1000, type=int)
    parser.add_argument("--prompt-len", default=10, type=int)
    parser.add_argument("--max-tokens", default=100, type=int)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    asyncio.run(main(args))