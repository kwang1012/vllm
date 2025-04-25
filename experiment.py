import asyncio

from tqdm import tqdm

from vllm import SamplingParams, envs
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import FlexibleArgumentParser, random_uuid

async def main(args):
    
    # Create an LLM.
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    prompts = [
        "How is the weather in Champaign?",
    ] * args.num_prompts
    
    pbar = tqdm(
        total=len(prompts),
        desc="Processed prompts",
        dynamic_ncols=True,
    )

    async def run(prompt: str):
        sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0, ignore_eos=True)

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
    pbar.close()

    avg_generated_length = []
    for output in outputs:
        avg_generated_length.append(len(output.outputs[0].token_ids))

    if envs.VLLM_TORCH_PROFILER_DIR:
        await engine.stop_profile()
        
    destroy_model_parallel()
    print("Average generated # tokens:", sum(avg_generated_length) / len(avg_generated_length))

if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--num-prompts", default=1000, type=int)
    parser.add_argument("--prompt-len", default=10, type=int)
    parser.add_argument("--max-tokens", default=100, type=int)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    asyncio.run(main(args))