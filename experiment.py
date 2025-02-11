from argparse import ArgumentParser
import asyncio

from tqdm import tqdm

from vllm import SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import FlexibleArgumentParser, random_uuid
from torch.profiler import profile, ProfilerActivity

async def main(args):
    # Sample prompts.
    prompts = [
        "How is the weather in Champaign?",
    ] * args.num_prompts

    # Create an LLM.
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)


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
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     await engine.start_profile()
    outputs = await generate()
    pbar.close()
    #     await engine.stop_profile()
    # prof.export_chrome_trace("trace.json")
    # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    avg_generated_text_len = []
    for output in outputs:
        generated_text = output.outputs[0].text
        print(generated_text)
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