import asyncio

from tqdm import tqdm
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.outputs import RequestOutput
from vllm.utils import random_uuid

async def main():
    # Sample prompts.
    prompts = [
        "Say any 20 words " * 100,
    ]

    # Create an LLM.
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model="meta-llama/Llama-2-7b-chat-hf", max_num_seqs=64, tensor_parallel_size=1, pipeline_parallel_size=1))

    pbar = tqdm(
        total=len(prompts),
        desc="Processed prompts",
        dynamic_ncols=True,
        postfix=(f"est. speed input: {0:.2f} toks/s, "
                    f"output: {0:.2f} toks/s"),
    )
    stats = {
        "total_in_toks": 0,
        "total_out_toks": 0
    }
    stats_per_request_finished = []
    async def run(prompt: str, stats: dict):
        sampling_params = SamplingParams(max_tokens=500)

        request_id = random_uuid()
        async for output in engine.generate(prompt,
                                            sampling_params,
                                            request_id=request_id):
            if output.finished:
                if isinstance(output, RequestOutput):
                    # Calculate tokens only for RequestOutput
                    stats["total_in_toks"] += len(output.prompt_token_ids)
                    in_spd = stats["total_in_toks"] / pbar.format_dict["elapsed"]
                    stats["total_out_toks"] += sum(
                        len(stp.token_ids) for stp in output.outputs)
                    out_spd = stats["total_out_toks"] / pbar.format_dict[
                        "elapsed"]
                    pbar.postfix = (
                        f"est. speed input: {in_spd:.2f} toks/s, "
                        f"output: {out_spd:.2f} toks/s")
                    stats_per_request_finished.append((in_spd, out_spd))
                pbar.update(1)
                final_output = output
        return final_output

    async def generate():
        return await asyncio.gather(
            *[run(prompt, stats) for prompt in prompts]
        )
    outputs = await generate()
    pbar.close()

    for virtual_engine, scheduler in enumerate(engine.engine.scheduler):
        print(f"{virtual_engine=} {scheduler.num_cumulative_preemption=}")

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    # print(json.dumps(stats_per_request_finished))
    # print(BLOCK_SIZE_LIST)

if __name__ == "__main__":
    asyncio.run(main())