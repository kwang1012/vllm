import asyncio
from dataclasses import dataclass, field
import json
import os
import random
import time
from typing import List, Optional, Tuple

from tqdm import tqdm
import numpy as np

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

MILLISECONDS_TO_SECONDS_CONVERSION = 1000

@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]

@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    generated_token_ids: list[int] = field(
        default_factory=list)
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(
        default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0

def calculate_metrics(
    outputs: list[RequestFuncOutput],
    dur_s: float,
    selected_percentiles: list[float] = [99],
) -> tuple[BenchmarkMetrics, list[int]]:
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    for i in range(len(outputs)):
        output_len = len(outputs[i].generated_token_ids)
        actual_output_lens.append(output_len)
        total_input += outputs[i].prompt_len
        tpot = 0
        if output_len > 1:
            latency_minus_ttft = outputs[i].latency - outputs[i].ttft
            tpot = latency_minus_ttft / (output_len - 1)
            tpots.append(tpot)
        # Note: if output_len <= 1, we regard tpot as 0 for goodput
        all_tpots.append(tpot)
        itls += outputs[i].itl
        ttfts.append(outputs[i].ttft)
        e2els.append(outputs[i].latency)
        completed += 1

    valid_metrics = []
    slo_values = []

    # if "ttft" in goodput_config_dict:
    #     valid_metrics.append(ttfts)
    #     slo_values.append(goodput_config_dict["ttft"] /
    #                         MILLISECONDS_TO_SECONDS_CONVERSION)
    # if "tpot" in goodput_config_dict:
    #     valid_metrics.append(all_tpots)
    #     slo_values.append(goodput_config_dict["tpot"] /
    #                         MILLISECONDS_TO_SECONDS_CONVERSION)
    # if "e2el" in goodput_config_dict:
    #     valid_metrics.append(e2els)
    #     slo_values.append(goodput_config_dict["e2el"] /
    #                         MILLISECONDS_TO_SECONDS_CONVERSION)

    # for req_metric in zip(*valid_metrics):
    #     is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
    #     if is_good_req:
    #         good_completed += 1

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000)
                            for p in selected_percentiles],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000)
                             for p in selected_percentiles],
    )

    return metrics, actual_output_lens

async def main(args):

    # Create an LLM.
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Sample prompts.
    if not os.path.exists("ShareGPT_V3.json"):
        import urllib.request
        urllib.request.urlretrieve(
            "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json", "ShareGPT_V3.json")

    tokenizer = AutoTokenizer.from_pretrained(engine_args.model)
    requests = sample_sharegpt_requests(
        "ShareGPT_V3.json", args.num_prompts, tokenizer)

    prompts = [request[0] for request in requests]

    prompts = [
        "How is the weather in Champaign?",
    ] * args.num_prompts

    pbar = tqdm(
        total=len(prompts),
        desc="Processed prompts",
        dynamic_ncols=True,
    )

    async def run(prompt: str, sampling_params: SamplingParams) -> RequestFuncOutput:

        output = RequestFuncOutput()

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st

        request_id = random_uuid()
        generator = engine.generate(
            prompt, sampling_params, request_id=request_id)
        request_output = None
        async for request_output in generator:
            timestamp = time.perf_counter()
            # First token
            if ttft == 0.0:
                ttft = time.perf_counter() - st
                output.ttft = ttft

            # Decoding phase
            else:
                output.itl.append(timestamp -
                                    most_recent_timestamp)

            most_recent_timestamp = timestamp
            
        pbar.update(1)
        output.latency = most_recent_timestamp - st
        output.prompt_len = len(request_output.prompt_token_ids)
        output.generated_text = request_output.outputs[0].text
        output.generated_token_ids = request_output.outputs[0].token_ids
        return output

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens, temperature=0, ignore_eos=True)

    benchmark_start_time = time.perf_counter()
    outputs = await asyncio.gather(
            *[run(prompt, sampling_params) for prompt in prompts]
    )
    pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        # selected_percentile_metrics=selected_percentile_metrics,
        # selected_percentiles=selected_percentiles,
        # goodput_config_dict=goodput_config_dict,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):",
                                    metrics.total_token_throughput))

    result = {
        "duration": benchmark_duration,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in ["ttft","tpot","itl"]:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms")))
        print("{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms")))
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics,
                                f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                            value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT",
                       "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    with open(args.result_filename, "w", encoding='utf-8') as outfile:
        json.dump(result, outfile)
    
    destroy_model_parallel()


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--num-prompts", default=1000, type=int)
    parser.add_argument("--prompt-len", default=10, type=int)
    parser.add_argument("--max-tokens", default=100, type=int)
    parser.add_argument("--result-filename", default="benchmark_result.json", type=str)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    asyncio.run(main(args))
