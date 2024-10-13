from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    # "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
    # "What " * 50
    "Santa Clara is a"
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=100)

# Create an LLM.
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
