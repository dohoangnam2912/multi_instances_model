from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import asyncio
import time

# Measure time to load models and tokenizer
start_loading_time = time.time()

# Load the tokenizer once
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# Load multiple model instances and move them to GPU or CPU
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

# Load 6 models on GPU
models_gpu = [AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device_gpu) for _ in range(6)]

# Load 6 models on CPU
models_cpu = [AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device_cpu) for _ in range(6)]

end_loading_time = time.time()
loading_time = end_loading_time - start_loading_time
print(f"Time to load models and tokenizer: {loading_time:.2f} seconds")

# Define the inputs for each model
inputs = [
    "I am better asd",
    "Stronger than the moon",
    "The sun 1234",
    "I feel like meh",
    "I am better chill",
    "Stronger than spicy",
    "More than ever",
    "I'm tired",
    "I'm feeling great",
    "I'm tired",
    "I'm feeling great",
    "I'm blue"
]

# Asynchronous function to generate text with a specific model
async def generate_text(prompt: str, model_index: int):
    if model_index < len(models_gpu):
        model = models_gpu[model_index]
        device = device_gpu
    else:
        model = models_cpu[model_index - len(models_gpu)]
        device = device_cpu

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Run the model generation in an executor
    loop = asyncio.get_event_loop()
    outputs = await loop.run_in_executor(None, lambda: model.generate(**inputs, max_length=50))
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Asynchronously process each input with its corresponding model
async def process_inputs():
    tasks = [generate_text(prompt, index) for index, prompt in enumerate(inputs)]
    results = await asyncio.gather(*tasks)
    return results

# Measure time to generate predictions
start_prediction_time = time.time()

# Run the asynchronous processing and print results
results = asyncio.run(process_inputs())
for i, result in enumerate(results):
    print(f"Result from model {i+1}: {result}")

end_prediction_time = time.time()
prediction_time = end_prediction_time - start_prediction_time
print(f"Time to generate predictions: {prediction_time:.2f} seconds")