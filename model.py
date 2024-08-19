from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import asyncio
import time
import onnx
import tensorrt as trt
import numpy as np
from onnx import optimizer
from onnxruntime import InferenceSession

start_loading_time = time.time()

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

# Load and convert 6 models on GPU using TensorRT
models_trt = []
for _ in range(6):
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device_gpu).half()
    dummy_input = torch.randint(0, 50257, (1, 10)).to(device_gpu)
    onnx_path = f"gpt2_trt_{_}.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=11, 
                      do_constant_folding=True, input_names=['input'], output_names=['output'])
    
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, 'rb') as model_file:
        parser.parse(model_file.read())

    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 30  # 1 GB workspace
    engine = builder.build_cuda_engine(network)
    models_trt.append(engine)

# Load 2 models on CPU without TensorRT
models_cpu = [AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device_cpu) for _ in range(2)]

end_loading_time = time.time()
loading_time = end_loading_time - start_loading_time
print(f"Time to load models and tokenizer: {loading_time:.2f} seconds")

inputs = [
    "I am better asd", "Stronger than the moon", "The sun 1234", "I feel like meh", 
    "I am better chill", "Stronger than spicy", "More than ever", "I'm tired"
]

async def generate_text(prompt: str, model_index: int):
    if model_index < len(models_trt):
        # Use TensorRT model on GPU
        engine = models_trt[model_index]
        context = engine.create_execution_context()

        input_ids = tokenizer(prompt, return_tensors="pt").to(device_gpu).input_ids
        output_shape = (1, engine.get_binding_shape(1)[1])
        
        d_input = torch.cuda.FloatTensor(input_ids.cpu().numpy()).half().cuda()
        d_output = torch.cuda.FloatTensor(output_shape).half().cuda()
        
        context.execute_v2(bindings=[int(d_input.data_ptr()), int(d_output.data_ptr())])
        output = d_output.cpu().numpy()
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        
    else:
        # Use standard PyTorch model on CPU
        model = models_cpu[model_index - len(models_trt)]
        inputs = tokenizer(prompt, return_tensors="pt").to(device_cpu)
        outputs = model.generate(**inputs, max_length=50)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded_output

async def process_inputs():
    tasks = [generate_text(prompt, index) for index, prompt in enumerate(inputs)]
    results = await asyncio.gather(*tasks)
    return results

start_prediction_time = time.time()
results = asyncio.run(process_inputs())
for i, result in enumerate(results):
    print(f"Result from model {i+1}: {result}")

end_prediction_time = time.time()
prediction_time = end_prediction_time - start_prediction_time
print(f"Time to generate predictions: {prediction_time:.2f} seconds")
