import os
import time
from llama_cpp import Llama

# Path to your model
MODEL_PATH = r"C:\Users\anapa\EzioFilhoUnified\modelos_hf\TheBloke--Phi-2-GGUF\phi-2.Q2_K.gguf"

print("=== Testing CUDA Support ===")
print(f"Loading model: {MODEL_PATH}")

# First test with CPU only
print("\n[TEST 1] CPU Only Mode")
start = time.time()
llm_cpu = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_gpu_layers=0,  # Force CPU only
    n_threads=4
)
cpu_load_time = time.time() - start
print(f"CPU load time: {cpu_load_time:.2f} seconds")

# Generate a short response
prompt = "Explain what is CUDA in one sentence:"
start = time.time()
output_cpu = llm_cpu(prompt, max_tokens=20)
cpu_gen_time = time.time() - start
print(f"CPU generation time: {cpu_gen_time:.2f} seconds")
print(f"CPU output: {output_cpu['choices'][0]['text']}")

# Clean up to free memory
del llm_cpu

# Now test with GPU
print("\n[TEST 2] GPU Mode")
start = time.time()
llm_gpu = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_gpu_layers=-1,  # Try to use all layers on GPU
    n_threads=4
)
gpu_load_time = time.time() - start
print(f"GPU load time: {gpu_load_time:.2f} seconds")

# Generate the same response
start = time.time()
output_gpu = llm_gpu(prompt, max_tokens=20)
gpu_gen_time = time.time() - start
print(f"GPU generation time: {gpu_gen_time:.2f} seconds")
print(f"GPU output: {output_gpu['choices'][0]['text']}")

# Compare times
print("\n=== CUDA Support Results ===")
print(f"CPU load time: {cpu_load_time:.2f}s | GPU load time: {gpu_load_time:.2f}s")
print(f"CPU generate: {cpu_gen_time:.2f}s | GPU generate: {gpu_gen_time:.2f}s")
print(f"Speedup (generation): {cpu_gen_time / gpu_gen_time:.2f}x")

# If GPU is at least 2x faster, we can assume CUDA is working
if gpu_gen_time < cpu_gen_time / 2:
    print("\n✅ CUDA IS WORKING! GPU generation is significantly faster.")
else:
    print("\n❌ CUDA might NOT be working. GPU generation is not significantly faster.")