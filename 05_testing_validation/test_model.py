from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose a model path from your collection
model_path = r"C:\Users\anapa\SuperIA\EzioFilhoUnified\modelos_hf\microsoft--phi-2"

print(f"Testing model loading from {model_path}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Test inference
    prompt = "What is the capital of France?"
    print(f"\nTesting inference with prompt: '{prompt}'")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_new_tokens=50)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(f"\nModel response: {response}")
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")