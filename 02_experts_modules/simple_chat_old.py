#!/usr/bin/env python3
"""
Simple Chat System - Versão funcional do eziofilho
"""

import os
import sys
from pathlib import Path

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("⚠️ llama-cpp-python not available - install it for GGUF support")

def main():
    print("🤖 EzioFilho Unified - Simple Chat")
    print("=" * 50)
    
    # Check for models
    models_found = []
    
    # Check GGUF models
    gguf_path = Path.home() / ".cache" / "models"
    if gguf_path.exists():
        for model_file in gguf_path.glob("*.gguf"):
            models_found.append({
                "name": model_file.stem,
                "path": str(model_file),
                "type": "gguf",
                "size": model_file.stat().st_size / (1024**3)  # GB
            })
    
    # Check transformers models
    models_path = Path("models")
    if models_path.exists():
        for model_dir in models_path.iterdir():
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                models_found.append({
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "type": "transformers"
                })
    
    if not models_found:
        print("❌ No models found!")
        return
    
    print(f"\n✅ Found {len(models_found)} models:")
    for i, model in enumerate(models_found):
        size_info = f" ({model['size']:.2f}GB)" if 'size' in model else ""
        print(f"  {i+1}. {model['name']} [{model['type']}]{size_info}")
    
    # Select model
    if len(models_found) == 1:
        selected = models_found[0]
        print(f"\n🔧 Using: {selected['name']}")
    else:
        try:
            choice = int(input("\nSelect model (number): ")) - 1
            selected = models_found[choice]
        except:
            selected = models_found[0]
            print(f"\n🔧 Using default: {selected['name']}")
    
    # Load model based on type
    if selected['type'] == 'gguf' and LLAMA_AVAILABLE:
        print(f"\n📥 Loading GGUF model: {selected['name']}...")
        try:
            model = Llama(
                model_path=selected['path'],
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
            print("✅ Model loaded successfully!")
            
            # Chat loop
            print("\n💬 Chat started! (type 'exit' to quit)")
            print("-" * 50)
            
            while True:
                user_input = input("\n👤 You: ")
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    break
                
                print("\n🤖 EzioFilho: ", end='', flush=True)
                
                response = model(
                    user_input,
                    max_tokens=512,
                    temperature=0.7,
                    stream=True
                )
                
                for chunk in response:
                    if 'choices' in chunk:
                        text = chunk['choices'][0].get('text', '')
                        print(text, end='', flush=True)
                
                print()  # New line after response
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            
    elif selected['type'] == 'transformers':
        print(f"\n📥 Loading Transformers model: {selected['name']}...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(selected['path'])
            model = AutoModelForCausalLM.from_pretrained(
                selected['path'],
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            )
            
            print("✅ Model loaded successfully!")
            
            # Chat loop
            print("\n💬 Chat started! (type 'exit' to quit)")
            print("-" * 50)
            
            while True:
                user_input = input("\n👤 You: ")
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    break
                
                print("\n🤖 EzioFilho: ", end='', flush=True)
                
                inputs = tokenizer(user_input, return_tensors="pt")
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the input from response
                response = response[len(user_input):].strip()
                print(response)
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Try installing: pip install accelerate")
    
    else:
        print(f"❌ Cannot load {selected['type']} model - missing dependencies")
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()