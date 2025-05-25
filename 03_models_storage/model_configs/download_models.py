#!/usr/bin/env python3
"""
Download models from Hugging Face for EzioFilho system
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import torch

# Set HF token
os.environ['HF_TOKEN'] = 'os.getenv("HUGGINGFACE_TOKEN", "your_token_here")'

# Create directories
models_dir = Path("./models")
cache_dir = Path.home() / ".cache" / "models"
models_dir.mkdir(exist_ok=True)
cache_dir.mkdir(exist_ok=True, parents=True)

print("🚀 Starting model downloads...")
print(f"Models directory: {models_dir}")
print(f"Cache directory: {cache_dir}")

# Models to download (optimized for your system)
models_to_download = [
    {
        "repo": "microsoft/phi-2",
        "name": "phi-2",
        "files": ["config.json", "tokenizer.json", "tokenizer_config.json"],
        "description": "Microsoft Phi-2 - 2.7B parameters, excellent for CPU"
    },
    {
        "repo": "microsoft/DialoGPT-medium",
        "name": "dialogpt-medium", 
        "files": ["config.json", "tokenizer_config.json"],
        "description": "DialoGPT Medium - Good for conversations"
    },
    {
        "repo": "distilbert-base-uncased",
        "name": "distilbert",
        "files": ["config.json", "tokenizer_config.json"],
        "description": "DistilBERT - Fast and efficient"
    }
]

# For GGUF models (if llama-cpp-python is available)
gguf_models = [
    {
        "repo": "TheBloke/phi-2-GGUF",
        "file": "phi-2.Q4_K_M.gguf",
        "name": "phi-2.gguf",
        "description": "Phi-2 GGUF quantized - CPU optimized"
    }
]

def download_transformers_model(model_info):
    """Download transformers model"""
    try:
        print(f"\n📦 Downloading {model_info['name']}...")
        print(f"   {model_info['description']}")
        
        local_dir = models_dir / model_info['name']
        
        # Download full model
        snapshot_download(
            repo_id=model_info['repo'],
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=os.environ.get('HF_TOKEN')
        )
        
        print(f"✅ {model_info['name']} downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {model_info['name']}: {e}")
        return False

def download_gguf_model(model_info):
    """Download GGUF model"""
    try:
        print(f"\n📦 Downloading {model_info['name']} (GGUF)...")
        print(f"   {model_info['description']}")
        
        # Download to cache directory for GGUF
        output_path = cache_dir / model_info['name']
        
        hf_hub_download(
            repo_id=model_info['repo'],
            filename=model_info['file'],
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
            token=os.environ.get('HF_TOKEN')
        )
        
        # Rename file
        downloaded_file = cache_dir / model_info['file']
        if downloaded_file.exists():
            downloaded_file.rename(output_path)
        
        print(f"✅ {model_info['name']} downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not download GGUF model {model_info['name']}: {e}")
        return False

# Download transformers models
success_count = 0
for model in models_to_download:
    if download_transformers_model(model):
        success_count += 1

# Try to download GGUF models
print("\n🔧 Attempting GGUF models (for llama-cpp-python)...")
for model in gguf_models:
    download_gguf_model(model)

print(f"\n✨ Download complete! {success_count}/{len(models_to_download)} models downloaded.")
print("\n📁 Models are stored in:")
print(f"   - Transformers models: {models_dir.absolute()}")
print(f"   - GGUF models: {cache_dir.absolute()}")

# Create a simple model config
config_data = {
    "models": [
        {
            "name": "phi-2",
            "path": str(models_dir / "phi-2"),
            "type": "transformers",
            "context_length": 2048
        },
        {
            "name": "dialogpt",
            "path": str(models_dir / "dialogpt-medium"),
            "type": "transformers",
            "context_length": 1024
        },
        {
            "name": "distilbert",
            "path": str(models_dir / "distilbert"),
            "type": "transformers",
            "context_length": 512
        }
    ]
}

# Save config
import json
config_path = Path("models_config.json")
with open(config_path, 'w') as f:
    json.dump(config_data, f, indent=2)

print(f"\n📝 Model configuration saved to {config_path}")
print("\n🎉 Setup complete! You can now run the main system.")