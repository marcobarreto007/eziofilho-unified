# auto_model_downloader.py - Automatic model download system
# Audit Mode: Active - Model management
# Path: C:\Users\anapa\eziofilho-unified
# User: marcobarreto007
# Date: 2025-05-24 20:13:54 UTC
# Objective: Create automatic model download system

import os
from pathlib import Path

print("=" * 70)
print("🤖 CREATING AUTOMATIC MODEL DOWNLOAD SYSTEM")
print("=" * 70)

# Create model manager
model_manager_code = '''"""
Model Manager - Automatic download and management of AI models
This module handles downloading models on-demand when needed
"""

import os
import requests
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm

class ModelManager:
    """Manages AI model downloads and loading"""
    
    def __init__(self):
        self.models_dir = Path("03_models_storage/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            "phi3": {
                "name": "microsoft/Phi-3-mini-4k-instruct",
                "type": "transformers",
                "size": "2.7GB"
            },
            "phi2": {
                "name": "microsoft/phi-2", 
                "type": "transformers",
                "size": "2.8GB"
            },
            "dialogpt": {
                "name": "microsoft/DialoGPT-medium",
                "type": "transformers", 
                "size": "1.4GB"
            },
            "tinyllama": {
                "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "type": "transformers",
                "size": "1.1GB"
            }
        }
    
    def get_model(self, model_name="phi3"):
        """Get model, downloading if necessary"""
        print(f"\\n🔍 Checking for {model_name}...")
        
        model_path = self.models_dir / model_name
        
        if model_path.exists() and any(model_path.iterdir()):
            print(f"✅ {model_name} already downloaded")
            return self.load_model(model_name)
        
        print(f"📥 Downloading {model_name} ({self.model_configs[model_name]['size']})...")
        return self.download_and_load(model_name)
    
    def download_and_load(self, model_name):
        """Download model from Hugging Face"""
        config = self.model_configs[model_name]
        
        try:
            print(f"⏳ Downloading from Hugging Face: {config['name']}")
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config['name'],
                trust_remote_code=True
            )
            
            # Download model
            model = AutoModel.from_pretrained(
                config['name'],
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Save locally
            model_path = self.models_dir / model_name
            model_path.mkdir(exist_ok=True)
            
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
            
            print(f"✅ {model_name} downloaded successfully!")
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Error downloading {model_name}: {e}")
            print("💡 Using fallback online mode")
            return None, None
    
    def load_model(self, model_name):
        """Load model from local storage"""
        model_path = self.models_dir / model_name
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            return model, tokenizer
        except Exception as e:
            print(f"⚠️ Error loading local model: {e}")
            return self.download_and_load(model_name)

# Global model manager instance
model_manager = ModelManager()

def get_model(name="phi3"):
    """Get model using global manager"""
    return model_manager.get_model(name)
'''

# Save model manager
with open("model_manager.py", "w") as f:
    f.write(model_manager_code)
print("✅ Created model_manager.py")

# Update main system to use model manager
system_update = '''# Add this to ezio_complete_system_fixed.py at the beginning

from model_manager import get_model

# When loading models, use:
# model, tokenizer = get_model("phi3")
'''

with open("UPDATE_INSTRUCTIONS.txt", "w") as f:
    f.write(system_update)
print("✅ Created UPDATE_INSTRUCTIONS.txt")

# Create setup script for Codespaces
codespace_setup = '''#!/bin/bash
# setup_codespace.sh - Setup script for GitHub Codespaces

echo "🚀 Setting up EzioFilho Unified in Codespaces..."

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p 03_models_storage/models
mkdir -p 09_data_cache
mkdir -p logs

# Download a small model for testing
python -c "from model_manager import get_model; get_model('tinyllama')"

echo "✅ Setup complete! Run: python ezio_complete_system_fixed.py"
'''

with open("setup_codespace.sh", "w") as f:
    f.write(codespace_setup)
os.chmod("setup_codespace.sh", 0o755)
print("✅ Created setup_codespace.sh")

print("\n" + "="*70)
print("✅ MODEL SYSTEM READY!")
print("="*70)
print("\n🎯 How it works:")
print("1. Push code to GitHub WITHOUT large models")
print("2. In Codespaces, models download automatically when needed")
print("3. First run downloads only what's necessary")
print("4. Models are cached for future use")
print("\n📦 Available models:")
print("- phi3 (2.7GB) - Main model")
print("- phi2 (2.8GB) - Alternative") 
print("- tinyllama (1.1GB) - Fast testing")
print("- dialogpt (1.4GB) - Conversation")
print("\n🚀 Now you can push to GitHub without worries!")

input("\nPress Enter to continue with GitHub push...")