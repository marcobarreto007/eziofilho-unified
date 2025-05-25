# download_phi3_mini.py - Download PHI-3 Mini model
# Audit Mode: Active - Download verified AI models
# Path: C:\Users\anapa\EzioFilhoUnified\ezio_experts\models
# User: marcobarreto007
# Date: 2025-05-24 16:59:51 UTC
# Objective: Download PHI-3 Mini 4K Instruct model for local execution

import os
import sys
import requests
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib
import json
from datetime import datetime

class ModelDownloader:
    """Secure model downloader with verification"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.models_dir = self.base_path / "models"
        self.models_dir.mkdir(exist_ok=True)
        
    def download_with_progress(self, url: str, filename: str, 
                             expected_hash: Optional[str] = None) -> bool:
        """Download file with progress bar and hash verification"""
        filepath = self.models_dir / filename
        
        # Check if already exists
        if filepath.exists():
            print(f"✅ Model already exists: {filename}")
            return True
            
        print(f"📥 Downloading: {filename}")
        print(f"📍 From: {url}")
        
        try:
            # Download with streaming
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            
            # Write to temporary file first
            temp_file = filepath.with_suffix('.tmp')
            
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress bar
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            bars = '█' * int(percent / 2)
                            spaces = ' ' * (50 - len(bars))
                            print(f"\r[{bars}{spaces}] {percent:.1f}% "
                                  f"({downloaded / 1024 / 1024:.1f} MB)", end='')
                                  
            print("\n✅ Download complete!")
            
            # Verify hash if provided
            if expected_hash:
                print("🔐 Verifying integrity...")
                if not self.verify_hash(temp_file, expected_hash):
                    print("❌ Hash verification failed!")
                    temp_file.unlink()
                    return False
                print("✅ Integrity verified!")
                
            # Move to final location
            temp_file.rename(filepath)
            return True
            
        except Exception as e:
            print(f"\n❌ Download error: {str(e)}")
            # Clean up temp file if exists
            if temp_file.exists():
                temp_file.unlink()
            return False
            
    def verify_hash(self, filepath: Path, expected_hash: str) -> bool:
        """Verify file hash"""
        sha256_hash = hashlib.sha256()
        
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
                
        return sha256_hash.hexdigest() == expected_hash
        
    def download_phi3_mini(self) -> bool:
        """Download Microsoft PHI-3 Mini 4K Instruct"""
        print("=" * 70)
        print("🤖 PHI-3 MINI 4K INSTRUCT DOWNLOAD")
        print("=" * 70)
        
        # Model options (GGUF format for llama.cpp)
        models = {
            "phi3-mini-q4": {
                "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
                "filename": "phi-3-mini-4k-instruct-q4.gguf",
                "size": "2.2 GB",
                "description": "4-bit quantized, best performance/quality ratio"
            },
            "phi3-mini-q8": {
                "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q8.gguf",
                "filename": "phi-3-mini-4k-instruct-q8.gguf",
                "size": "4.1 GB",
                "description": "8-bit quantized, higher quality"
            }
        }
        
        # Show options
        print("\n📋 Available versions:")
        for key, model in models.items():
            print(f"\n{key}:")
            print(f"  Size: {model['size']}")
            print(f"  Description: {model['description']}")
            
        # Select model
        choice = input("\n🔍 Which version? (phi3-mini-q4 recommended) [q4/q8]: ").strip().lower()
        
        if choice == "q8":
            model_key = "phi3-mini-q8"
        else:
            model_key = "phi3-mini-q4"
            
        model = models[model_key]
        
        print(f"\n📦 Downloading: {model['filename']}")
        print(f"💾 Size: {model['size']}")
        
        # Download model
        success = self.download_with_progress(
            model['url'],
            model['filename']
        )
        
        if success:
            # Save model info
            info = {
                "model": "PHI-3 Mini 4K Instruct",
                "version": model_key,
                "filename": model['filename'],
                "path": str(self.models_dir / model['filename']),
                "download_date": datetime.utcnow().isoformat(),
                "size": model['size']
            }
            
            info_file = self.models_dir / "phi3_info.json"
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=2)
                
            print(f"\n✅ Model saved to: {self.models_dir / model['filename']}")
            print(f"📄 Info saved to: {info_file}")
            
            # Show usage example
            self.show_usage_example(model['filename'])
            
        return success
        
    def show_usage_example(self, filename: str):
        """Show how to use the downloaded model"""
        print("\n" + "=" * 70)
        print("💡 HOW TO USE THIS MODEL")
        print("=" * 70)
        
        print("\n1️⃣ Install llama-cpp-python:")
        print("   pip install llama-cpp-python")
        
        print("\n2️⃣ Example code:")
        print("""
from llama_cpp import Llama

# Load model
llm = Llama(
    model_path="models/{filename}",
    n_ctx=4096,  # Context window
    n_threads=8,  # CPU threads
    n_gpu_layers=35  # GPU layers (if CUDA available)
)

# Generate response
response = llm(
    "You are a helpful assistant. User: What is Bitcoin? Assistant:",
    max_tokens=256,
    temperature=0.7,
    stop=["User:", "\\n\\n"]
)

print(response['choices'][0]['text'])
""".format(filename=filename))
        
        print("\n3️⃣ Or use with EzioFilho system:")
        print(f"   python ezio_local_phi3.py --model models/{filename}")

def download_alternative_models():
    """Download alternative models if PHI-3 fails"""
    print("\n🔄 Alternative models available:")
    
    alternatives = {
        "1": {
            "name": "TinyLlama 1.1B",
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "filename": "tinyllama-1.1b-chat-q4.gguf",
            "size": "669 MB",
            "description": "Very small, fast on CPU"
        },
        "2": {
            "name": "Mistral 7B Instruct",
            "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            "filename": "mistral-7b-instruct-q4.gguf",
            "size": "4.1 GB",
            "description": "High quality, needs more RAM"
        }
    }
    
    for key, model in alternatives.items():
        print(f"\n{key}. {model['name']}")
        print(f"   Size: {model['size']}")
        print(f"   {model['description']}")
        
    choice = input("\n📥 Download alternative? [1/2/n]: ").strip()
    
    if choice in alternatives:
        model = alternatives[choice]
        downloader = ModelDownloader()
        downloader.download_with_progress(model['url'], model['filename'])

# Main execution
if __name__ == "__main__":
    try:
        # Security check - ensure we're in the right directory
        current_path = Path.cwd()
        print(f"📁 Current directory: {current_path}")
        
        # Create downloader
        downloader = ModelDownloader()
        
        # Download PHI-3
        success = downloader.download_phi3_mini()
        
        if not success:
            print("\n❌ PHI-3 download failed!")
            download_alternative_models()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Download cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()