# download_phi3.py - Download PHI-3 model for EzioFilho
# Audit Mode: Active - Secure model download
# Path: C:\Users\anapa\EzioFilhoUnified\ezio_experts\models
# User: marcobarreto007
# Date: 2025-05-24 17:03:44 UTC
# Objective: Download and setup PHI-3 Mini 4K Instruct

import os
import subprocess
import sys
from pathlib import Path

print("=" * 70)
print("🚀 EZIOFILHO MODEL DOWNLOADER")
print("📥 Downloading PHI-3 Mini 4K Instruct")
print("=" * 70)

# Create models directory
models_dir = Path(".")
models_dir.mkdir(exist_ok=True)

print("\n1️⃣ Installing required packages...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "tqdm", "huggingface-hub"])
    print("✅ Packages installed!")
except Exception as e:
    print(f"❌ Error installing packages: {e}")
    sys.exit(1)

print("\n2️⃣ Downloading PHI-3 model...")

# Import after installation
try:
    from huggingface_hub import snapshot_download
    
    # Download PHI-3 Mini
    print("📦 Model: microsoft/Phi-3-mini-4k-instruct")
    print("💾 This will take a few minutes...")
    
    model_path = snapshot_download(
        repo_id="microsoft/Phi-3-mini-4k-instruct",
        cache_dir="./models",
        local_dir="./models/phi3-mini",
        local_dir_use_symlinks=False
    )
    
    print(f"\n✅ Model downloaded to: {model_path}")
    
    # Create config file
    config = {
        "model": "PHI-3 Mini 4K Instruct",
        "path": str(model_path),
        "ready": True
    }
    
    import json
    with open("model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ SUCCESS! Model ready to use!")
    print("\n💡 Next steps:")
    print("1. Run: python ezio_experts\\core\\ezio_main_system.py")
    print("2. The system will use this model automatically")
    
except Exception as e:
    print(f"\n❌ Download error: {e}")
    print("\n🔄 Alternative: Download GGUF version")
    print("Visit: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf")
    
input("\n✅ Press Enter to exit...")