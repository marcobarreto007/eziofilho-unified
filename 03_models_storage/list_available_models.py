# list_available_models.py - List all downloaded models
# Audit Mode: Scan for available local models
# Path: C:\Users\anapa\eziofilho-unified\03_models_storage
# User: marcobarreto007
# Date: 2025-05-24 16:47:44 UTC

import os
import sys
from pathlib import Path
import json
from datetime import datetime

print("=" * 80)
print("🔍 SCANNING FOR DOWNLOADED MODELS")
print("=" * 80)

# Common model cache locations
cache_dirs = [
    Path.home() / ".cache" / "huggingface" / "hub",
    Path("C:/Users/anapa/.cache/huggingface/hub"),
    Path("C:/Users/anapa/eziofilho-unified/models"),
    Path("C:/Users/anapa/eziofilho-unified/03_models_storage"),
    Path("D:/models"),  # Check other drives
    Path("E:/models")
]

# Known model patterns
model_patterns = [
    "phi-3", "phi3", "phi-2", "phi2",
    "llama", "mistral", "gpt4all",
    "ggml", "gguf", "pytorch_model",
    "model.safetensors"
]

found_models = []

# Scan each directory
for cache_dir in cache_dirs:
    if cache_dir.exists():
        print(f"\n📁 Scanning: {cache_dir}")
        
        # Look for model files
        for pattern in ["*.bin", "*.gguf", "*.ggml", "*.safetensors", "*.pt"]:
            for model_file in cache_dir.rglob(pattern):
                # Get size in MB
                size_mb = model_file.stat().st_size / (1024 * 1024)
                
                # Only show models > 100MB
                if size_mb > 100:
                    model_info = {
                        "name": model_file.name,
                        "path": str(model_file),
                        "size_mb": round(size_mb, 2),
                        "parent": model_file.parent.name
                    }
                    found_models.append(model_info)
                    print(f"  ✓ Found: {model_file.name} ({size_mb:.0f} MB)")

# Check models_config.json
config_path = Path("C:/Users/anapa/eziofilho-unified/models_config.json")
if config_path.exists():
    print(f"\n📄 Reading models_config.json...")
    with open(config_path, "r") as f:
        config = json.load(f)
        print(f"  Models in config: {list(config.keys())}")

# Check LM Studio models
lm_studio_path = Path("C:/Users/anapa/.cache/lm-studio/models")
if lm_studio_path.exists():
    print(f"\n🤖 LM Studio models:")
    for model_dir in lm_studio_path.iterdir():
        if model_dir.is_dir():
            print(f"  ✓ {model_dir.name}")

# Summary
print("\n" + "=" * 80)
print(f"📊 SUMMARY: Found {len(found_models)} models")
print("=" * 80)

# Save report
report = {
    "scan_date": datetime.utcnow().isoformat(),
    "models_found": len(found_models),
    "details": found_models
}

report_path = Path("03_models_storage/model_inventory.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n📄 Report saved to: {report_path}")