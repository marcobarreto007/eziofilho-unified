# check_local_models.py - Check available local models
# Audit Mode: Local model inventory
# Path: C:\Users\anapa\eziofilho-unified\03_models_storage
# User: marcobarreto007
# Date: 2025-05-24 16:55:36 UTC

import os
import json
from pathlib import Path
import sys

print("=" * 60)
print("🔍 CHECKING LOCAL MODELS")
print("=" * 60)

# Check models_config.json
config_path = Path("C:/Users/anapa/eziofilho-unified/models_config.json")
if config_path.exists():
    print("\n📄 models_config.json found!")
    with open(config_path, "r") as f:
        config = json.load(f)
        print(f"Models configured: {json.dumps(config, indent=2)}")
else:
    print("\n❌ models_config.json not found")

# Check for GGUF models
print("\n🔍 Searching for GGUF models...")
search_paths = [
    Path("C:/Users/anapa/eziofilho-unified/models"),
    Path("C:/Users/anapa/eziofilho-unified/03_models_storage"),
    Path("C:/Users/anapa/.cache"),
    Path("C:/Users/anapa/AppData/Local"),
]

gguf_files = []
for search_path in search_paths:
    if search_path.exists():
        for gguf in search_path.rglob("*.gguf"):
            size_mb = gguf.stat().st_size / (1024 * 1024)
            gguf_files.append({
                "name": gguf.name,
                "path": str(gguf),
                "size_mb": round(size_mb, 2)
            })
            print(f"✓ Found: {gguf.name} ({size_mb:.0f} MB)")

# Check HuggingFace cache
hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
if hf_cache.exists():
    print(f"\n📁 HuggingFace cache: {hf_cache}")
    model_dirs = [d for d in hf_cache.iterdir() if d.is_dir()]
    print(f"Found {len(model_dirs)} cached models")

# Summary
print("\n" + "=" * 60)
print(f"📊 SUMMARY: {len(gguf_files)} GGUF models found")
print("=" * 60)

# Save inventory
inventory = {
    "gguf_models": gguf_files,
    "config_exists": config_path.exists()
}

with open("model_inventory.json", "w") as f:
    json.dump(inventory, f, indent=2)

print("\n✅ Inventory saved to model_inventory.json")