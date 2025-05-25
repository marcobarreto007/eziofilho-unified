# download_small_model.py - Download small model for testing
# Audit Mode: Download verified models only
# Path: C:\Users\anapa\eziofilho-unified\03_models_storage
# User: marcobarreto007
# Date: 2025-05-24 16:55:36 UTC

import requests
import os
from pathlib import Path

print("📥 Downloading small model for testing...")

# Option 1: Download a small GGUF model
url = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
output_path = Path("phi-2-Q4_K_M.gguf")

if not output_path.exists():
    print(f"Downloading from: {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            percent = (downloaded / total_size) * 100
            print(f"\rProgress: {percent:.1f}%", end="")
    
    print(f"\n✅ Downloaded: {output_path}")
else:
    print(f"✅ Model already exists: {output_path}")