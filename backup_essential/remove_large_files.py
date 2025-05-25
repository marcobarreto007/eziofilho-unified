# remove_large_files.py - Remove large model files for GitHub
# Audit Mode: Active - File cleanup for Git
# Path: C:\Users\anapa\eziofilho-unified
# User: marcobarreto007
# Date: 2025-05-24 20:12:37 UTC
# Objective: Remove large files blocking GitHub push

import os
import shutil
from pathlib import Path

print("=" * 70)
print("🧹 REMOVING LARGE MODEL FILES")
print("=" * 70)

# Patterns to remove
large_file_patterns = [
    "*.bin",
    "*.safetensors", 
    "*.gguf",
    "*.pt",
    "*.pth",
    "*.h5",
    "*.ckpt",
    "*.msgpack",
    "*.onnx"
]

# Directories to clean
dirs_to_remove = [
    "models",
    "03_models_storage/models",
    ".git/lfs"
]

removed_count = 0
total_size = 0

# Remove directories
for dir_path in dirs_to_remove:
    full_path = Path(dir_path)
    if full_path.exists():
        try:
            size = sum(f.stat().st_size for f in full_path.rglob('*') if f.is_file())
            total_size += size
            shutil.rmtree(full_path)
            print(f"✅ Removed directory: {dir_path} ({size/1024/1024:.1f}MB)")
            removed_count += 1
        except Exception as e:
            print(f"⚠️  Could not remove {dir_path}: {e}")

# Remove large files by pattern
for pattern in large_file_patterns:
    for file in Path(".").rglob(pattern):
        try:
            size = file.stat().st_size
            if size > 50 * 1024 * 1024:  # Files > 50MB
                total_size += size
                file.unlink()
                print(f"✅ Removed: {file} ({size/1024/1024:.1f}MB)")
                removed_count += 1
        except Exception as e:
            print(f"⚠️  Could not remove {file}: {e}")

print(f"\n📊 Summary:")
print(f"   Files/dirs removed: {removed_count}")
print(f"   Space freed: {total_size/1024/1024:.1f}MB")

# Update .gitignore
gitignore_additions = """
# Large model files (added by cleanup script)
*.bin
*.safetensors
*.gguf
*.pt
*.pth
*.h5
*.ckpt
*.msgpack
*.onnx
models/
03_models_storage/models/
"""

with open(".gitignore", "a") as f:
    f.write(gitignore_additions)
print("\n✅ Updated .gitignore")

# Create download script for models
download_script = '''#!/usr/bin/env python3
"""
download_models.py - Download required models after cloning
Run this after cloning the repository to get the models
"""

print("📥 Model Downloader")
print("Models will be downloaded when needed by the system")
print("PHI-3 will be loaded from Hugging Face on first run")
'''

with open("download_models.py", "w") as f:
    f.write(download_script)
print("✅ Created download_models.py")

print("\n🎯 Next steps:")
print("1. Run: git add .")
print("2. Run: git commit -m 'fix: Remove large model files'")
print("3. Run: git push -u origin main")
print("\n✅ Repository is now ready for GitHub!")

input("\nPress Enter to exit...")