# fix_lfs_permanently.py - Remove LFS and large files permanently
# Audit Mode: Active - Git LFS removal
# Path: C:\Users\anapa\eziofilho-unified
# User: marcobarreto007
# Date: 2025-05-24 20:25:32 UTC
# Objective: Remove Git LFS tracking and large files

import os
import subprocess
import shutil
from pathlib import Path

print("=" * 70)
print("🔧 FIXING GIT LFS PERMANENTLY")
print("=" * 70)

# Step 1: Remove LFS tracking
print("\n1️⃣ Removing Git LFS...")
commands = [
    "git lfs uninstall",
    "git lfs untrack '*'",
    "git rm -r --cached .",
]

for cmd in commands:
    print(f"   Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, capture_output=True)
        print("   ✅ Done")
    except:
        pass

# Step 2: Remove .gitattributes if it has LFS entries
gitattributes = Path(".gitattributes")
if gitattributes.exists():
    print("\n2️⃣ Cleaning .gitattributes...")
    gitattributes.unlink()
    print("   ✅ Removed .gitattributes")

# Step 3: Remove all large files
print("\n3️⃣ Removing large files...")
large_patterns = [
    "**/*.bin",
    "**/*.safetensors",
    "**/*.gguf",
    "**/*.msgpack",
    "**/*.pt",
    "**/*.pth",
    "**/*.onnx",
    "**/*.h5"
]

removed = 0
for pattern in large_patterns:
    for file in Path(".").glob(pattern):
        try:
            size_mb = file.stat().st_size / 1024 / 1024
            file.unlink()
            print(f"   ✅ Removed: {file.name} ({size_mb:.1f}MB)")
            removed += 1
        except:
            pass

# Step 4: Remove model directories
print("\n4️⃣ Removing model directories...")
model_dirs = [
    "models",
    "03_models_storage/models",
    ".git/lfs"
]

for dir_path in model_dirs:
    if Path(dir_path).exists():
        try:
            shutil.rmtree(dir_path)
            print(f"   ✅ Removed: {dir_path}")
        except:
            pass

# Step 5: Create comprehensive .gitignore
print("\n5️⃣ Creating comprehensive .gitignore...")
gitignore_content = """# Large files
*.bin
*.safetensors
*.gguf
*.msgpack
*.pt
*.pth
*.onnx
*.h5
*.ckpt
*.pkl
*.tar
*.gz
*.zip

# Model directories
models/
03_models_storage/models/
model_cache/
.cache/

# LFS
.git/lfs/

# Python
__pycache__/
*.py[cod]
*.pyc
.Python
env/
venv/
.venv/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Database
*.db
*.sqlite

# Environment
.env
.env.local

# Test
test_outputs/
tmp/
"""

with open(".gitignore", "w") as f:
    f.write(gitignore_content)
print("   ✅ Created comprehensive .gitignore")

# Step 6: Add all files and commit
print("\n6️⃣ Preparing clean commit...")
subprocess.run("git add .", shell=True)
subprocess.run('git commit -m "fix: Remove LFS and large files permanently"', shell=True)

print("\n" + "="*70)
print("✅ FIXED! Now push without LFS:")
print("="*70)
print("\n📋 Run this command:")
print("   git push -u origin main --force")
print("\n💡 This will:")
print("   - Push without any large files")
print("   - No LFS tracking")
print("   - Fast upload")
print("\n🚀 Your repository will be clean and ready!")

input("\nPress Enter to exit...")