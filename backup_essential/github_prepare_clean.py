# github_prepare_clean.py - Clean and prepare for GitHub
# Audit Mode: Active - GitHub preparation
# Path: C:\Users\anapa\eziofilho-unified
# User: marcobarreto007
# Date: 2025-05-24 17:42:52 UTC
# Objective: Clean unnecessary files and prepare for GitHub

import os
import shutil
from pathlib import Path
import json

print("=" * 70)
print("🧹 CLEANING AND PREPARING FOR GITHUB")
print("=" * 70)

# Current directory
base_path = Path.cwd()
print(f"📁 Working in: {base_path}")

# Remove junk files
junk_files = [
    "#", "1.20.0", "1.3", "2.11.3", "25.1.1", "3.7", "4.5.0", "5.6.1",
    "cls", "CMAKE_ARGS", "git", "llama-cpp-python)", "main", "mkdir", 
    "move", "Music", "OneDrive", "pip", "py", "Saved", "Searches", 
    "set", "timeout)", "Videos", ".matplotlib", ".vscode", "Favorites",
    "autogen-agentchat", "eziofilho-unified"
]

removed = 0
for junk in junk_files:
    junk_path = base_path / junk
    if junk_path.exists():
        try:
            junk_path.unlink()
            removed += 1
            print(f"✓ Removed: {junk}")
        except:
            pass

print(f"\n🧹 Cleaned {removed} junk files")

# Create proper .gitignore
gitignore = """# Python
__pycache__/
*.py[cod]
*.pyc
.Python
env/
venv/
.venv/

# Cache
.cache/
*.cache
chunks/

# Models
*.gguf
*.bin
*.safetensors
*.pt
*.pth
models/phi3-mini/
03_models_storage/*.gguf

# Data
*.db
*.sqlite
09_data_cache/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Environment
.env
.env.local

# Logs
*.log

# Test folders
testes_*/
tests_modelos/
autogen_generated/
universal_env/

# Large files
ezio_copilot_commands_*.txt
ezio_organization_plan_*.json
ezio_organizer_*.bat
"""

with open(".gitignore", "w") as f:
    f.write(gitignore)
print("✅ Created .gitignore")

# Create README.md
readme = """# 🚀 EzioFilho Unified - Advanced Financial AI System

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/marcobarreto007/eziofilho-unified)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![AI](https://img.shields.io/badge/AI-AutoGen%20%2B%20LangChain-red)

## 🎯 Overview

EzioFilho Unified is a cutting-edge financial AI assistant combining:
- 🤖 **Multi-Agent System** (AutoGen) - 9 specialized agents
- 📚 **RAG Capabilities** (LangChain) - Document analysis
- 🏠 **Local LLM Support** (PHI-3) - No API keys needed
- 🌍 **Multilingual** - PT/EN/FR support
- 📊 **Real-time Analysis** - Crypto, stocks, forex
- 🔮 **ML Predictions** - Advanced analytics

## 🚀 Quick Start

### GitHub Codespaces (Recommended)
1. Click badge above or go to Code → Codespaces → Create codespace
2. Wait for environment setup (~2 min)
3. Run: `python ezio_complete_system_fixed.py`

### Local Setup
```bash
cd eziofilho-unified
py -m pip install -r requirements.txt
py ezio_complete_system_fixed.py