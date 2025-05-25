# github_prepare_clean_fixed.py - Clean and prepare for GitHub
# Audit Mode: Active - GitHub preparation
# Path: C:\Users\anapa\eziofilho-unified
# User: marcobarreto007
# Date: 2025-05-24 17:44:45 UTC
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
gitignore_content = """# Python
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
    f.write(gitignore_content)
print("✅ Created .gitignore")

# Create README.md - FIXED QUOTES
readme_lines = [
    "# 🚀 EzioFilho Unified - Advanced Financial AI System",
    "",
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/marcobarreto007/eziofilho-unified)",
    "![Python](https://img.shields.io/badge/Python-3.11-blue)",
    "![AI](https://img.shields.io/badge/AI-AutoGen%20%2B%20LangChain-red)",
    "",
    "## 🎯 Overview",
    "",
    "EzioFilho Unified is a cutting-edge financial AI assistant combining:",
    "- 🤖 **Multi-Agent System** (AutoGen) - 9 specialized agents",
    "- 📚 **RAG Capabilities** (LangChain) - Document analysis",
    "- 🏠 **Local LLM Support** (PHI-3) - No API keys needed",
    "- 🌍 **Multilingual** - PT/EN/FR support",
    "- 📊 **Real-time Analysis** - Crypto, stocks, forex",
    "- 🔮 **ML Predictions** - Advanced analytics",
    "",
    "## 🚀 Quick Start",
    "",
    "### GitHub Codespaces (Recommended)",
    "1. Click badge above or go to Code → Codespaces → Create codespace",
    "2. Wait for environment setup (~2 min)",
    "3. Run: `python ezio_complete_system_fixed.py`",
    "",
    "### Local Setup",
    "```bash",
    "cd eziofilho-unified",
    "py -m pip install -r requirements.txt",
    "py ezio_complete_system_fixed.py",
    "```",
    "",
    "## 📁 Project Structure",
    "",
    "```",
    "eziofilho-unified/",
    "├── 01_core_system/      # Core system components",
    "├── 02_experts_modules/  # AI expert agents",
    "│   ├── autogen_agents/  # AutoGen multi-agent",
    "│   ├── langchain_system/# LangChain RAG",
    "│   └── local_llm/       # Local model support",
    "├── 03_models_storage/   # Model storage",
    "├── config/              # Configuration",
    "├── ezio_experts/        # Main expert modules",
    "└── core/                # Core utilities",
    "```",
    "",
    "## 🛠️ Features",
    "",
    "- ✅ Cryptocurrency real-time analysis",
    "- ✅ Stock market insights & predictions",
    "- ✅ Risk assessment & portfolio optimization",
    "- ✅ News sentiment analysis",
    "- ✅ Technical indicators & patterns",
    "- ✅ Multi-language support (PT/EN/FR)",
    "- ✅ Local execution (no cloud dependency)",
    "",
    "## 👨‍💻 Author",
    "",
    "**Marco Barreto** (@marcobarreto007)",
    "",
    "---",
    "*\"Democratizing financial AI for everyone\"* 🚀"
]

with open("README.md", "w", encoding="utf-8") as f:
    f.write("\n".join(readme_lines))
print("✅ Created README.md")

# Create requirements.txt
requirements_lines = [
    "# Core Dependencies",
    "python-dotenv==1.0.0",
    "requests>=2.31.0",
    "beautifulsoup4>=4.12.0",
    "colorama>=0.4.6",
    "tqdm>=4.65.0",
    "",
    "# AI/ML Frameworks",
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "langchain>=0.1.0",
    "autogen>=0.2.0",
    "sentence-transformers>=2.2.0",
    "",
    "# Financial Data",
    "yfinance>=0.2.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "ta>=0.10.0",
    "",
    "# Web Framework",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "streamlit>=1.25.0",
    "",
    "# Vector Database",
    "chromadb>=0.4.0",
    "",
    "# Visualization",
    "plotly>=5.15.0",
    "matplotlib>=3.7.0"
]

with open("requirements.txt", "w") as f:
    f.write("\n".join(requirements_lines))
print("✅ Created requirements.txt")

# Create .devcontainer for Codespaces
devcontainer_dir = Path(".devcontainer")
devcontainer_dir.mkdir(exist_ok=True)

devcontainer = {
    "name": "EzioFilho Unified",
    "image": "mcr.microsoft.com/devcontainers/python:3.11",
    "features": {
        "ghcr.io/devcontainers/features/python:1": {
            "version": "3.11"
        },
        "ghcr.io/devcontainers/features/github-cli:1": {},
        "ghcr.io/devcontainers/features/node:1": {}
    },
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-python.vscode-pylance",
                "github.copilot",
                "github.copilot-chat"
            ],
            "settings": {
                "python.defaultInterpreterPath": "/usr/local/bin/python"
            }
        }
    },
    "postCreateCommand": "pip install -r requirements.txt",
    "forwardPorts": [8000, 3000, 5000, 8501],
    "remoteUser": "vscode"
}

with open(devcontainer_dir / "devcontainer.json", "w") as f:
    json.dump(devcontainer, f, indent=2)
print("✅ Created .devcontainer/")

# Git init
print("\n📦 Initializing Git...")
try:
    os.system("git init")
    os.system("git add .")
    os.system('git commit -m "Initial commit: EzioFilho Unified - Advanced Financial AI System"')
except:
    pass

print("\n" + "=" * 70)
print("✅ PROJECT READY FOR GITHUB!")
print("=" * 70)
print("\n📋 Next steps:")
print("\n1. Create repository on GitHub:")
print("   https://github.com/new")
print("   Name: eziofilho-unified")
print("   Description: Advanced Financial AI System with AutoGen + LangChain")
print("\n2. Push to GitHub:")
print("   git remote add origin https://github.com/marcobarreto007/eziofilho-unified.git")
print("   git branch -M main") 
print("   git push -u origin main")
print("\n3. Open in Codespaces!")
print("   Click 'Code' → 'Codespaces' → 'Create codespace on main'")
print("\n🚀 Your AI system will be ready in the cloud!")

input("\n✅ Press Enter to exit...")