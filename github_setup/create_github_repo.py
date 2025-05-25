# create_github_repo.py - Setup GitHub repository for EzioFilho
# Audit Mode: Active - GitHub integration
# Path: C:\Users\anapa\EzioFilhoUnified\ezio_experts\github_setup
# User: marcobarreto007
# Date: 2025-05-24 17:36:29 UTC
# Objective: Create and setup GitHub repository with Codespaces

import os
import subprocess
import json
from pathlib import Path
from datetime import datetime

print("=" * 70)
print("🚀 EZIOFILHO GITHUB SETUP")
print("📦 Preparing for GitHub Codespaces")
print("=" * 70)

# Create project structure
project_structure = {
    ".github": {
        "workflows": ["ci.yml", "deploy.yml"],
        "ISSUE_TEMPLATE": ["bug_report.md", "feature_request.md"],
        "dependabot.yml": None
    },
    "src": {
        "agents": ["autogen_system.py", "langchain_rag.py"],
        "models": ["phi3_local.py", "embeddings.py"],
        "api": ["rest_api.py", "websocket.py"],
        "frontend": ["dashboard.html", "app.js", "styles.css"]
    },
    "docker": ["Dockerfile", "docker-compose.yml"],
    "docs": ["README.md", "ARCHITECTURE.md", "API.md"],
    "tests": ["test_agents.py", "test_api.py"],
    ".devcontainer": ["devcontainer.json", "Dockerfile"]
}

# Create .devcontainer/devcontainer.json for Codespaces
devcontainer_config = {
    "name": "EzioFilho Unified",
    "dockerFile": "Dockerfile",
    "features": {
        "ghcr.io/devcontainers/features/python:1": {
            "version": "3.11"
        },
        "ghcr.io/devcontainers/features/node:1": {},
        "ghcr.io/devcontainers/features/docker-in-docker:2": {}
    },
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-python.vscode-pylance",
                "github.copilot",
                "github.copilot-chat",
                "ms-toolsai.jupyter",
                "dbaeumer.vscode-eslint",
                "esbenp.prettier-vscode"
            ],
            "settings": {
                "python.defaultInterpreterPath": "/usr/local/bin/python",
                "python.linting.enabled": true,
                "python.linting.pylintEnabled": true,
                "python.formatting.provider": "black"
            }
        }
    },
    "postCreateCommand": "pip install -r requirements.txt && npm install",
    "forwardPorts": [8000, 3000, 5000],
    "remoteUser": "vscode"
}

# Create Dockerfile for Codespaces
dockerfile_content = """FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    vim \\
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \\
    pip install -r /tmp/requirements.txt

# Install Node.js for frontend
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \\
    apt-get install -y nodejs

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Expose ports
EXPOSE 8000 3000 5000

CMD ["python", "src/api/rest_api.py"]
"""

# Create requirements.txt
requirements = """# EzioFilho Unified Requirements
# Core AI/ML
torch>=2.0.0
transformers>=4.30.0
langchain>=0.1.0
autogen>=0.2.0
sentence-transformers>=2.2.0

# Vector Database
chromadb>=0.4.0
faiss-cpu>=1.7.4

# API & Web
fastapi>=0.100.0
uvicorn>=0.23.0
websockets>=11.0
pydantic>=2.0.0

# Financial Data
yfinance>=0.2.0
pandas>=2.0.0
numpy>=1.24.0
ta>=0.10.0

# Visualization
plotly>=5.15.0
streamlit>=1.25.0

# Database
sqlalchemy>=2.0.0
redis>=4.5.0

# Utils
python-dotenv>=1.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
pytest>=7.4.0
black>=23.0.0
pylint>=2.17.0
"""

# Create README.md
readme_content = """# 🚀 EzioFilho Unified - Advanced Financial AI Assistant

![GitHub Codespaces](https://img.shields.io/badge/GitHub-Codespaces-blue)
![Python](https://img.shields.io/badge/Python-3.11-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Overview

EzioFilho Unified is a comprehensive financial AI assistant that combines:
- 🤖 Multi-agent systems (AutoGen)
- 📚 RAG capabilities (LangChain)
- 🏠 Local LLM support (PHI-3)
- 🌍 Multilingual support (PT/EN/FR)
- 📊 Real-time market analysis
- 🔮 Predictive analytics

## 🚀 Quick Start with GitHub Codespaces

1. Click the green "Code" button
2. Select "Codespaces" tab
3. Click "Create codespace on main"
4. Wait for environment setup
5. Run: `python src/main.py`

## 🏗️ Architecture
