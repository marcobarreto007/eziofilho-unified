# system_config.py - Complete system configuration
# Audit Mode: Centralized configuration management
# Path: C:\Users\anapa\eziofilho-unified\config
# User: marcobarreto007
# Date: 2025-05-24 16:41:01 UTC

import os
from pathlib import Path
from typing import Dict, Any
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# System paths
BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "09_data_cache"
MODELS_PATH = BASE_PATH / "03_models_storage"
LOGS_PATH = BASE_PATH / "logs"

# Create directories if they don't exist
for path in [DATA_PATH, MODELS_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# API Configuration
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "coingecko": os.getenv("COINGECKO_API_KEY"),
    "newsapi": os.getenv("NEWSAPI_KEY"),
    "alphavantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
    "twelvedata": os.getenv("TWELVE_DATA_API_KEY"),
    "huggingface": os.getenv("HUGGINGFACE_API_KEY"),
}

# Model Configuration
MODEL_CONFIG = {
    "default_llm": "gpt-4",
    "embeddings_model": "sentence-transformers/all-mpnet-base-v2",
    "local_models": {
        "phi2": "microsoft/phi-2",
        "llama2": "meta-llama/Llama-2-7b-chat-hf",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.1"
    }
}

# Agent Configuration
AGENT_CONFIG = {
    "max_iterations": 10,
    "temperature": 0.7,
    "max_tokens": 2048,
    "timeout": 30,
    "agents": {
        "crypto_expert": {
            "name": "CryptoExpert",
            "description": "Cryptocurrency and DeFi specialist",
            "tools": ["crypto_price", "defi_analytics", "on_chain_data"]
        },
        "stock_analyst": {
            "name": "StockAnalyst", 
            "description": "Equity markets specialist",
            "tools": ["stock_data", "earnings", "technicals"]
        },
        "risk_manager": {
            "name": "RiskManager",
            "description": "Risk assessment specialist",
            "tools": ["var_calculator", "portfolio_analytics", "stress_test"]
        }
    }
}

# System Features
FEATURES = {
    "autogen": True,
    "langchain": True,
    "rag_system": True,
    "voice_interface": False,
    "web_interface": False,
    "api_server": False,
    "real_time_data": True,
    "backtesting": True,
    "ml_predictions": True
}

# Database Configuration
DATABASE_CONFIG = {
    "vector_db": {
        "type": "chromadb",
        "persist_directory": str(DATA_PATH / "chroma_db")
    },
    "cache_db": {
        "type": "sqlite",
        "path": str(DATA_PATH / "cache.db")
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(LOGS_PATH / "ezio_system.log"),
    "max_size": "10MB",
    "backup_count": 5
}

# Export configuration
CONFIG = {
    "api_keys": API_KEYS,
    "models": MODEL_CONFIG,
    "agents": AGENT_CONFIG,
    "features": FEATURES,
    "database": DATABASE_CONFIG,
    "logging": LOGGING_CONFIG,
    "paths": {
        "base": str(BASE_PATH),
        "data": str(DATA_PATH),
        "models": str(MODELS_PATH),
        "logs": str(LOGS_PATH)
    }
}

# Save configuration to file
config_file = BASE_PATH / "config" / "system_config.json"
with open(config_file, "w") as f:
    json.dump(CONFIG, f, indent=2)

print(f"✅ Configuration saved to: {config_file}")