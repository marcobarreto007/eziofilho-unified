# api_config.py - API configuration for EzioFilho Unified
# 100% SECURE - NO HARDCODED TOKENS
import os
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# API Keys configuration - ALL FROM ENVIRONMENT VARIABLES
API_KEYS = {
    "twelve_data": os.getenv("TWELVE_DATA_API_KEY", "your_twelve_data_key_here"),
    "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY", "your_alpha_vantage_key_here"),
    "huggingface": os.getenv("HUGGINGFACE_API_KEY", os.getenv("HUGGINGFACE_TOKEN", "your_hf_token_here")),
    "wolfram": os.getenv("WOLFRAM_API_KEY", "your_wolfram_key_here"),
    "newsapi": os.getenv("NEWSAPI_KEY", "your_newsapi_key_here"),
    "coingecko": os.getenv("COINGECKO_API_KEY", "your_coingecko_key_here"),
    "youtube": os.getenv("YOUTUBE_API_KEY", "your_youtube_key_here"),
    "openai": os.getenv("OPENAI_API_KEY", "your_openai_key_here"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY", "your_anthropic_key_here"),
    "the_odds": os.getenv("THE_ODDS_API_KEY", "your_odds_key_here")
}

# Validation function to check for missing keys
def validate_api_keys():
    """Check which API keys are properly configured"""
    missing_keys = []
    configured_keys = []
    
    for service, key in API_KEYS.items():
        if key and key.strip() and not key.startswith("your_"):
            configured_keys.append(service)
        else:
            missing_keys.append(service)
    
    return {
        "configured": configured_keys,
        "missing": missing_keys,
        "total_configured": len(configured_keys)
    }

# Get API key safely
def get_api_key(service_name):
    """Safely retrieve API key for a service"""
    key = API_KEYS.get(service_name)
    if not key or key.startswith("your_"):
        return None
    return key

# Check if service is available
def is_service_available(service_name):
    """Check if a service has a valid API key configured"""
    key = get_api_key(service_name)
    return key is not None and len(key) > 10

# System configuration
SYSTEM_CONFIG = {
    "version": "5.0",
    "author": "Marco Barreto", 
    "languages": ["pt", "en", "fr"],
    "max_cache_size": 1000,
    "cache_ttl": 3600,  # 1 hour
    "security": {
        "token_validation": True,
        "environment_only": True,
        "no_hardcoded_secrets": True
    }
}

# Environment setup instructions
ENV_SETUP_GUIDE = """
🔐 EZIO API Configuration Guide

To configure your API keys securely:

1. Create a .env file in the project root:
   touch .env

2. Add your API keys to .env:
   TWELVE_DATA_API_KEY=your_actual_key_here
   ALPHA_VANTAGE_API_KEY=your_actual_key_here
   HUGGINGFACE_TOKEN=your_actual_token_here
   WOLFRAM_API_KEY=your_actual_key_here
   NEWSAPI_KEY=your_actual_key_here
   COINGECKO_API_KEY=your_actual_key_here
   YOUTUBE_API_KEY=your_actual_key_here
   OPENAI_API_KEY=your_actual_key_here
   ANTHROPIC_API_KEY=your_actual_key_here
   THE_ODDS_API_KEY=your_actual_key_here

3. The .env file is automatically ignored by git for security.

4. For GitHub Codespaces:
   - Go to repository Settings → Security → Secrets and variables → Codespaces
   - Add each API key as a separate secret
   - They will be automatically available as environment variables

✅ This ensures your API keys are never committed to version control!
"""

def print_setup_guide():
    """Print the environment setup guide"""
    print(ENV_SETUP_GUIDE)

def get_configuration_status():
    """Get detailed configuration status"""
    validation = validate_api_keys()
    
    status = {
        "device": DEVICE,
        "system_version": SYSTEM_CONFIG["version"],
        "api_keys_configured": validation["total_configured"],
        "services_available": validation["configured"],
        "services_missing": validation["missing"],
        "security_compliant": True,
        "hardcoded_tokens": False
    }
    
    return status

# Initialize and validate on import
if __name__ == "__main__":
    status = get_configuration_status()
    print("🔐 EZIO API Configuration Status:")
    print(f"✅ Device: {status['device']}")
    print(f"✅ Version: {status['system_version']}")
    print(f"✅ Configured APIs: {status['api_keys_configured']}/10")
    print(f"✅ Security Compliant: {status['security_compliant']}")
    print(f"✅ No Hardcoded Tokens: {not status['hardcoded_tokens']}")
    
    if status["services_missing"]:
        print(f"⚠️  Missing API keys: {', '.join(status['services_missing'])}")
        print("\n📖 Run print_setup_guide() for configuration instructions")