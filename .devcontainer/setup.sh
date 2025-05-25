#!/bin/bash
# EZIO Financial AI - GitHub Codespaces Setup Script
# Auto-configures environment for trading system (NO LARGE MODELS)

echo "================================================================"
echo "🤖 EZIO FINANCIAL AI - CODESPACES SETUP (LIGHT)"
echo "================================================================"

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -q
sudo apt-get install -y curl wget git

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
python -m pip install --upgrade pip setuptools wheel

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "📋 Installing from requirements.txt..."  
    pip install -r requirements.txt
fi

if [ -f "universal_env/requirements_universal.txt" ]; then
    echo "📋 Installing universal requirements..."
    pip install -r universal_env/requirements_universal.txt
fi

# Install additional packages for financial analysis
echo "💰 Installing financial packages..."
pip install yfinance pandas numpy matplotlib seaborn plotly

# Install AI/ML packages (lightweight versions)
echo "🤖 Installing AI/ML packages (lightweight)..."
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
pip install autogen langchain openai

# Download only small models for testing (optional)
echo "📥 Setting up model configuration (no downloads)..."
cat > model_config.json << EOF
{
    "available_models": {
        "gpt-3.5-turbo": {
            "provider": "openai",
            "type": "cloud",
            "description": "Fast and efficient for most tasks"
        },
        "claude-3-haiku": {
            "provider": "anthropic", 
            "type": "cloud",
            "description": "Fast and cost-effective"
        }
    },
    "local_models": {
        "note": "Local models can be downloaded when needed via HuggingFace Hub"
    }
}
EOF

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p logs
mkdir -p .cache
mkdir -p 09_data_cache/data/shared_cache
mkdir -p config

# Set permissions
chmod +x *.py
chmod +x scripts/*.sh 2>/dev/null || true

# Create default config if not exists
if [ ! -f "config/.env.example" ]; then
    echo "⚙️ Creating config template..."
    cat > config/.env.example << EOF
# EZIO Financial AI Configuration
# Copy this to .env and fill with your values

# OpenAI API (Optional - for GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# Yahoo Finance (Free - no key needed)
YFINANCE_ENABLED=true

# Cache Settings
CACHE_ENABLED=true
CACHE_EXPIRY_HOURS=24

# Expert System Settings
ENABLE_ALL_EXPERTS=true
DEBUG_MODE=false

# GitHub Codespaces
CODESPACES_MODE=true
EOF
fi

# Test Python modules
echo "🧪 Testing Python modules..."
python -c "import yfinance; print('✅ yfinance works!')" 2>/dev/null || echo "❌ yfinance failed"
python -c "import pandas; print('✅ pandas works!')" 2>/dev/null || echo "❌ pandas failed"
python -c "import autogen; print('✅ autogen works!')" 2>/dev/null || echo "❌ autogen failed"

# Create startup script for Codespaces
cat > start_ezio_codespaces.py << 'EOF'
#!/usr/bin/env python3
"""
EZIO Financial AI - GitHub Codespaces Startup
Optimized for cloud development environment
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if all dependencies are installed"""
    required = ['yfinance', 'pandas', 'numpy', 'transformers']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg}")
            missing.append(pkg)
    
    if missing:
        print(f"\n🔧 Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)

def main():
    print("🤖 EZIO Financial AI Trading System")
    print("☁️  GitHub Codespaces Environment")
    print("=" * 50)
    
    print("\n📦 Checking dependencies...")
    check_dependencies()
    
    print("\n🚀 Available commands:")
    print("  python ezio_main.py              # Start main system")
    print("  python 01_core_system/main.py   # Run core system")
    print("  python tests/test_basic.py       # Run tests")
    
    print("\n📊 Testing Yahoo Finance connection...")
    try:
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1d")
        if not data.empty:
            print("✅ Yahoo Finance connection working!")
            print(f"📈 AAPL latest price: ${data['Close'].iloc[-1]:.2f}")
        else:
            print("⚠️  Yahoo Finance connection issue")
    except Exception as e:
        print(f"❌ Yahoo Finance error: {e}")
    
    print("\n🎯 EZIO System Ready for Development!")
    print("💡 Tip: Use 'python start_ezio_codespaces.py' anytime to check status")
    
if __name__ == "__main__":
    main()
EOF

chmod +x start_ezio_codespaces.py

echo "================================================================"
echo "✅ EZIO FINANCIAL AI CODESPACES SETUP COMPLETED!"
echo "🚀 Run: python start_ezio.py"
echo "================================================================"