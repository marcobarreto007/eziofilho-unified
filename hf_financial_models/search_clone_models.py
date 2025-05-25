# search_clone_models.py - Search and clone financial models from HuggingFace
# Audit Mode: Active - HF Pro models search
# Path: C:\Users\anapa\EzioFilhoUnified\ezio_experts\hf_financial_models
# User: marcobarreto007
# Date: 2025-05-24 21:07:59 UTC
# Objective: Search HuggingFace for ready financial models to clone

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import subprocess
import requests
from datetime import datetime

print("=" * 80)
print("🔍 HUGGING FACE FINANCIAL MODELS SEARCH")
print("=" * 80)

# Best financial models on HuggingFace
RECOMMENDED_MODELS = {
    "financial_llms": [
        {
            "name": "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
            "description": "Financial forecasting for Dow 30 stocks",
            "size": "7B",
            "type": "llama2-based"
        },
        {
            "name": "FinGPT/fingpt-sentiment_llama2-7b_lora",
            "description": "Financial sentiment analysis",
            "size": "7B", 
            "type": "llama2-based"
        },
        {
            "name": "TheBloke/finance-LLM-GGUF",
            "description": "Quantized finance model",
            "size": "7B",
            "type": "gguf"
        },
        {
            "name": "mrm8488/financial-bert-small",
            "description": "BERT for financial text",
            "size": "small",
            "type": "bert"
        },
        {
            "name": "ProsusAI/finbert",
            "description": "Financial sentiment BERT",
            "size": "base",
            "type": "bert"
        },
        {
            "name": "yiyanghkust/finbert-tone",
            "description": "Financial tone analysis",
            "size": "base",
            "type": "bert"
        },
        {
            "name": "StephanAkkerman/FinTwitBERT",
            "description": "Twitter financial sentiment",
            "size": "base",
            "type": "bert"
        },
        {
            "name": "hateminers/finance-ner-v0.0.9-finer-139",
            "description": "Financial NER model",
            "size": "base",
            "type": "bert"
        }
    ],
    "trading_models": [
        {
            "name": "Linq-AI-Research/Mamba-7B",
            "description": "Advanced architecture for time series",
            "size": "7B",
            "type": "mamba"
        },
        {
            "name": "microsoft/phi-2",
            "description": "Small but powerful",
            "size": "2.7B",
            "type": "phi"
        },
        {
            "name": "stabilityai/stablelm-2-1_6b",
            "description": "Stable small model",
            "size": "1.6B",
            "type": "stablelm"
        }
    ],
    "autogen_compatible": [
        {
            "name": "lmsys/vicuna-7b-v1.5",
            "description": "Great for multi-agent",
            "size": "7B",
            "type": "llama"
        },
        {
            "name": "teknium/OpenHermes-2.5-Mistral-7B",
            "description": "Excellent instruction following",
            "size": "7B",
            "type": "mistral"
        },
        {
            "name": "NousResearch/Hermes-2-Pro-Mistral-7B",
            "description": "Professional multi-agent",
            "size": "7B",
            "type": "mistral"
        }
    ]
}

class HFModelManager:
    """Manage HuggingFace model downloads"""
    
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.base_path = Path("C:/Users/anapa/EzioFilhoUnified/models")
        self.base_path.mkdir(exist_ok=True)
        
    def search_models(self, query: str = "finance") -> List[Dict]:
        """Search models on HuggingFace"""
        print(f"\n🔍 Searching for '{query}' models...")
        
        url = "https://huggingface.co/api/models"
        params = {
            "search": query,
            "sort": "downloads",
            "direction": -1,
            "limit": 20,
            "filter": "text-generation"
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                models = response.json()
                filtered = []
                for model in models:
                    if any(keyword in model['id'].lower() for keyword in ['finance', 'fin', 'stock', 'trade']):
                        filtered.append({
                            "id": model['id'],
                            "downloads": model.get('downloads', 0),
                            "likes": model.get('likes', 0)
                        })
                return filtered
        except Exception as e:
            print(f"❌ Search error: {e}")
        
        return []
    
    def clone_model(self, model_id: str, use_lfs: bool = True) -> bool:
        """Clone model from HuggingFace"""
        print(f"\n📥 Cloning {model_id}...")
        
        model_path = self.base_path / model_id.replace("/", "_")
        
        if model_path.exists():
            print(f"✅ Model already exists: {model_path}")
            return True
        
        # Git clone command
        cmd = [
            "git", "clone",
            f"https://huggingface.co/{model_id}",
            str(model_path)
        ]
        
        if not use_lfs:
            cmd.extend(["--depth", "1"])
        
        try:
            # Clone repository
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Cloned successfully: {model_path}")
                
                # If not using LFS, download files separately
                if not use_lfs:
                    self._download_model_files(model_id, model_path)
                
                return True
            else:
                print(f"❌ Clone failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error cloning: {e}")
        
        return False
    
    def _download_model_files(self, model_id: str, model_path: Path):
        """Download model files without LFS"""
        print(f"📥 Downloading model files...")
        
        # Key files to download
        files = [
            "config.json",
            "tokenizer_config.json", 
            "tokenizer.json",
            "special_tokens_map.json",
            "pytorch_model.bin",
            "model.safetensors"
        ]
        
        for file in files:
            url = f"https://huggingface.co/{model_id}/resolve/main/{file}"
            file_path = model_path / file
            
            try:
                response = requests.get(url, stream=True, allow_redirects=True)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"  ✅ Downloaded: {file}")
            except:
                print(f"  ⚠️  Skipped: {file}")
    
    def setup_autogen_config(self, models: List[str]) -> Dict:
        """Create AutoGen configuration for local models"""
        config = {
            "config_list": [],
            "temperature": 0.7,
            "cache_seed": 42
        }
        
        for model_id in models:
            model_path = self.base_path / model_id.replace("/", "_")
            if model_path.exists():
                config["config_list"].append({
                    "model": model_id,
                    "api_type": "local",
                    "model_path": str(model_path),
                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                })
        
        # Save config
        config_path = self.base_path / "autogen_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ AutoGen config saved: {config_path}")
        return config


def main():
    """Main execution"""
    manager = HFModelManager()
    
    # Show recommended models
    print("\n📋 RECOMMENDED FINANCIAL MODELS:")
    print("-" * 60)
    
    for category, models in RECOMMENDED_MODELS.items():
        print(f"\n🏷️  {category.upper()}:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model['name']}")
            print(f"     📝 {model['description']}")
            print(f"     📊 Size: {model['size']}")
    
    # Menu
    while True:
        print("\n" + "="*60)
        print("HUGGING FACE MODEL MANAGER")
        print("="*60)
        print("1. 🔍 Search financial models")
        print("2. 📥 Clone recommended model")
        print("3. 🚀 Setup AutoGen with local models")
        print("4. 📊 Show downloaded models")
        print("5. 🔧 Test model loading")
        print("6. 🚪 Exit")
        
        choice = input("\nChoice (1-6): ").strip()
        
        if choice == "1":
            query = input("Search query [finance]: ").strip() or "finance"
            results = manager.search_models(query)
            print(f"\nFound {len(results)} models:")
            for r in results[:10]:
                print(f"  • {r['id']} (⬇️ {r['downloads']:,} 👍 {r['likes']})")
                
        elif choice == "2":
            print("\nSelect category:")
            categories = list(RECOMMENDED_MODELS.keys())
            for i, cat in enumerate(categories, 1):
                print(f"{i}. {cat}")
            
            cat_idx = int(input("Category: ")) - 1
            if 0 <= cat_idx < len(categories):
                models = RECOMMENDED_MODELS[categories[cat_idx]]
                for i, m in enumerate(models, 1):
                    print(f"{i}. {m['name']}")
                
                model_idx = int(input("Model: ")) - 1
                if 0 <= model_idx < len(models):
                    model_id = models[model_idx]['name']
                    use_lfs = input("Use Git LFS? (y/n) [n]: ").lower() == 'y'
                    manager.clone_model(model_id, use_lfs)
                    
        elif choice == "3":
            # Setup AutoGen
            downloaded = []
            for cat_models in RECOMMENDED_MODELS.values():
                for m in cat_models:
                    path = manager.base_path / m['name'].replace("/", "_")
                    if path.exists():
                        downloaded.append(m['name'])
            
            if downloaded:
                print(f"\n✅ Found {len(downloaded)} models")
                config = manager.setup_autogen_config(downloaded)
                print("\nAutoGen configuration created!")
            else:
                print("\n❌ No models found. Clone some first!")
                
        elif choice == "4":
            print("\n📊 Downloaded models:")
            for path in manager.base_path.iterdir():
                if path.is_dir() and not path.name.startswith('.'):
                    size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                    print(f"  • {path.name} ({size/1024/1024:.1f} MB)")
                    
        elif choice == "5":
            # Test loading
            print("\n🔧 Testing model loading...")
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                test_path = next(manager.base_path.iterdir(), None)
                if test_path and test_path.is_dir():
                    print(f"Testing: {test_path.name}")
                    tokenizer = AutoTokenizer.from_pretrained(str(test_path))
                    print("✅ Tokenizer loaded!")
                    # Don't load full model to save memory
                    print("✅ Model path valid!")
                else:
                    print("❌ No models to test")
            except Exception as e:
                print(f"❌ Error: {e}")
                
        elif choice == "6":
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    # Check for dependencies
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA: {torch.cuda.is_available()}")
    except:
        print("❌ PyTorch not found")
    
    main()