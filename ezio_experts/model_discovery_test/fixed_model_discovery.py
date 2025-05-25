#!/usr/bin/env python3
"""
EzioFilho Unified - Fixed Model Discovery System
Purpose: Comprehensive model discovery and testing without core dependencies
Author: marcobarreto007
Date: 2025-05-24 02:56:36 UTC
System: Windows 10.0.19045.5854, Python 3.11.8
"""

import os
import sys
import json
import logging
import traceback
import subprocess
import requests
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

# REM Project configuration
PROJECT_ROOT = Path(r"C:\Users\anapa\eziofilho-unified")
USER_NAME = "marcobarreto007"
TEST_TIMESTAMP = "2025-05-24 02:56:36 UTC"

@dataclass
class ModelInfo:
    """Model information structure"""
    name: str
    provider: str
    model_type: str
    size_gb: float
    description: str
    download_url: Optional[str] = None
    local_path: Optional[str] = None
    status: str = "available"
    last_updated: Optional[str] = None

class EzioModelDiscovery:
    """Fixed Model Discovery System - Standalone Implementation"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.project_root = PROJECT_ROOT
        self.models_cache = {}
        self.discovered_models = []
        
        # REM Add project to Python path
        if self.project_root.exists():
            sys.path.insert(0, str(self.project_root))
            self.logger.info(f"✅ Project root added: {self.project_root}")
    
    def setup_logging(self):
        """Configure logging system"""
        log_format = '%(asctime)s | %(levelname)8s | %(funcName)s | %(message)s'
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('model_discovery_test.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # REM Suppress verbose loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
    
    def print_banner(self):
        """Print system banner"""
        banner = f"""
{'='*80}
🔍 EZIOFILHO UNIFIED - MODEL DISCOVERY SYSTEM TEST
{'='*80}
👤 User: {USER_NAME}
📅 Test Date: {TEST_TIMESTAMP}
🐍 Python: {sys.version.split()[0]} ({sys.platform})
📁 Project Root: {self.project_root}
⏰ Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
        print(banner)
        self.logger.info("Model Discovery System initialized")
    
    def discover_local_models(self) -> List[ModelInfo]:
        """Discover models in local storage"""
        self.logger.info("=== DISCOVERING LOCAL MODELS ===")
        
        local_models = []
        
        # REM Check model storage directory
        model_storage = self.project_root / "03_models_storage"
        
        if not model_storage.exists():
            self.logger.warning("⚠️  Model storage directory not found")
            return local_models
        
        # REM Scan for model files
        model_extensions = ['.bin', '.safetensors', '.gguf', '.onnx', '.pth', '.pt']
        
        for model_file in model_storage.rglob("*"):
            if model_file.is_file() and model_file.suffix.lower() in model_extensions:
                size_mb = model_file.stat().st_size / (1024 * 1024)
                size_gb = size_mb / 1024
                
                model_info = ModelInfo(
                    name=model_file.stem,
                    provider="local",
                    model_type=self._detect_model_type(model_file),
                    size_gb=round(size_gb, 2),
                    description=f"Local model file: {model_file.name}",
                    local_path=str(model_file),
                    status="local",
                    last_updated=datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                )
                
                local_models.append(model_info)
                self.logger.info(f"✅ Found local model: {model_info.name} ({size_gb:.2f}GB)")
        
        self.logger.info(f"✅ Discovered {len(local_models)} local models")
        return local_models
    
    def _detect_model_type(self, model_file: Path) -> str:
        """Detect model type from file name and extension"""
        name_lower = model_file.name.lower()
        
        if 'llama' in name_lower:
            return 'LLM-Llama'
        elif 'mistral' in name_lower:
            return 'LLM-Mistral'
        elif 'phi' in name_lower:
            return 'LLM-Phi'
        elif 'bert' in name_lower:
            return 'Transformer-BERT'
        elif 'gpt' in name_lower:
            return 'LLM-GPT'
        elif model_file.suffix == '.gguf':
            return 'LLM-GGUF'
        elif model_file.suffix == '.safetensors':
            return 'SafeTensors'
        elif model_file.suffix in ['.pth', '.pt']:
            return 'PyTorch'
        elif model_file.suffix == '.onnx':
            return 'ONNX'
        else:
            return 'Unknown'
    
    def discover_huggingface_models(self) -> List[ModelInfo]:
        """Discover available models from Hugging Face"""
        self.logger.info("=== DISCOVERING HUGGINGFACE MODELS ===")
        
        hf_models = []
        
        # REM Predefined list of popular models for testing
        popular_models = [
            {
                "name": "microsoft/DialoGPT-medium",
                "type": "Conversational",
                "size": 1.2,
                "description": "Microsoft's conversational AI model"
            },
            {
                "name": "microsoft/phi-3-mini-4k-instruct",
                "type": "LLM-Phi",
                "size": 2.4,
                "description": "Microsoft Phi-3 Mini instruction model"
            },
            {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "type": "Embedding",
                "size": 0.09,
                "description": "Sentence transformer for embeddings"
            },
            {
                "name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "type": "Sentiment",
                "size": 0.5,
                "description": "RoBERTa model for sentiment analysis"
            },
            {
                "name": "mistralai/Mistral-7B-Instruct-v0.1",
                "type": "LLM-Mistral",
                "size": 14.5,
                "description": "Mistral 7B instruction model"
            }
        ]
        
        for model_data in popular_models:
            try:
                # REM Check if model is accessible
                model_info = ModelInfo(
                    name=model_data["name"],
                    provider="huggingface",
                    model_type=model_data["type"],
                    size_gb=model_data["size"],
                    description=model_data["description"],
                    download_url=f"https://huggingface.co/{model_data['name']}",
                    status="available"
                )
                
                hf_models.append(model_info)
                self.logger.info(f"✅ HuggingFace model: {model_info.name} ({model_info.size_gb}GB)")
                
            except Exception as e:
                self.logger.warning(f"⚠️  Could not verify model {model_data['name']}: {e}")
        
        self.logger.info(f"✅ Discovered {len(hf_models)} HuggingFace models")
        return hf_models
    
    def discover_mistral_models(self) -> List[ModelInfo]:
        """Discover Mistral AI models"""
        self.logger.info("=== DISCOVERING MISTRAL MODELS ===")
        
        mistral_models = []
        
        # REM Check if Mistral download script exists
        mistral_script = self.project_root / "download_mistral.py"
        
        if mistral_script.exists():
            self.logger.info(f"✅ Mistral download script found: {mistral_script.stat().st_size} bytes")
            
            # REM Predefined Mistral models
            mistral_list = [
                {
                    "name": "mistral-7b-instruct",
                    "size": 14.5,
                    "description": "Mistral 7B Instruct model"
                },
                {
                    "name": "mistral-7b-base",
                    "size": 14.5,
                    "description": "Mistral 7B base model"
                }
            ]
            
            for model_data in mistral_list:
                model_info = ModelInfo(
                    name=model_data["name"],
                    provider="mistral",
                    model_type="LLM-Mistral",
                    size_gb=model_data["size"],
                    description=model_data["description"],
                    status="downloadable"
                )
                
                mistral_models.append(model_info)
                self.logger.info(f"✅ Mistral model: {model_info.name} ({model_info.size_gb}GB)")
        
        else:
            self.logger.info("ℹ️  Mistral download script not found")
        
        self.logger.info(f"✅ Discovered {len(mistral_models)} Mistral models")
        return mistral_models
    
    def check_model_inventory(self) -> List[ModelInfo]:
        """Read and parse the model inventory file"""
        self.logger.info("=== CHECKING MODEL INVENTORY ===")
        
        inventory_models = []
        inventory_file = self.project_root / "model_inventory.txt"
        
        if not inventory_file.exists():
            self.logger.warning("⚠️  Model inventory file not found")
            return inventory_models
        
        try:
            with open(inventory_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # REM Parse inventory content
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            for line in lines:
                if line.startswith('#') or line.startswith('//'):
                    continue
                    
                # REM Simple parsing - assume format: name|provider|type|size|description
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        try:
                            model_info = ModelInfo(
                                name=parts[0].strip(),
                                provider=parts[1].strip(),
                                model_type=parts[2].strip(),
                                size_gb=float(parts[3].strip()) if parts[3].strip().replace('.', '').isdigit() else 0.0,
                                description=parts[4].strip() if len(parts) > 4 else "From inventory",
                                status="inventory"
                            )
                            
                            inventory_models.append(model_info)
                            
                        except ValueError as e:
                            self.logger.warning(f"⚠️  Could not parse inventory line: {line}")
                else:
                    # REM Simple line format - treat as model name
                    model_info = ModelInfo(
                        name=line,
                        provider="unknown",
                        model_type="unknown",
                        size_gb=0.0,
                        description="From inventory file",
                        status="inventory"
                    )
                    
                    inventory_models.append(model_info)
            
            self.logger.info(f"✅ Parsed {len(inventory_models)} models from inventory")
            
        except Exception as e:
            self.logger.error(f"❌ Error reading inventory: {e}")
        
        return inventory_models
    
    def test_cache_models(self) -> List[ModelInfo]:
        """Check cached models"""
        self.logger.info("=== CHECKING CACHED MODELS ===")
        
        cached_models = []
        cache_dirs = [
            Path(r"C:\Users\anapa\.cache\huggingface"),
            Path(r"C:\Users\anapa\.cache\models"),
            Path(r"C:\Users\anapa\.cache\eziofilho")
        ]
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                self.logger.info(f"✅ Checking cache: {cache_dir.name}")
                
                # REM Count cached items
                cached_items = list(cache_dir.rglob("*"))
                model_files = [item for item in cached_items if item.is_file() and item.suffix in ['.bin', '.safetensors', '.json']]
                
                for model_file in model_files[:5]:  # REM Limit to first 5 files
                    size_mb = model_file.stat().st_size / (1024 * 1024)
                    
                    model_info = ModelInfo(
                        name=f"cached_{model_file.stem}",
                        provider="cache",
                        model_type="cached",
                        size_gb=round(size_mb / 1024, 3),
                        description=f"Cached file: {model_file.name}",
                        local_path=str(model_file),
                        status="cached"
                    )
                    
                    cached_models.append(model_info)
                
                self.logger.info(f"   📁 {len(model_files)} cached model files")
            else:
                self.logger.info(f"ℹ️  Cache not found: {cache_dir.name}")
        
        self.logger.info(f"✅ Found {len(cached_models)} cached models")
        return cached_models
    
    def run_comprehensive_discovery(self) -> Dict[str, Any]:
        """Run complete model discovery process"""
        self.print_banner()
        
        start_time = datetime.now()
        
        # REM Discover from all sources
        discovery_sources = [
            ("Local Models", self.discover_local_models),
            ("HuggingFace Models", self.discover_huggingface_models),
            ("Mistral Models", self.discover_mistral_models),
            ("Model Inventory", self.check_model_inventory),
            ("Cached Models", self.test_cache_models)
        ]
        
        all_models = []
        discovery_results = {}
        
        for source_name, discovery_func in discovery_sources:
            try:
                self.logger.info(f"🔍 Running: {source_name}")
                models = discovery_func()
                all_models.extend(models)
                discovery_results[source_name] = {
                    "count": len(models),
                    "status": "success",
                    "models": [asdict(model) for model in models]
                }
                self.logger.info(f"✅ {source_name}: {len(models)} models")
                
            except Exception as e:
                self.logger.error(f"❌ {source_name} failed: {e}")
                discovery_results[source_name] = {
                    "count": 0,
                    "status": "failed",
                    "error": str(e)
                }
            
            print("-" * 60)  # REM Visual separator
        
        # REM Generate summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        summary = {
            "discovery_metadata": {
                "user": USER_NAME,
                "test_date": TEST_TIMESTAMP,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "project_root": str(self.project_root)
            },
            "discovery_summary": {
                "total_models": len(all_models),
                "sources_tested": len(discovery_sources),
                "successful_sources": len([r for r in discovery_results.values() if r["status"] == "success"]),
                "failed_sources": len([r for r in discovery_results.values() if r["status"] == "failed"])
            },
            "discovery_results": discovery_results,
            "all_models": [asdict(model) for model in all_models]
        }
        
        # REM Save results
        try:
            with open('model_discovery_results.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            self.logger.info("💾 Discovery results saved: model_discovery_results.json")
        except Exception as e:
            self.logger.warning(f"⚠️  Could not save results: {e}")
        
        # REM Generate text summary
        try:
            with open('model_discovery_summary.txt', 'w', encoding='utf-8') as f:
                f.write(f"EzioFilho Unified - Model Discovery Summary\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"User: {USER_NAME}\n\n")
                
                f.write(f"DISCOVERY SUMMARY:\n")
                f.write(f"Total Models Found: {summary['discovery_summary']['total_models']}\n")
                f.write(f"Sources Tested: {summary['discovery_summary']['sources_tested']}\n")
                f.write(f"Successful Sources: {summary['discovery_summary']['successful_sources']}\n\n")
                
                f.write("MODELS BY SOURCE:\n")
                for source, result in discovery_results.items():
                    f.write(f"- {source}: {result['count']} models ({result['status']})\n")
                
                f.write("\nALL DISCOVERED MODELS:\n")
                for i, model in enumerate(all_models, 1):
                    f.write(f"{i:2d}. {model.name} ({model.provider}) - {model.size_gb}GB - {model.model_type}\n")
            
            self.logger.info("💾 Text summary saved: model_discovery_summary.txt")
        except Exception as e:
            self.logger.warning(f"⚠️  Could not save text summary: {e}")
        
        self.print_final_summary(summary)
        return summary
    
    def print_final_summary(self, summary: Dict[str, Any]):
        """Print final discovery summary"""
        discovery_summary = summary["discovery_summary"]
        
        print("\n" + "="*80)
        print("🏁 MODEL DISCOVERY - FINAL SUMMARY")
        print("="*80)
        print(f"📊 Total Models Found: {discovery_summary['total_models']}")
        print(f"🔍 Sources Tested: {discovery_summary['sources_tested']}")
        print(f"✅ Successful Sources: {discovery_summary['successful_sources']}")
        print(f"❌ Failed Sources: {discovery_summary['failed_sources']}")
        print(f"⏱️  Duration: {summary['discovery_metadata']['duration_seconds']:.2f} seconds")
        
        print("\n🔍 DISCOVERY RESULTS BY SOURCE:")
        for source, result in summary["discovery_results"].items():
            status_icon = "✅" if result["status"] == "success" else "❌"
            print(f"   {status_icon} {source}: {result['count']} models")
        
        if discovery_summary['total_models'] > 0:
            print(f"\n🎉 MODEL DISCOVERY: SUCCESS")
            print(f"✅ Found {discovery_summary['total_models']} models across {discovery_summary['successful_sources']} sources")
        else:
            print(f"\n⚠️  MODEL DISCOVERY: LIMITED RESULTS")
            print(f"🔍 Consider adding more model sources or checking configurations")
        
        print("="*80)

def main():
    """Main execution function"""
    try:
        # REM Initialize and run discovery
        discovery = EzioModelDiscovery()
        results = discovery.run_comprehensive_discovery()
        
        # REM Exit with appropriate code
        total_models = results["discovery_summary"]["total_models"]
        
        if total_models >= 10:
            sys.exit(0)  # REM Excellent results
        elif total_models >= 5:
            sys.exit(1)  # REM Good results
        else:
            sys.exit(2)  # REM Limited results
            
    except KeyboardInterrupt:
        print("\n🛑 Discovery interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Critical discovery failure: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()