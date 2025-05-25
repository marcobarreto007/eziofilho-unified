#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced AutoConfiguration System for Financial AI Models
========================================================
Version: 2.1.0 - Optimized and Fixed
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try imports with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - CPU mode only")

# Constants
DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 2048

# Expert Types
class ExpertType(Enum):
    DATA_ANALYST = "data_analyst"
    MARKET_PREDICTOR = "market_predictor"
    RISK_ASSESSOR = "risk_assessor"
    PORTFOLIO_MANAGER = "portfolio_manager"
    TREND_ANALYST = "trend_analyst"
    NEWS_INTERPRETER = "news_interpreter"
    REPORT_GENERATOR = "report_generator"
    STRATEGY_ADVISOR = "strategy_advisor"

@dataclass
class ModelConfig:
    """Model configuration data class"""
    name: str
    path: str
    model_type: str = "unknown"
    capabilities: List[str] = field(default_factory=list)
    context_length: int = DEFAULT_CONTEXT_LENGTH
    size_mb: float = 1000.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "model_type": self.model_type,
            "capabilities": self.capabilities,
            "context_length": self.context_length,
            "size_mb": self.size_mb,
            "parameters": self.parameters,
            "device": self.device
        }

@dataclass
class ExpertConfig:
    """Expert configuration data class"""
    name: str
    expert_type: str
    description: str
    model: str
    fallback_model: Optional[str] = None
    context_length: int = DEFAULT_CONTEXT_LENGTH
    parameters: Dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    enabled: bool = True
    priority: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "expert_type": self.expert_type,
            "description": self.description,
            "model": self.model,
            "fallback_model": self.fallback_model,
            "context_length": self.context_length,
            "parameters": self.parameters,
            "system_prompt": self.system_prompt,
            "enabled": self.enabled,
            "priority": self.priority
        }

# Expert Requirements
EXPERT_REQUIREMENTS = {
    ExpertType.DATA_ANALYST: {
        "min_context": 4096,
        "capabilities": ["precise", "math"],
        "temperature": 0.4,
        "description": "Financial data analyst specializing in pattern recognition"
    },
    ExpertType.MARKET_PREDICTOR: {
        "min_context": 8192,
        "capabilities": ["precise", "finance"],
        "temperature": 0.5,
        "description": "Market forecaster for trend prediction"
    },
    ExpertType.RISK_ASSESSOR: {
        "min_context": 4096,
        "capabilities": ["precise", "math"],
        "temperature": 0.3,
        "description": "Risk analysis and assessment specialist"
    },
    ExpertType.PORTFOLIO_MANAGER: {
        "min_context": 8192,
        "capabilities": ["precise", "math", "finance"],
        "temperature": 0.4,
        "description": "Portfolio optimization expert"
    },
    ExpertType.TREND_ANALYST: {
        "min_context": 8192,
        "capabilities": ["precise", "finance"],
        "temperature": 0.5,
        "description": "Long-term market trend analyst"
    },
    ExpertType.NEWS_INTERPRETER: {
        "min_context": 4096,
        "capabilities": ["general"],
        "temperature": 0.6,
        "description": "Financial news analysis expert"
    },
    ExpertType.REPORT_GENERATOR: {
        "min_context": 4096,
        "capabilities": ["creative"],
        "temperature": 0.7,
        "description": "Financial report writer"
    },
    ExpertType.STRATEGY_ADVISOR: {
        "min_context": 8192,
        "capabilities": ["creative", "finance"],
        "temperature": 0.6,
        "description": "Strategic financial advisor"
    }
}

class GPUManager:
    """Simple GPU management"""
    def __init__(self):
        self.available_gpus = self._detect_gpus()
        self.assignments = {}
        self._lock = threading.Lock()
    
    def _detect_gpus(self) -> List[int]:
        """Detect available GPUs"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return []
        
        gpu_count = torch.cuda.device_count()
        gpus = []
        for i in range(gpu_count):
            try:
                torch.cuda.get_device_properties(i)
                gpus.append(i)
                logger.info(f"Found GPU {i}: {torch.cuda.get_device_name(i)}")
            except:
                pass
        return gpus
    
    def get_device(self, model_name: str, size_mb: float) -> str:
        """Get best device for model"""
        if not self.available_gpus:
            return "cpu"
        
        # Simple round-robin assignment
        with self._lock:
            if model_name in self.assignments:
                return f"cuda:{self.assignments[model_name]}"
            
            # Find GPU with least assignments
            gpu_counts = {gpu: 0 for gpu in self.available_gpus}
            for assigned_gpu in self.assignments.values():
                if assigned_gpu in gpu_counts:
                    gpu_counts[assigned_gpu] += 1
            
            best_gpu = min(gpu_counts, key=gpu_counts.get)
            self.assignments[model_name] = best_gpu
            return f"cuda:{best_gpu}"

class MockModelDiscovery:
    """Mock model discovery for testing"""
    def __init__(self):
        self.models = self._create_mock_models()
    
    def _create_mock_models(self) -> List[ModelConfig]:
        """Create mock models"""
        return [
            ModelConfig(
                name="llama2-7b-finance",
                path="./models/llama2-7b-finance",
                model_type="llama",
                capabilities=["precise", "fast", "finance"],
                context_length=4096,
                size_mb=7000.0
            ),
            ModelConfig(
                name="mistral-7b-instruct",
                path="./models/mistral-7b-instruct", 
                model_type="mistral",
                capabilities=["general", "fast", "creative"],
                context_length=8192,
                size_mb=7200.0
            ),
            ModelConfig(
                name="phi-2",
                path="./models/phi-2",
                model_type="phi",
                capabilities=["precise", "fast"],
                context_length=2048,
                size_mb=2700.0
            )
        ]
    
    def scan_all(self):
        """Mock scan method"""
        pass

class AutoConfiguration:
    """Simplified auto-configuration system"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = Path(config_dir or "./config")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.gpu_manager = GPUManager()
        self.discovery = MockModelDiscovery()
        
        self.model_configs = []
        self.expert_configs = []
        
        logger.info("AutoConfiguration initialized")
    
    def discover_models(self) -> List[ModelConfig]:
        """Discover available models"""
        try:
            self.discovery.scan_all()
            models = self.discovery.models
            logger.info(f"Discovered {len(models)} models")
            return models
        except Exception as e:
            logger.error(f"Model discovery failed: {e}")
            return []
    
    def generate_model_configs(self) -> List[Dict[str, Any]]:
        """Generate model configurations"""
        models = self.discover_models()
        if not models:
            logger.warning("No models found")
            return []
        
        configs = []
        for model in models:
            # Get optimal device
            device = self.gpu_manager.get_device(model.name, model.size_mb)
            model.device = device
            
            # Set default parameters
            if not model.parameters:
                model.parameters = {
                    "temperature": DEFAULT_TEMPERATURE,
                    "top_p": DEFAULT_TOP_P,
                    "max_tokens": DEFAULT_MAX_TOKENS,
                    "do_sample": True
                }
            
            configs.append(model.to_dict())
            logger.debug(f"Configured model {model.name} on {device}")
        
        self.model_configs = configs
        logger.info(f"Generated {len(configs)} model configurations")
        return configs
    
    def generate_expert_configs(self) -> List[Dict[str, Any]]:
        """Generate expert configurations"""
        if not self.model_configs:
            self.generate_model_configs()
        
        if not self.model_configs:
            logger.error("No models available for experts")
            return []
        
        configs = []
        
        for expert_type in ExpertType:
            expert_config = self._create_expert_config(expert_type)
            if expert_config:
                configs.append(expert_config.to_dict())
        
        self.expert_configs = configs
        logger.info(f"Generated {len(configs)} expert configurations")
        return configs
    
    def _create_expert_config(self, expert_type: ExpertType) -> Optional[ExpertConfig]:
        """Create configuration for specific expert"""
        requirements = EXPERT_REQUIREMENTS.get(expert_type, {})
        
        # Find best model
        best_model = None
        best_score = -1
        
        for model in self.model_configs:
            score = self._score_model_for_expert(model, requirements)
            if score > best_score:
                best_score = score
                best_model = model
        
        if not best_model:
            logger.warning(f"No suitable model for {expert_type.value}")
            return None
        
        # Create expert config
        config = ExpertConfig(
            name=f"{expert_type.value}_expert",
            expert_type=expert_type.value,
            description=requirements.get("description", ""),
            model=best_model["name"],
            context_length=best_model["context_length"],
            parameters={
                "temperature": requirements.get("temperature", DEFAULT_TEMPERATURE),
                "top_p": DEFAULT_TOP_P,
                "max_tokens": DEFAULT_MAX_TOKENS
            },
            system_prompt=self._generate_system_prompt(expert_type),
            priority=self._get_expert_priority(expert_type)
        )
        
        return config
    
    def _score_model_for_expert(self, model: Dict, requirements: Dict) -> float:
        """Score model for expert requirements"""
        score = 0.0
        
        # Check context length
        min_context = requirements.get("min_context", DEFAULT_CONTEXT_LENGTH)
        if model["context_length"] >= min_context:
            score += 20
        else:
            return -1  # Disqualify
        
        # Check capabilities
        required_caps = set(requirements.get("capabilities", []))
        model_caps = set(model.get("capabilities", []))
        
        if required_caps.issubset(model_caps):
            score += 30
        else:
            missing = required_caps - model_caps
            score -= len(missing) * 10
        
        # Prefer smaller models for efficiency
        size_gb = model["size_mb"] / 1024
        if size_gb < 3:
            score += 20
        elif size_gb < 8:
            score += 10
        
        return score
    
    def _generate_system_prompt(self, expert_type: ExpertType) -> str:
        """Generate system prompt for expert"""
        prompts = {
            ExpertType.DATA_ANALYST: "You are a financial data analyst. Analyze data and provide insights.",
            ExpertType.MARKET_PREDICTOR: "You are a market predictor. Forecast market trends accurately.",
            ExpertType.RISK_ASSESSOR: "You are a risk assessor. Identify and evaluate financial risks.",
            ExpertType.PORTFOLIO_MANAGER: "You are a portfolio manager. Optimize investment portfolios.",
            ExpertType.TREND_ANALYST: "You are a trend analyst. Identify long-term market trends.",
            ExpertType.NEWS_INTERPRETER: "You are a news interpreter. Analyze financial news impact.",
            ExpertType.REPORT_GENERATOR: "You are a report generator. Create clear financial reports.",
            ExpertType.STRATEGY_ADVISOR: "You are a strategy advisor. Develop financial strategies."
        }
        return prompts.get(expert_type, "You are a financial expert.")
    
    def _get_expert_priority(self, expert_type: ExpertType) -> int:
        """Get expert priority"""
        priorities = {
            ExpertType.RISK_ASSESSOR: 10,
            ExpertType.PORTFOLIO_MANAGER: 9,
            ExpertType.MARKET_PREDICTOR: 8,
            ExpertType.DATA_ANALYST: 7,
            ExpertType.STRATEGY_ADVISOR: 6,
            ExpertType.TREND_ANALYST: 5,
            ExpertType.NEWS_INTERPRETER: 4,
            ExpertType.REPORT_GENERATOR: 3
        }
        return priorities.get(expert_type, 5)
    
    def save_configuration(self, filename: Optional[str] = None) -> str:
        """Save configuration to file"""
        if not filename:
            filename = f"config_{int(time.time())}.json"
        
        config_data = {
            "version": "2.1.0",
            "generated_at": time.time(),
            "models": self.model_configs,
            "experts": self.expert_configs,
            "gpu_info": {
                "available_gpus": len(self.gpu_manager.available_gpus),
                "assignments": self.gpu_manager.assignments
            }
        }
        
        config_path = self.config_dir / filename
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            return str(config_path)
        
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def load_configuration(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self.model_configs = config_data.get("models", [])
            self.expert_configs = config_data.get("experts", [])
            
            logger.info(f"Loaded configuration from {config_path}")
            return config_data
        
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate current configuration"""
        issues = []
        
        # Check models
        if not self.model_configs:
            issues.append("No model configurations found")
        
        # Check experts  
        if not self.expert_configs:
            issues.append("No expert configurations found")
        
        # Check model references
        model_names = {m["name"] for m in self.model_configs}
        for expert in self.expert_configs:
            if expert["model"] not in model_names:
                issues.append(f"Expert {expert['name']} references unknown model {expert['model']}")
        
        return len(issues) == 0, issues

# Convenience functions
def create_auto_configuration() -> AutoConfiguration:
    """Create AutoConfiguration instance"""
    return AutoConfiguration()

def generate_default_config() -> str:
    """Generate default configuration"""
    config = create_auto_configuration()
    config.generate_model_configs()
    config.generate_expert_configs()
    return config.save_configuration()

# Main execution
if __name__ == "__main__":
    import sys
    
    print("AutoConfiguration System v2.1.0")
    print("-" * 40)
    
    try:
        # Create configuration
        config = create_auto_configuration()
        
        # Generate configurations
        models = config.generate_model_configs()
        print(f"Generated {len(models)} model configurations")
        
        experts = config.generate_expert_configs()
        print(f"Generated {len(experts)} expert configurations")
        
        # Validate
        is_valid, issues = config.validate_configuration()
        if is_valid:
            print("✓ Configuration is valid")
        else:
            print("✗ Configuration has issues:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Save
        config_file = config.save_configuration()
        print(f"✓ Saved to {config_file}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)