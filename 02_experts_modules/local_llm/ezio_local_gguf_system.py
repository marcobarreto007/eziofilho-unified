# ezio_local_gguf_system.py - Local GGUF Model System
# Audit Mode: Using GGUF models with llama.cpp
# Path: C:\Users\anapa\eziofilho-unified\02_experts_modules\local_llm
# User: marcobarreto007
# Date: 2025-05-24 16:55:36 UTC

import os
import sys
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent.parent))

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("⚠️  llama-cpp-python not installed")
    print("Install with: pip install llama-cpp-python")

class EzioLocalGGUFSystem:
    """Financial AI System using local GGUF models"""
    
    def __init__(self, model_path: Optional[str] = None):
        print("=" * 80)
        print("🤖 EZIOFILHO LOCAL GGUF SYSTEM")
        print("🔧 100% Local - No Internet Required")
        print("=" * 80)
        
        # Find model if not specified
        if not model_path:
            model_path = self.find_gguf_model()
            
        if not model_path:
            print("❌ No GGUF model found!")
            print("Please download a model (e.g., phi-3.gguf)")
            return
            
        self.model_path = model_path
        print(f"📁 Using model: {Path(model_path).name}")
        
        # Initialize model
        self.initialize_model()
        
    def find_gguf_model(self) -> Optional[str]:
        """Find first available GGUF model"""
        search_paths = [
            Path("C:/Users/anapa/eziofilho-unified/models"),
            Path("C:/Users/anapa/eziofilho-unified/03_models_storage"),
            Path(".")
        ]
        
        for path in search_paths:
            if path.exists():
                for gguf in path.glob("*.gguf"):
                    return str(gguf)
                    
        return None
        
    def initialize_model(self):
        """Initialize GGUF model with llama.cpp"""
        if not LLAMA_CPP_AVAILABLE:
            print("❌ Cannot load model without llama-cpp-python")
            return
            
        try:
            print("🔄 Loading model...")
            
            # Initialize with GPU support if available
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,  # Context window
                n_threads=8,  # CPU threads
                n_gpu_layers=35,  # GPU layers (if CUDA available)
                verbose=False
            )
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.llm = None
            
    def create_agents(self):
        """Create financial expert agents"""
        self.agents = {
            "crypto_expert": {
                "name": "CryptoExpert",
                "prompt": """You are a cryptocurrency expert. Analyze crypto markets, 
                provide insights on Bitcoin, Ethereum, DeFi, and trading strategies.
                Be concise and data-driven."""
            },
            "stock_analyst": {
                "name": "StockAnalyst", 
                "prompt": """You are a stock market analyst. Provide analysis on stocks,
                market trends, valuations, and investment opportunities.
                Focus on actionable insights."""
            },
            "risk_manager": {
                "name": "RiskManager",
                "prompt": """You are a risk management expert. Assess investment risks,
                portfolio diversification, and provide risk mitigation strategies.
                Be clear about risk levels."""
            }
        }
        
    def query_agent(self, agent_name: str, query: str) -> str:
        """Query specific agent"""
        if not self.llm:
            return "Model not loaded"
            
        agent = self.agents.get(agent_name)
        if not agent:
            return "Agent not found"
            
        # Build prompt
        prompt = f"{agent['prompt']}\n\nUser: {query}\nAssistant:"
        
        # Generate response
        try:
            response = self.llm(
                prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.95,
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            return f"Error: {str(e)}"
            
    def analyze_financial_query(self, query: str) -> Dict[str, Any]:
        """Analyze query with multiple agents"""
        print(f"\n🔍 Analyzing: {query}")
        print("-" * 60)
        
        results = {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": {}
        }
        
        # Get analysis from each agent
        for agent_name in self.agents:
            print(f"\n[{self.agents[agent_name]['name']}]")
            response = self.query_agent(agent_name, query)
            print(response[:200] + "..." if len(response) > 200 else response)
            results["analysis"][agent_name] = response
            
        return results
        
    def interactive_session(self):
        """Run interactive chat"""
        if not self.llm:
            print("❌ Cannot start session without model")
            return
            
        print("\n💬 Local Financial Assistant")
        print("Type 'exit' to quit")
        print("-" * 60)
        
        # Create agents
        self.create_agents()
        
        while True:
            user_input = input("\n📝 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'sair']:
                print("\n👋 Goodbye!")
                break
                
            if user_input:
                # Simple mode - direct response
                response = self.llm(
                    f"You are a helpful financial assistant. User: {user_input}\nAssistant:",
                    max_tokens=512,
                    temperature=0.7,
                    echo=False
                )
                
                print(f"\n🤖 Assistant: {response['choices'][0]['text'].strip()}")

# Alternative: Use transformers if no GGUF available

class EzioTransformersSystem:
    """Fallback system using transformers"""
    
    def __init__(self):
        print("=" * 80)
        print("🤖 EZIOFILHO TRANSFORMERS SYSTEM")
        print("📦 Using HuggingFace models")
        print("=" * 80)
        
        self.load_model()
        
    def load_model(self):
        """Load a small model that works on CPU"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            # Use a small model
            model_name = "microsoft/DialoGPT-small"
            print(f"📥 Loading {model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=200
            )
            
            print("✅ Model loaded!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.pipeline = None
            
    def chat(self, query: str) -> str:
        """Simple chat function"""
        if not self.pipeline:
            return "Model not loaded"
            
        try:
            # Generate response
            outputs = self.pipeline(
                query,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = outputs[0]['generated_text']
            # Extract only new text
            response = response[len(query):].strip()
            
            return response
            
        except Exception as e:
            return f"Error: {str(e)}"
            
    def run(self):
        """Run chat interface"""
        print("\n💬 Chat Interface")
        print("Type 'exit' to quit")
        print("-" * 60)
        
        while True:
            user_input = input("\n📝 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                break
                
            if user_input:
                response = self.chat(user_input)
                print(f"\n🤖 Assistant: {response}")

# Main execution
if __name__ == "__main__":
    print("🔍 Checking available options...")
    
    # Option 1: Try GGUF model
    if LLAMA_CPP_AVAILABLE:
        system = EzioLocalGGUFSystem()
        if system.llm:
            system.interactive_session()
        else:
            print("\n💡 Switching to transformers...")
            fallback = EzioTransformersSystem()
            fallback.run()
    else:
        # Option 2: Use transformers
        print("\n📦 Using transformers (no GGUF support)")
        system = EzioTransformersSystem()
        system.run()