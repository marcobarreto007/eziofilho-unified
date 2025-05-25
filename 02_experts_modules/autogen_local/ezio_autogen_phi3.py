# ezio_autogen_phi3.py - AutoGen Multi-Agent System with Local PHI-3
# Audit Mode: Using local models only, no API keys required
# Path: C:\Users\anapa\eziofilho-unified\02_experts_modules\autogen_local
# User: marcobarreto007
# Date: 2025-05-24 16:47:44 UTC

import os
import sys
from pathlib import Path
import autogen
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class LocalLLMWrapper:
    """Wrapper for local LLM models to work with AutoGen"""
    
    def __init__(self, model_name: str = "microsoft/phi-3-mini-4k-instruct"):
        print(f"🤖 Loading local model: {model_name}")
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📍 Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        print("✅ Model loaded successfully!")
        
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from messages"""
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Generate response
        outputs = self.pipeline(prompt)
        response = outputs[0]['generated_text']
        
        # Extract only the new response
        response = response[len(prompt):].strip()
        
        return response
        
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt format"""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
                
        prompt += "Assistant: "
        return prompt

class EzioAutoGenLocalSystem:
    """Multi-Agent Financial System using Local Models"""
    
    def __init__(self):
        print("=" * 80)
        print("🤖 EZIOFILHO AUTOGEN SYSTEM - LOCAL PHI-3")
        print("🔧 No API keys required - 100% local execution")
        print("=" * 80)
        
        # Initialize local LLM
        self.llm = LocalLLMWrapper()
        
        # Configuration for AutoGen with local model
        self.config_list = [{
            "model": "local-phi3",
            "api_type": "custom",
            "base_url": "local",
            "api_key": "not-needed"
        }]
        
        # Custom LLM config for local execution
        self.llm_config = {
            "config_list": self.config_list,
            "temperature": 0.7,
            "functions": [
                {
                    "name": "generate_response",
                    "description": "Generate response using local PHI-3 model",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "messages": {
                                "type": "array",
                                "description": "Chat messages"
                            }
                        }
                    }
                }
            ]
        }
        
        # Initialize agents
        self.initialize_agents()
        
        print("✅ Local Multi-Agent System Ready!")
        print("=" * 80)
        
    def initialize_agents(self):
        """Initialize financial expert agents"""
        
        # Custom agent class that uses local LLM
        class LocalAssistantAgent(autogen.AssistantAgent):
            def __init__(self, name, system_message, llm_wrapper, **kwargs):
                super().__init__(name=name, system_message=system_message, **kwargs)
                self.llm_wrapper = llm_wrapper
                
            def generate_reply(self, messages=None, sender=None, config=None):
                """Generate reply using local model"""
                if messages is None:
                    messages = self._oai_messages[sender]
                
                # Convert to format for local LLM
                formatted_messages = [
                    {"role": "system", "content": self.system_message}
                ]
                
                for msg in messages:
                    if isinstance(msg, dict):
                        formatted_messages.append({
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", "")
                        })
                        
                # Generate response
                response = self.llm_wrapper.generate(formatted_messages)
                
                return True, response
        
        # 1. Chief Financial Analyst
        self.chief_analyst = LocalAssistantAgent(
            name="ChiefAnalyst",
            system_message="""You are the Chief Financial Analyst.
            Coordinate financial analysis and provide insights on:
            - Market trends and opportunities
            - Investment strategies
            - Risk assessment
            - Portfolio recommendations
            Always be clear, concise, and data-driven.""",
            llm_wrapper=self.llm,
            llm_config=self.llm_config
        )
        
        # 2. Crypto Expert
        self.crypto_expert = LocalAssistantAgent(
            name="CryptoExpert",
            system_message="""You are a Cryptocurrency Expert.
            Provide analysis on:
            - Bitcoin, Ethereum, and altcoins
            - DeFi protocols
            - Market trends
            - Technical analysis
            Focus on actionable insights.""",
            llm_wrapper=self.llm,
            llm_config=self.llm_config
        )
        
        # 3. Risk Manager
        self.risk_manager = LocalAssistantAgent(
            name="RiskManager",
            system_message="""You are a Risk Management Expert.
            Analyze:
            - Investment risks
            - Portfolio diversification
            - Market volatility
            - Risk mitigation strategies
            Provide clear risk assessments.""",
            llm_wrapper=self.llm,
            llm_config=self.llm_config
        )
        
        # 4. Technical Analyst
        self.technical_analyst = LocalAssistantAgent(
            name="TechnicalAnalyst",
            system_message="""You are a Technical Analysis Expert.
            Focus on:
            - Chart patterns
            - Technical indicators
            - Support/resistance levels
            - Entry/exit points
            Be specific with price levels.""",
            llm_wrapper=self.llm,
            llm_config=self.llm_config
        )
        
        # User proxy
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
        )
        
    def analyze_financial_query(self, query: str) -> Dict[str, Any]:
        """Analyze query using local multi-agent system"""
        
        print(f"\n🔍 Analyzing: {query}")
        print("-" * 60)
        
        results = {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": {}
        }
        
        # Get analysis from each agent
        agents = [
            (self.chief_analyst, "Overall Analysis"),
            (self.crypto_expert, "Crypto Perspective"),
            (self.risk_manager, "Risk Assessment"),
            (self.technical_analyst, "Technical View")
        ]
        
        for agent, description in agents:
            print(f"\n[{agent.name}] {description}:")
            
            # Generate response
            success, response = agent.generate_reply(
                messages=[{"role": "user", "content": query}]
            )
            
            if success:
                print(response[:200] + "..." if len(response) > 200 else response)
                results["analysis"][agent.name] = response
            else:
                print("❌ Failed to generate response")
                
        return results
        
    def interactive_session(self):
        """Run interactive session with local models"""
        
        print("\n💬 Local Multi-Agent Financial Analysis")
        print("🔧 Using PHI-3 - No internet required!")
        print("Type 'exit' to quit")
        print("-" * 60)
        
        while True:
            user_input = input("\n📝 Your question: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'sair']:
                print("\n👋 Thank you for using EzioFilho!")
                break
                
            if user_input:
                analysis = self.analyze_financial_query(user_input)
                
                print("\n" + "=" * 60)
                print("📊 MULTI-AGENT ANALYSIS")
                print("=" * 60)
                
                for agent_name, response in analysis["analysis"].items():
                    print(f"\n[{agent_name}]")
                    print(response[:300] + "..." if len(response) > 300 else response)

# Utility functions for local model management

class LocalModelManager:
    """Manage local models"""
    
    @staticmethod
    def list_available_models() -> List[str]:
        """List available local models"""
        models = []
        
        # Check common model locations
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if cache_dir.exists():
            for model_dir in cache_dir.iterdir():
                if model_dir.is_dir() and "model" in model_dir.name.lower():
                    models.append(model_dir.name)
                    
        return models
        
    @staticmethod
    def download_model(model_name: str):
        """Download a model for local use"""
        print(f"📥 Downloading {model_name}...")
        
        from transformers import AutoModel, AutoTokenizer
        
        # Download model and tokenizer
        AutoTokenizer.from_pretrained(model_name)
        AutoModel.from_pretrained(model_name)
        
        print("✅ Model downloaded successfully!")
        
    @staticmethod
    def get_model_info(model_path: str) -> Dict[str, Any]:
        """Get information about a local model"""
        config_path = Path(model_path) / "config.json"
        
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        return {}

# Tool functions for agents

def get_crypto_price_local(symbol: str) -> str:
    """Get crypto price (simulated for local testing)"""
    # In production, this would fetch real data
    prices = {
        "bitcoin": "$108,000",
        "ethereum": "$3,800",
        "bnb": "$720"
    }
    return prices.get(symbol.lower(), "Price not available")

def calculate_risk_score(volatility: float, drawdown: float) -> str:
    """Calculate risk score"""
    score = (volatility * 0.6 + drawdown * 0.4) * 10
    
    if score < 3:
        return f"Low Risk (Score: {score:.1f}/10)"
    elif score < 7:
        return f"Medium Risk (Score: {score:.1f}/10)"
    else:
        return f"High Risk (Score: {score:.1f}/10)"

# Main execution
if __name__ == "__main__":
    try:
        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"🎮 GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("💻 Running on CPU (slower but works!)")
            
        # List available models
        print("\n📦 Checking for available models...")
        manager = LocalModelManager()
        models = manager.list_available_models()
        
        if models:
            print(f"Found {len(models)} local models")
        else:
            print("No local models found. Will download PHI-3...")
            
        # Create and run system
        system = EzioAutoGenLocalSystem()
        
        # Run interactive session
        system.interactive_session()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Tip: Make sure you have enough RAM/VRAM for the model")
        import traceback
        traceback.print_exc()