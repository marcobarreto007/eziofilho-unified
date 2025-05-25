# ezio_autogen_local_models.py - AutoGen with Local Models and LangGraph
# Audit Mode: Active - Local models integration
# Path: C:\Users\anapa\EzioFilhoUnified\ezio_experts\autogen_local
# User: marcobarreto007
# Date: 2025-05-24 21:01:03 UTC
# Objective: Integrate AutoGen with local models (Phi-3, LLama) and LangGraph

import os
import sys
import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent paths
sys.path.extend([
    str(Path(__file__).parent.parent.parent),
    str(Path(__file__).parent.parent)
])

# Import AutoGen
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    logger.info("✅ AutoGen loaded")
except ImportError:
    logger.error("❌ AutoGen not found")
    sys.exit(1)

# Import LangGraph
try:
    from langgraph.graph import Graph, StateGraph
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    logger.info("✅ LangGraph loaded")
except ImportError:
    logger.warning("⚠️ LangGraph not found, installing...")
    os.system(f"{sys.executable} -m pip install langgraph langchain-core")
    from langgraph.graph import Graph, StateGraph
    from langgraph.prebuilt import ToolNode

# Import local model support
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_community.llms import HuggingFacePipeline
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    logger.info("✅ Transformers loaded")
except ImportError:
    logger.error("❌ Transformers not found")

@dataclass
class LocalModelConfig:
    """Configuration for local models"""
    name: str
    model_path: str
    model_type: str  # phi3, llama2, mistral, etc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 2048
    temperature: float = 0.7
    
class LocalModelLLM:
    """Custom LLM wrapper for local models to work with AutoGen"""
    
    def __init__(self, config: LocalModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._load_model()
        
    def _load_model(self):
        """Load local model"""
        logger.info(f"Loading local model: {self.config.name}")
        
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            # Try common paths
            possible_paths = [
                Path("C:/Users/anapa/models") / self.config.name,
                Path("C:/Users/anapa/EzioFilhoUnified/models") / self.config.name,
                Path("C:/Users/anapa/eziofilho-unified/models") / self.config.name,
                Path("C:/Users/anapa/.cache/huggingface/hub") / self.config.name
            ]
            
            for path in possible_paths:
                if path.exists():
                    model_path = path
                    break
            else:
                logger.error(f"Model not found in any path: {self.config.name}")
                return
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map=self.config.device,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                device=0 if self.config.device == "cuda" else -1
            )
            
            logger.info(f"✅ Model loaded: {self.config.name} on {self.config.device}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response compatible with AutoGen"""
        if not self.pipeline:
            return "Error: Model not loaded"
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Generate response
        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", self.config.temperature),
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1
            )
            
            response = outputs[0]["generated_text"]
            # Extract only the new generated text
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating response: {str(e)}"
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert AutoGen messages to prompt format"""
        prompt = ""
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if self.config.model_type == "phi3":
                if role == "system":
                    prompt += f"<|system|>\n{content}<|end|>\n"
                elif role == "user":
                    prompt += f"<|user|>\n{content}<|end|>\n"
                elif role == "assistant":
                    prompt += f"<|assistant|>\n{content}<|end|>\n"
                    
            elif self.config.model_type in ["llama2", "mistral"]:
                if role == "system":
                    prompt += f"<<SYS>>\n{content}\n<</SYS>>\n\n"
                elif role == "user":
                    prompt += f"[INST] {content} [/INST]\n"
                elif role == "assistant":
                    prompt += f"{content}\n"
                    
            else:
                # Generic format
                prompt += f"{role}: {content}\n"
        
        # Add assistant prompt
        if self.config.model_type == "phi3":
            prompt += "<|assistant|>\n"
        
        return prompt

class EzioAutoGenLocal:
    """EzioFilho AutoGen with Local Models and LangGraph"""
    
    def __init__(self):
        self.models = {}
        self.agents = {}
        self.graphs = {}
        self._load_local_models()
        self._initialize_agents()
        self._create_langgraph_workflows()
        
    def _load_local_models(self):
        """Load available local models"""
        logger.info("🔍 Searching for local models...")
        
        # Define model configurations
        model_configs = [
            LocalModelConfig(
                name="microsoft/Phi-3-mini-4k-instruct",
                model_path="C:/Users/anapa/models/phi3-mini",
                model_type="phi3"
            ),
            LocalModelConfig(
                name="meta-llama/Llama-2-7b-chat-hf",
                model_path="C:/Users/anapa/models/llama2-7b",
                model_type="llama2"
            ),
            LocalModelConfig(
                name="mistralai/Mistral-7B-Instruct-v0.2",
                model_path="C:/Users/anapa/models/mistral-7b",
                model_type="mistral"
            )
        ]
        
        # Load available models
        for config in model_configs:
            try:
                model = LocalModelLLM(config)
                if model.pipeline:
                    self.models[config.name] = model
                    logger.info(f"✅ Loaded: {config.name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load {config.name}: {e}")
        
        if not self.models:
            logger.error("❌ No local models found!")
            # Create mock model for demo
            self.models["mock"] = self._create_mock_model()
    
    def _create_mock_model(self):
        """Create mock model for testing"""
        class MockModel:
            def generate(self, messages, **kwargs):
                last_message = messages[-1]["content"] if messages else ""
                return f"Mock response to: {last_message[:50]}..."
        
        return MockModel()
    
    def _get_llm_config_for_local_model(self, model_name: str):
        """Create LLM config for AutoGen with local model"""
        model = self.models.get(model_name, list(self.models.values())[0])
        
        # Create custom model function for AutoGen
        def custom_model_func(messages, **kwargs):
            return model.generate(messages, **kwargs)
        
        return {
            "config_list": [{
                "model": model_name,
                "api_type": "custom",
                "custom_llm_provider": "local",
                "custom_model_func": custom_model_func
            }],
            "temperature": 0.7,
            "max_tokens": 1024,
            "seed": 42
        }
    
    def _initialize_agents(self):
        """Initialize AutoGen agents with local models"""
        logger.info("🤖 Initializing agents with local models...")
        
        # Get first available model
        default_model = list(self.models.keys())[0] if self.models else "mock"
        
        # 1. EzioMaster with local model
        self.agents["ezio_master"] = AssistantAgent(
            name="EzioMaster",
            system_message="""You are EzioFilho, the master financial AI assistant.
            Coordinate analysis and provide clear insights.
            Focus on: accuracy, risk management, actionable advice.""",
            llm_config=self._get_llm_config_for_local_model(default_model),
            max_consecutive_auto_reply=5
        )
        
        # 2. Market Analyst
        self.agents["market_analyst"] = AssistantAgent(
            name="MarketAnalyst",
            system_message="""You are a market analyst expert.
            Analyze: technical indicators, trends, support/resistance.
            Provide: specific levels, timeframes, probabilities.""",
            llm_config=self._get_llm_config_for_local_model(default_model)
        )
        
        # 3. Risk Manager
        self.agents["risk_manager"] = AssistantAgent(
            name="RiskManager",
            system_message="""You are a risk management expert.
            Focus on: position sizing, stop losses, portfolio protection.
            Always prioritize capital preservation.""",
            llm_config=self._get_llm_config_for_local_model(default_model)
        )
        
        # 4. Data Analyst
        self.agents["data_analyst"] = AssistantAgent(
            name="DataAnalyst",
            system_message="""You are a financial data analyst.
            Process: price data, volume, market metrics.
            Identify: patterns, anomalies, opportunities.""",
            llm_config=self._get_llm_config_for_local_model(default_model)
        )
        
        # 5. User Proxy (no LLM needed)
        self.agents["user_proxy"] = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            code_execution_config=False  # Disable for security
        )
        
        logger.info(f"✅ Initialized {len(self.agents)} agents")
    
    def _create_langgraph_workflows(self):
        """Create LangGraph workflows for complex analysis"""
        logger.info("🔄 Creating LangGraph workflows...")
        
        # 1. Market Analysis Workflow
        def create_market_analysis_graph():
            def analyze_state(state: Dict[str, Any]) -> Dict[str, Any]:
                """Market analysis state"""
                return state
            
            def collect_data(state: Dict[str, Any]) -> Dict[str, Any]:
                """Collect market data"""
                state["data"] = {
                    "price": "Current price data",
                    "volume": "Volume analysis",
                    "indicators": "Technical indicators"
                }
                return state
            
            def analyze_technicals(state: Dict[str, Any]) -> Dict[str, Any]:
                """Technical analysis"""
                state["technical_analysis"] = "RSI, MACD, Bollinger Bands analysis"
                return state
            
            def assess_risk(state: Dict[str, Any]) -> Dict[str, Any]:
                """Risk assessment"""
                state["risk_assessment"] = "Risk levels and recommendations"
                return state
            
            def generate_signal(state: Dict[str, Any]) -> Dict[str, Any]:
                """Generate trading signal"""
                state["signal"] = {
                    "action": "BUY/SELL/HOLD",
                    "confidence": "High/Medium/Low",
                    "reasons": ["Reason 1", "Reason 2"]
                }
                return state
            
            # Create graph
            graph = StateGraph(dict)
            
            # Add nodes
            graph.add_node("collect_data", collect_data)
            graph.add_node("analyze_technicals", analyze_technicals)
            graph.add_node("assess_risk", assess_risk)
            graph.add_node("generate_signal", generate_signal)
            
            # Add edges
            graph.add_edge("collect_data", "analyze_technicals")
            graph.add_edge("analyze_technicals", "assess_risk")
            graph.add_edge("assess_risk", "generate_signal")
            
            # Set entry point
            graph.set_entry_point("collect_data")
            
            return graph.compile()
        
        # 2. Portfolio Optimization Workflow
        def create_portfolio_graph():
            def analyze_portfolio(state: Dict[str, Any]) -> Dict[str, Any]:
                """Analyze current portfolio"""
                state["portfolio_analysis"] = "Current allocation and performance"
                return state
            
            def calculate_risk_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
                """Calculate risk metrics"""
                state["risk_metrics"] = {
                    "var": "Value at Risk",
                    "sharpe": "Sharpe Ratio",
                    "max_drawdown": "Maximum Drawdown"
                }
                return state
            
            def optimize_allocation(state: Dict[str, Any]) -> Dict[str, Any]:
                """Optimize portfolio allocation"""
                state["optimized_allocation"] = "New recommended allocation"
                return state
            
            # Create graph
            graph = StateGraph(dict)
            
            graph.add_node("analyze", analyze_portfolio)
            graph.add_node("risk", calculate_risk_metrics)
            graph.add_node("optimize", optimize_allocation)
            
            graph.add_edge("analyze", "risk")
            graph.add_edge("risk", "optimize")
            
            graph.set_entry_point("analyze")
            
            return graph.compile()
        
        # Store compiled graphs
        self.graphs["market_analysis"] = create_market_analysis_graph()
        self.graphs["portfolio_optimization"] = create_portfolio_graph()
        
        logger.info(f"✅ Created {len(self.graphs)} LangGraph workflows")
    
    def analyze_with_autogen(self, query: str) -> Dict[str, Any]:
        """Analyze using AutoGen multi-agent system"""
        logger.info(f"🔍 AutoGen Analysis: {query}")
        
        # Create group chat
        agents = [
            self.agents["ezio_master"],
            self.agents["market_analyst"],
            self.agents["risk_manager"],
            self.agents["data_analyst"],
            self.agents["user_proxy"]
        ]
        
        group_chat = GroupChat(
            agents=agents,
            messages=[],
            max_round=10
        )
        
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=self._get_llm_config_for_local_model(list(self.models.keys())[0])
        )
        
        # Start analysis
        self.agents["user_proxy"].initiate_chat(
            manager,
            message=f"Analyze: {query}"
        )
        
        # Extract results
        results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "messages": group_chat.messages,
            "analysis": self._extract_insights(group_chat.messages)
        }
        
        return results
    
    def analyze_with_langgraph(self, workflow: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze using LangGraph workflow"""
        logger.info(f"🔄 LangGraph Analysis: {workflow}")
        
        if workflow not in self.graphs:
            return {"error": f"Workflow '{workflow}' not found"}
        
        graph = self.graphs[workflow]
        
        # Run workflow
        try:
            result = graph.invoke(input_data)
            return {
                "workflow": workflow,
                "timestamp": datetime.now().isoformat(),
                "input": input_data,
                "output": result
            }
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            return {"error": str(e)}
    
    def _extract_insights(self, messages: List[Dict]) -> Dict[str, List[str]]:
        """Extract insights from agent messages"""
        insights = {
            "key_points": [],
            "recommendations": [],
            "risks": [],
            "data": []
        }
        
        for msg in messages:
            if msg.get("name") and msg.get("content"):
                content = msg["content"]
                agent = msg["name"]
                
                # Simple extraction (enhance with NLP)
                if "recommend" in content.lower():
                    insights["recommendations"].append(f"{agent}: {content[:100]}...")
                if "risk" in content.lower():
                    insights["risks"].append(f"{agent}: {content[:100]}...")
                    
        return insights


def main():
    """Main demonstration"""
    print("="*80)
    print("🚀 EZIOFILHO AUTOGEN WITH LOCAL MODELS + LANGGRAPH")
    print("="*80)
    
    # Initialize system
    system = EzioAutoGenLocal()
    
    # Show loaded models
    print(f"\n📦 Loaded Models: {list(system.models.keys())}")
    print(f"🤖 Active Agents: {list(system.agents.keys())}")
    print(f"🔄 Workflows: {list(system.graphs.keys())}")
    
    # Demo menu
    while True:
        print("\n" + "-"*50)
        print("Choose demo:")
        print("1. AutoGen Multi-Agent Analysis")
        print("2. LangGraph Market Analysis")
        print("3. LangGraph Portfolio Optimization")
        print("4. Show System Info")
        print("5. Exit")
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == "1":
            query = input("Enter analysis query: ").strip()
            if query:
                result = system.analyze_with_autogen(query)
                print(f"\n📊 Analysis Results:")
                print(f"Query: {result['query']}")
                print(f"Messages: {len(result['messages'])}")
                print(f"Key Insights: {result['analysis']}")
                
        elif choice == "2":
            asset = input("Enter asset (e.g., BTC, AAPL): ").strip()
            result = system.analyze_with_langgraph(
                "market_analysis",
                {"asset": asset, "timeframe": "1D"}
            )
            print(f"\n📊 Market Analysis: {result}")
            
        elif choice == "3":
            result = system.analyze_with_langgraph(
                "portfolio_optimization",
                {"portfolio": ["BTC", "ETH", "AAPL", "GOOGL"]}
            )
            print(f"\n💼 Portfolio Optimization: {result}")
            
        elif choice == "4":
            print(f"\n🔧 System Information:")
            print(f"Models: {system.models}")
            print(f"Agents: {system.agents}")
            print(f"Graphs: {system.graphs}")
            
        elif choice == "5":
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()