# ezio_autogen_system.py - Advanced Multi-Agent Financial System
# Audit Mode: AutoGen integration for collaborative AI agents
# Path: C:\Users\anapa\eziofilho-unified\02_experts_modules\autogen_agents
# User: marcobarreto007
# Date: 2025-05-24 16:41:01 UTC

import os
import sys
from pathlib import Path
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import core components
from config.api_config import API_KEYS
from core.multilingual_responses import FACTUAL_ANSWERS

class EzioAutoGenSystem:
    """Advanced Multi-Agent Financial System using AutoGen"""
    
    def __init__(self):
        print("=" * 80)
        print("🤖 EZIOFILHO AUTOGEN MULTI-AGENT SYSTEM")
        print("🔄 Initializing collaborative AI agents...")
        print("=" * 80)
        
        # Configuration for AutoGen
        self.config_list = [
            {
                "model": "gpt-4",
                "api_key": API_KEYS.get("openai", ""),
            }
        ]
        
        # LLM configuration
        self.llm_config = {
            "seed": 42,
            "config_list": self.config_list,
            "temperature": 0.7,
        }
        
        # Initialize all specialized agents
        self.initialize_agents()
        
        print("✅ Multi-Agent System Ready!")
        print("=" * 80)
        
    def initialize_agents(self):
        """Initialize all specialized financial agents"""
        
        # 1. Chief Financial Analyst (Coordinator)
        self.chief_analyst = AssistantAgent(
            name="ChiefAnalyst",
            system_message="""You are the Chief Financial Analyst coordinating a team of specialists.
            Your role is to:
            1. Understand user queries about financial markets
            2. Delegate tasks to appropriate specialists
            3. Synthesize insights from multiple agents
            4. Provide comprehensive financial advice
            
            You work with:
            - CryptoExpert: Cryptocurrency analysis
            - StockAnalyst: Stock market analysis
            - RiskManager: Risk assessment
            - NewsAnalyst: Market news and sentiment
            - TechnicalAnalyst: Charts and technical indicators
            - FundamentalAnalyst: Company fundamentals
            - MacroEconomist: Economic indicators
            - QuantTrader: Algorithmic strategies
            
            Always provide analysis in the user's language (PT/EN/FR).""",
            llm_config=self.llm_config,
        )
        
        # 2. Cryptocurrency Expert
        self.crypto_expert = AssistantAgent(
            name="CryptoExpert",
            system_message="""You are a Cryptocurrency Expert specializing in:
            - Bitcoin, Ethereum, and altcoin analysis
            - DeFi protocols and yield farming
            - NFT market trends
            - Blockchain technology insights
            - Crypto market sentiment
            - Technical analysis for crypto
            
            Provide detailed crypto analysis with:
            - Current prices and 24h changes
            - Market cap and volume analysis
            - Support/resistance levels
            - On-chain metrics
            - Risk factors specific to crypto""",
            llm_config=self.llm_config,
        )
        
        # 3. Stock Market Analyst
        self.stock_analyst = AssistantAgent(
            name="StockAnalyst",
            system_message="""You are a Stock Market Analyst specializing in:
            - Individual stock analysis (US, Brazil, Global)
            - Sector rotation strategies
            - Earnings analysis and forecasts
            - Dividend strategies
            - Growth vs Value investing
            
            Provide analysis including:
            - P/E ratios and valuation metrics
            - Revenue and earnings growth
            - Competitive positioning
            - Industry trends
            - Buy/Hold/Sell recommendations""",
            llm_config=self.llm_config,
        )
        
        # 4. Risk Manager
        self.risk_manager = AssistantAgent(
            name="RiskManager",
            system_message="""You are a Risk Management Expert focusing on:
            - Portfolio risk assessment
            - Value at Risk (VaR) calculations
            - Diversification strategies
            - Hedging recommendations
            - Risk/reward ratios
            
            Analyze and report on:
            - Volatility metrics
            - Correlation analysis
            - Maximum drawdown scenarios
            - Black swan events
            - Risk mitigation strategies""",
            llm_config=self.llm_config,
        )
        
        # 5. News and Sentiment Analyst
        self.news_analyst = AssistantAgent(
            name="NewsAnalyst",
            system_message="""You are a Financial News and Sentiment Analyst monitoring:
            - Breaking market news
            - Social media sentiment
            - Regulatory changes
            - Geopolitical events
            - Market moving announcements
            
            Provide:
            - News impact analysis
            - Sentiment scores
            - Trend identification
            - Event-driven opportunities
            - Risk alerts from news""",
            llm_config=self.llm_config,
        )
        
        # 6. Technical Analyst
        self.technical_analyst = AssistantAgent(
            name="TechnicalAnalyst",
            system_message="""You are a Technical Analysis Expert specializing in:
            - Chart patterns and formations
            - Technical indicators (RSI, MACD, Bollinger Bands)
            - Support and resistance levels
            - Trend analysis
            - Volume analysis
            
            Provide:
            - Entry and exit points
            - Stop-loss recommendations
            - Chart pattern recognition
            - Momentum indicators
            - Price targets""",
            llm_config=self.llm_config,
        )
        
        # 7. Fundamental Analyst
        self.fundamental_analyst = AssistantAgent(
            name="FundamentalAnalyst",
            system_message="""You are a Fundamental Analysis Expert focusing on:
            - Financial statement analysis
            - Business model evaluation
            - Competitive advantages
            - Management quality
            - Industry analysis
            
            Analyze:
            - Balance sheets and income statements
            - Cash flow analysis
            - Return on equity/assets
            - Debt levels and coverage
            - Growth prospects""",
            llm_config=self.llm_config,
        )
        
        # 8. Macro Economist
        self.macro_economist = AssistantAgent(
            name="MacroEconomist",
            system_message="""You are a Macroeconomic Expert analyzing:
            - Global economic indicators
            - Interest rate trends
            - Inflation analysis
            - Currency movements
            - Economic cycles
            
            Provide insights on:
            - GDP growth forecasts
            - Central bank policies
            - Trade dynamics
            - Recession probabilities
            - Sector implications""",
            llm_config=self.llm_config,
        )
        
        # 9. Quantitative Trader
        self.quant_trader = AssistantAgent(
            name="QuantTrader",
            system_message="""You are a Quantitative Trading Expert specializing in:
            - Algorithmic trading strategies
            - Statistical arbitrage
            - Machine learning models
            - Backtesting results
            - High-frequency trading insights
            
            Develop and analyze:
            - Trading algorithms
            - Risk-adjusted returns
            - Sharpe ratios
            - Alpha generation
            - Portfolio optimization""",
            llm_config=self.llm_config,
        )
        
        # 10. User Proxy (for human interaction)
        self.user_proxy = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
        )
        
        # Create Group Chat with all agents
        self.agents = [
            self.chief_analyst,
            self.crypto_expert,
            self.stock_analyst,
            self.risk_manager,
            self.news_analyst,
            self.technical_analyst,
            self.fundamental_analyst,
            self.macro_economist,
            self.quant_trader,
            self.user_proxy
        ]
        
        self.group_chat = GroupChat(
            agents=self.agents,
            messages=[],
            max_round=10,
            speaker_selection_method="auto",
        )
        
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
        )
        
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze user query with multi-agent collaboration"""
        
        print(f"\n🔍 Analyzing: {query}")
        print("-" * 60)
        
        # Initiate the conversation
        self.user_proxy.initiate_chat(
            self.manager,
            message=query,
        )
        
        # Collect insights from all agents
        insights = {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "agents_involved": [],
            "analysis": {},
            "recommendations": [],
            "risks": [],
            "opportunities": []
        }
        
        # Process group chat messages
        for message in self.group_chat.messages:
            agent_name = message.get("name", "Unknown")
            content = message.get("content", "")
            
            if agent_name != "User":
                insights["agents_involved"].append(agent_name)
                insights["analysis"][agent_name] = content
                
        return insights
        
    def get_comprehensive_analysis(self, asset: str, asset_type: str = "crypto") -> Dict[str, Any]:
        """Get comprehensive analysis for an asset"""
        
        if asset_type == "crypto":
            query = f"""Provide a comprehensive analysis of {asset} cryptocurrency including:
            1. Current price and market trends
            2. Technical analysis with key levels
            3. Fundamental analysis and use cases
            4. Risk assessment
            5. News and sentiment
            6. Trading recommendations
            7. Long-term outlook"""
        else:
            query = f"""Provide a comprehensive analysis of {asset} stock including:
            1. Current price and valuation
            2. Technical analysis
            3. Fundamental analysis
            4. Risk factors
            5. Industry position
            6. Investment recommendation
            7. Price targets"""
            
        return self.analyze_query(query)
        
    def get_market_outlook(self) -> Dict[str, Any]:
        """Get overall market outlook from all agents"""
        
        query = """Provide a comprehensive market outlook covering:
        1. Cryptocurrency market trends
        2. Stock market conditions
        3. Economic indicators
        4. Major risks and opportunities
        5. Sector recommendations
        6. Trading strategies for current conditions"""
        
        return self.analyze_query(query)
        
    def run_interactive_session(self):
        """Run interactive multi-agent session"""
        
        print("\n💬 Multi-Agent Financial Analysis System")
        print("Type 'exit' to quit, 'market' for outlook, or ask any question")
        print("-" * 60)
        
        while True:
            user_input = input("\n📝 Your question: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'sair']:
                print("\n👋 Thank you for using EzioFilho AutoGen System!")
                break
                
            if user_input.lower() == 'market':
                analysis = self.get_market_outlook()
            else:
                analysis = self.analyze_query(user_input)
                
            # Display results
            print("\n" + "=" * 60)
            print("📊 MULTI-AGENT ANALYSIS RESULTS")
            print("=" * 60)
            
            print(f"\n🤝 Agents Involved: {', '.join(analysis['agents_involved'])}")
            
            for agent, content in analysis['analysis'].items():
                if agent != "User":
                    print(f"\n[{agent}]")
                    print(content[:500] + "..." if len(content) > 500 else content)
                    
            print("\n" + "=" * 60)

# Additional AutoGen Tools and Utilities

class FinancialTools:
    """Tools for AutoGen agents to use"""
    
    @staticmethod
    def get_crypto_price(symbol: str) -> Dict[str, float]:
        """Get cryptocurrency price"""
        import requests
        
        try:
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": symbol.lower(),
                "vs_currencies": "usd,brl",
                "include_24hr_change": "true"
            }
            
            response = requests.get(url, params=params, timeout=5)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
            
    @staticmethod
    def calculate_risk_metrics(returns: List[float]) -> Dict[str, float]:
        """Calculate risk metrics"""
        import numpy as np
        
        returns_array = np.array(returns)
        
        return {
            "volatility": np.std(returns_array) * np.sqrt(252),
            "sharpe_ratio": np.mean(returns_array) / np.std(returns_array) * np.sqrt(252),
            "max_drawdown": np.min(np.minimum.accumulate(returns_array) - returns_array),
            "var_95": np.percentile(returns_array, 5)
        }

# LangChain Integration

class EzioLangChainAgents:
    """LangChain integration for advanced reasoning"""
    
    def __init__(self):
        from langchain.agents import initialize_agent, Tool
        from langchain.agents import AgentType
        from langchain.llms import OpenAI
        from langchain.memory import ConversationBufferMemory
        
        # Initialize LLM
        self.llm = OpenAI(
            api_key=API_KEYS.get("openai", ""),
            temperature=0.7
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Define tools
        self.tools = [
            Tool(
                name="CryptoPrice",
                func=FinancialTools.get_crypto_price,
                description="Get current cryptocurrency prices"
            ),
            Tool(
                name="RiskCalculator",
                func=FinancialTools.calculate_risk_metrics,
                description="Calculate risk metrics for investments"
            )
        ]
        
        # Initialize agent
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
        
    def analyze(self, query: str) -> str:
        """Analyze query using LangChain agent"""
        return self.agent.run(query)

# Main execution
if __name__ == "__main__":
    try:
        # Check if OpenAI API key is available
        if not API_KEYS.get("openai"):
            print("⚠️  Warning: OpenAI API key not found!")
            print("AutoGen requires OpenAI API for full functionality")
            print("\nRunning in demo mode...")
            
        # Create and run the system
        system = EzioAutoGenSystem()
        
        # Run interactive session
        system.run_interactive_session()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()