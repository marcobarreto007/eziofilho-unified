# ezio_autogen_complete.py - Complete AutoGen Multi-Agent System
# Audit Mode: Active - AutoGen implementation
# Path: C:\Users\anapa\eziofilho-unified\autogen_system
# User: marcobarreto007
# Date: 2025-05-24 20:54:50 UTC
# Objective: Create complete AutoGen multi-agent financial system

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

print("=" * 80)
print("🚀 EZIOFILHO AUTOGEN MULTI-AGENT SYSTEM v3.0")
print("=" * 80)

# First, let's install AutoGen if not installed
print("\n📦 Checking AutoGen installation...")
try:
    import autogen
    print("✅ AutoGen is installed")
except ImportError:
    print("⚠️ AutoGen not found. Installing...")
    os.system(f"{sys.executable} -m pip install pyautogen")
    import autogen

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

class EzioAutoGenSystem:
    """Complete EzioFilho AutoGen Multi-Agent Financial System"""
    
    def __init__(self):
        """Initialize the AutoGen system"""
        self.agents = {}
        self.config = self._load_config()
        self.llm_config = self._get_llm_config()
        self._initialize_agents()
        self.group_chat = None
        self.manager = None
        print("\n✅ EzioFilho AutoGen System Initialized!")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration"""
        config = {
            "system_name": "EzioFilho AutoGen",
            "version": "3.0",
            "language": "multilingual",
            "max_rounds": 20,
            "agents": {
                "ezio_master": {
                    "role": "Master Coordinator",
                    "capabilities": ["orchestration", "analysis", "decision"]
                },
                "market_analyst": {
                    "role": "Market Analysis Expert",
                    "capabilities": ["technical_analysis", "fundamental_analysis", "trends"]
                },
                "risk_manager": {
                    "role": "Risk Management Expert",
                    "capabilities": ["risk_assessment", "portfolio_protection", "position_sizing"]
                },
                "data_collector": {
                    "role": "Data Collection Specialist",
                    "capabilities": ["real_time_data", "historical_data", "news_sentiment"]
                },
                "strategy_expert": {
                    "role": "Trading Strategy Expert",
                    "capabilities": ["algo_trading", "backtesting", "optimization"]
                },
                "crypto_specialist": {
                    "role": "Cryptocurrency Expert",
                    "capabilities": ["defi", "onchain_analysis", "crypto_trends"]
                },
                "macro_economist": {
                    "role": "Macroeconomic Analyst",
                    "capabilities": ["economic_indicators", "policy_analysis", "global_trends"]
                },
                "quant_analyst": {
                    "role": "Quantitative Analyst",
                    "capabilities": ["mathematical_models", "statistics", "ml_models"]
                }
            }
        }
        return config
    
    def _get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration for agents"""
        # Try to get API key from environment or use placeholder
        api_key = os.getenv("OPENAI_API_KEY", "sk-your-openai-api-key")
        
        return {
            "timeout": 600,
            "seed": 42,
            "config_list": [
                {
                    "model": "gpt-4",
                    "api_key": api_key
                },
                {
                    "model": "gpt-3.5-turbo",
                    "api_key": api_key
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
    
    def _initialize_agents(self):
        """Initialize all AutoGen agents"""
        print("\n🤖 Initializing AutoGen Agents...")
        
        # 1. EzioMaster - The Main Coordinator
        self.agents["ezio_master"] = AssistantAgent(
            name="EzioMaster",
            system_message="""You are EzioFilho, the master financial AI coordinator.
            
            Your responsibilities:
            - Coordinate all expert agents to provide comprehensive financial analysis
            - Synthesize insights from multiple experts
            - Ensure accuracy and actionable recommendations
            - Communicate in Portuguese, English, or French based on user preference
            
            Always:
            - Be professional and helpful
            - Provide clear, structured responses
            - Include specific numbers and data
            - Give actionable recommendations
            - Consider risk management in all advice""",
            llm_config=self.llm_config,
            max_consecutive_auto_reply=10
        )
        
        # 2. Market Analyst
        self.agents["market_analyst"] = AssistantAgent(
            name="MarketAnalyst",
            system_message="""You are a senior market analyst with 20+ years experience.
            
            Your expertise:
            - Technical Analysis: RSI, MACD, Bollinger Bands, Fibonacci, Elliott Waves
            - Fundamental Analysis: P/E ratios, earnings, revenue, growth metrics
            - Market Sentiment: Fear & Greed Index, put/call ratios, VIX
            - Chart Patterns: Head & Shoulders, triangles, flags, wedges
            - Support/Resistance levels and trend analysis
            
            Always provide:
            - Specific price targets with timeframes
            - Technical indicator values
            - Key support and resistance levels
            - Probability assessments
            - Risk/reward ratios""",
            llm_config=self.llm_config
        )
        
        # 3. Risk Manager
        self.agents["risk_manager"] = AssistantAgent(
            name="RiskManager",
            system_message="""You are a chief risk officer specializing in portfolio protection.
            
            Your expertise:
            - Risk Assessment: VaR, CVaR, stress testing, scenario analysis
            - Position Sizing: Kelly Criterion, fixed fractional, volatility-based
            - Stop Loss Strategy: ATR-based, percentage-based, support levels
            - Portfolio Diversification: correlation analysis, sector allocation
            - Risk/Reward Optimization: Sharpe ratio, Sortino ratio
            
            Always prioritize:
            - Capital preservation over returns
            - Maximum drawdown limits (usually 10-20%)
            - Risk/reward ratios of at least 1:2
            - Proper position sizing (max 2-5% per trade)
            - Hedge strategies when appropriate""",
            llm_config=self.llm_config
        )
        
        # 4. Data Collector
        self.agents["data_collector"] = AssistantAgent(
            name="DataCollector",
            system_message="""You are a financial data specialist with API expertise.
            
            Your capabilities:
            - Real-time price data from multiple exchanges
            - Historical data analysis and pattern recognition
            - News sentiment analysis and impact assessment
            - Economic calendar and indicator tracking
            - Volume analysis and liquidity assessment
            - Social media sentiment tracking
            
            Always ensure:
            - Data accuracy and timeliness
            - Multiple source verification
            - Anomaly detection and filtering
            - Proper data normalization
            - Clear data visualization suggestions""",
            llm_config=self.llm_config
        )
        
        # 5. Strategy Expert
        self.agents["strategy_expert"] = AssistantAgent(
            name="StrategyExpert",
            system_message="""You are a quantitative trading strategist and algorithm developer.
            
            Your expertise:
            - Trading Strategies: momentum, mean reversion, arbitrage, pairs trading
            - Entry/Exit Optimization: multiple timeframe analysis, confluence zones
            - Backtesting: walk-forward analysis, Monte Carlo simulation
            - Algorithm Development: signal generation, execution logic
            - Market Microstructure: order flow, market making, liquidity provision
            
            Focus on:
            - Strategies with positive Sharpe ratios (>1.5)
            - Win rates above 40% with good risk/reward
            - Robust strategies that work in different market conditions
            - Clear entry, exit, and position sizing rules
            - Performance metrics and drawdown analysis""",
            llm_config=self.llm_config
        )
        
        # 6. Crypto Specialist
        self.agents["crypto_specialist"] = AssistantAgent(
            name="CryptoSpecialist",
            system_message="""You are a cryptocurrency and DeFi expert.
            
            Your expertise:
            - Bitcoin & Altcoin Analysis: dominance, cycles, halvings
            - DeFi Protocols: yield farming, liquidity provision, lending/borrowing
            - On-chain Metrics: NUPL, SOPR, exchange flows, whale activity
            - NFTs and Gaming: market trends, valuations, opportunities
            - Layer 1/2 Analysis: Ethereum, Solana, Avalanche, Polygon
            
            Always consider:
            - Regulatory risks and compliance
            - Smart contract risks
            - Liquidity and slippage
            - Gas fees and transaction costs
            - Cross-chain opportunities and bridges""",
            llm_config=self.llm_config
        )
        
        # 7. Macro Economist
        self.agents["macro_economist"] = AssistantAgent(
            name="MacroEconomist",
            system_message="""You are a chief economist specializing in global macro analysis.
            
            Your expertise:
            - Economic Indicators: GDP, inflation, employment, PMI
            - Central Bank Policy: Fed, ECB, BoJ, interest rates
            - Currency Analysis: DXY, major pairs, emerging markets
            - Commodity Markets: oil, gold, agricultural products
            - Geopolitical Risk: trade wars, sanctions, political events
            
            Provide insights on:
            - Economic cycles and turning points
            - Policy impacts on markets
            - Sector rotation strategies
            - Currency hedging strategies
            - Long-term structural trends""",
            llm_config=self.llm_config
        )
        
        # 8. Quantitative Analyst
        self.agents["quant_analyst"] = AssistantAgent(
            name="QuantAnalyst",
            system_message="""You are a quantitative analyst with PhD in financial mathematics.
            
            Your expertise:
            - Mathematical Models: Black-Scholes, GARCH, stochastic calculus
            - Statistical Analysis: regression, time series, hypothesis testing
            - Machine Learning: prediction models, clustering, anomaly detection
            - Portfolio Optimization: Markowitz, Black-Litterman, risk parity
            - Derivatives Pricing: options, futures, structured products
            
            Always provide:
            - Mathematical rigor and proofs
            - Statistical significance tests
            - Model assumptions and limitations
            - Confidence intervals
            - Out-of-sample validation results""",
            llm_config=self.llm_config
        )
        
        # 9. User Proxy (for code execution if needed)
        self.agents["user_proxy"] = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={
                "work_dir": "workspace",
                "use_docker": False
            },
            is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0
        )
        
        print(f"✅ Initialized {len(self.agents)} agents successfully!")
        
    def create_group_chat(self, agents: Optional[List[str]] = None, max_round: int = 20):
        """Create a group chat with specified agents"""
        if agents is None:
            # Default to all agents except user_proxy
            agents = [name for name in self.agents.keys() if name != "user_proxy"]
        
        # Get agent objects
        agent_list = [self.agents[name] for name in agents if name in self.agents]
        
        # Add user proxy at the end
        agent_list.append(self.agents["user_proxy"])
        
        # Create group chat
        self.group_chat = GroupChat(
            agents=agent_list,
            messages=[],
            max_round=max_round
        )
        
        # Create manager
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config
        )
        
        return self.group_chat, self.manager
    
    def analyze_market(self, query: str) -> Dict[str, Any]:
        """Perform comprehensive market analysis"""
        print(f"\n🔍 Analyzing: {query}")
        
        # Select relevant agents based on query
        agents = ["ezio_master", "market_analyst", "risk_manager", "data_collector"]
        
        # Add specialized agents based on keywords
        query_lower = query.lower()
        if any(word in query_lower for word in ["bitcoin", "btc", "ethereum", "crypto", "defi"]):
            agents.append("crypto_specialist")
        if any(word in query_lower for word in ["strategy", "trade", "algoritmo"]):
            agents.append("strategy_expert")
        if any(word in query_lower for word in ["economy", "inflation", "fed", "interest"]):
            agents.append("macro_economist")
        if any(word in query_lower for word in ["model", "statistical", "quantitative"]):
            agents.append("quant_analyst")
        
        # Create group chat
        self.create_group_chat(agents)
        
        # Formulate analysis request
        analysis_request = f"""
        Analyze the following query and provide comprehensive insights:
        
        Query: {query}
        
        Please provide:
        1. Current market analysis
        2. Key data points and metrics
        3. Risk assessment
        4. Recommended actions
        5. Time horizons and price targets
        
        Each expert should contribute their specialized insights.
        """
        
        # Initiate chat
        self.agents["user_proxy"].initiate_chat(
            self.manager,
            message=analysis_request
        )
        
        # Extract and structure results
        results = self._extract_analysis_results()
        
        return results
    
    def _extract_analysis_results(self) -> Dict[str, Any]:
        """Extract structured results from group chat"""
        messages = self.group_chat.messages
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "analysis": {},
            "recommendations": [],
            "risks": [],
            "data_points": {}
        }
        
        # Extract insights from each agent
        for msg in messages:
            if msg.get("name") and msg.get("content"):
                agent_name = msg["name"]
                content = msg["content"]
                
                # Store agent-specific analysis
                if agent_name not in results["analysis"]:
                    results["analysis"][agent_name] = []
                
                results["analysis"][agent_name].append({
                    "content": content[:500] + "..." if len(content) > 500 else content,
                    "full_content": content
                })
        
        return results
    
    def get_trading_signal(self, asset: str, timeframe: str = "1D") -> Dict[str, Any]:
        """Generate trading signal for specific asset"""
        print(f"\n📊 Generating trading signal for {asset} ({timeframe})")
        
        # Select relevant agents
        agents = ["ezio_master", "market_analyst", "risk_manager", "strategy_expert"]
        if "crypto" in asset.lower() or any(coin in asset.upper() for coin in ["BTC", "ETH", "BNB"]):
            agents.append("crypto_specialist")
        
        # Create group chat
        self.create_group_chat(agents, max_round=15)
        
        # Signal request
        signal_request = f"""
        Generate a detailed trading signal for {asset} on {timeframe} timeframe.
        
        Required information:
        1. Direction: BUY/SELL/HOLD
        2. Entry Price: Specific level or range
        3. Stop Loss: Specific level (with reasoning)
        4. Take Profit: Multiple targets (TP1, TP2, TP3)
        5. Position Size: Percentage of portfolio
        6. Confidence Level: High/Medium/Low with percentage
        7. Key Reasons: Technical and fundamental factors
        8. Risk/Reward Ratio: Calculated ratio
        9. Time Horizon: Expected duration
        10. Alternative Scenarios: What could go wrong
        
        Provide consensus from all experts.
        """
        
        # Initiate chat
        self.agents["user_proxy"].initiate_chat(
            self.manager,
            message=signal_request
        )
        
        # Parse signal
        signal = self._parse_trading_signal(asset, timeframe)
        
        return signal
    
    def _parse_trading_signal(self, asset: str, timeframe: str) -> Dict[str, Any]:
        """Parse trading signal from agent discussions"""
        messages = self.group_chat.messages
        
        signal = {
            "asset": asset,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "direction": "HOLD",
            "entry_price": None,
            "stop_loss": None,
            "take_profit": [],
            "position_size": 0,
            "confidence": "Low",
            "confidence_percentage": 0,
            "risk_reward_ratio": 0,
            "reasons": {
                "technical": [],
                "fundamental": [],
                "sentiment": []
            },
            "risks": [],
            "time_horizon": "Not specified",
            "expert_consensus": {}
        }
        
        # Extract signal components from messages
        for msg in messages:
            if msg.get("content"):
                content = msg["content"].lower()
                # Parse direction
                if "buy" in content and "signal" in content:
                    signal["direction"] = "BUY"
                elif "sell" in content and "signal" in content:
                    signal["direction"] = "SELL"
                
                # Extract other components (simplified parsing)
                # In production, use more sophisticated NLP parsing
                
        return signal
    
    def portfolio_analysis(self, portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze complete portfolio"""
        print("\n💼 Analyzing portfolio...")
        
        # Select all relevant agents
        agents = ["ezio_master", "risk_manager", "market_analyst", 
                 "strategy_expert", "macro_economist", "quant_analyst"]
        
        # Create group chat
        self.create_group_chat(agents)
        
        # Portfolio request
        portfolio_str = "\n".join([f"- {item['asset']}: {item['quantity']} units at ${item.get('avg_price', 'N/A')}" 
                                  for item in portfolio])
        
        analysis_request = f"""
        Analyze the following portfolio and provide comprehensive recommendations:
        
        Portfolio:
        {portfolio_str}
        
        Please provide:
        1. Overall portfolio health score (0-100)
        2. Risk assessment and VaR calculation
        3. Diversification analysis
        4. Rebalancing recommendations
        5. Hedging strategies
        6. Expected returns (1M, 3M, 1Y)
        7. Correlation analysis
        8. Optimization suggestions
        
        Consider current market conditions and provide actionable advice.
        """
        
        # Initiate chat
        self.agents["user_proxy"].initiate_chat(
            self.manager,
            message=analysis_request
        )
        
        # Extract results
        analysis = self._extract_portfolio_analysis()
        
        return analysis
    
    def _extract_portfolio_analysis(self) -> Dict[str, Any]:
        """Extract portfolio analysis results"""
        messages = self.group_chat.messages
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "health_score": 0,
            "risk_metrics": {},
            "recommendations": [],
            "expected_returns": {},
            "optimization": {},
            "expert_insights": {}
        }
        
        # Parse messages for analysis components
        for msg in messages:
            if msg.get("name") and msg.get("content"):
                agent_name = msg["name"]
                analysis["expert_insights"][agent_name] = msg["content"]
        
        return analysis
    
    def economic_outlook(self, region: str = "Global") -> Dict[str, Any]:
        """Generate economic outlook report"""
        print(f"\n🌍 Generating economic outlook for {region}...")
        
        # Select macro-focused agents
        agents = ["ezio_master", "macro_economist", "market_analyst", 
                 "risk_manager", "quant_analyst"]
        
        # Create group chat
        self.create_group_chat(agents)
        
        # Outlook request
        outlook_request = f"""
        Provide comprehensive economic outlook for {region}:
        
        1. GDP growth projections
        2. Inflation expectations
        3. Central bank policy outlook
        4. Currency trends
        5. Key risks and opportunities
        6. Sector recommendations
        7. Asset allocation suggestions
        8. Time horizon: 3M, 6M, 1Y, 5Y
        
        Include specific numbers and probability assessments.
        """
        
        # Initiate chat
        self.agents["user_proxy"].initiate_chat(
            self.manager,
            message=outlook_request
        )
        
        # Extract outlook
        outlook = self._extract_economic_outlook(region)
        
        return outlook
    
    def _extract_economic_outlook(self, region: str) -> Dict[str, Any]:
        """Extract economic outlook from discussions"""
        messages = self.group_chat.messages
        
        outlook = {
            "region": region,
            "timestamp": datetime.now().isoformat(),
            "gdp_forecast": {},
            "inflation_forecast": {},
            "policy_outlook": {},
            "currency_forecast": {},
            "risks": [],
            "opportunities": [],
            "sector_recommendations": {},
            "expert_views": {}
        }
        
        # Extract insights from messages
        for msg in messages:
            if msg.get("name") and msg.get("content"):
                outlook["expert_views"][msg["name"]] = msg["content"]
        
        return outlook


class EzioAutoGenCLI:
    """Command-line interface for EzioFilho AutoGen System"""
    
    def __init__(self):
        self.system = EzioAutoGenSystem()
        self.running = True
        
    def display_menu(self):
        """Display main menu"""
        print("\n" + "="*60)
        print("🤖 EZIOFILHO AUTOGEN SYSTEM - MAIN MENU")
        print("="*60)
        print("1. 📊 Market Analysis")
        print("2. 📈 Trading Signal")
        print("3. 💼 Portfolio Analysis")
        print("4. 🌍 Economic Outlook")
        print("5. 🔧 Configure Agents")
        print("6. 📚 Help")
        print("7. 🚪 Exit")
        print("="*60)
        
    def run(self):
        """Run the CLI interface"""
        print("\n🎯 Welcome to EzioFilho AutoGen System!")
        print("💡 Tip: Make sure to set your OPENAI_API_KEY environment variable")
        
        while self.running:
            self.display_menu()
            choice = input("\n👉 Select option (1-7): ").strip()
            
            if choice == "1":
                self.market_analysis()
            elif choice == "2":
                self.trading_signal()
            elif choice == "3":
                self.portfolio_analysis()
            elif choice == "4":
                self.economic_outlook()
            elif choice == "5":
                self.configure_agents()
            elif choice == "6":
                self.show_help()
            elif choice == "7":
                self.running = False
                print("\n👋 Thank you for using EzioFilho AutoGen System!")
            else:
                print("\n❌ Invalid option. Please try again.")
                
    def market_analysis(self):
        """Handle market analysis request"""
        print("\n📊 MARKET ANALYSIS")
        print("-" * 40)
        query = input("Enter your analysis query: ").strip()
        
        if query:
            print("\n⏳ Processing with multiple agents...")
            results = self.system.analyze_market(query)
            
            print("\n📋 ANALYSIS RESULTS:")
            print("="*60)
            
            # Display results from each agent
            for agent, insights in results["analysis"].items():
                print(f"\n🤖 {agent}:")
                for insight in insights:
                    print(f"   {insight['content']}")
                    
    def trading_signal(self):
        """Handle trading signal request"""
        print("\n📈 TRADING SIGNAL GENERATOR")
        print("-" * 40)
        asset = input("Enter asset symbol (e.g., AAPL, BTC, EUR/USD): ").strip().upper()
        timeframe = input("Enter timeframe (1H, 4H, 1D, 1W) [default: 1D]: ").strip() or "1D"
        
        if asset:
            print(f"\n⏳ Generating signal for {asset} on {timeframe}...")
            signal = self.system.get_trading_signal(asset, timeframe)
            
            print("\n📊 TRADING SIGNAL:")
            print("="*60)
            print(f"Asset: {signal['asset']}")
            print(f"Direction: {signal['direction']}")
            print(f"Confidence: {signal['confidence']}")
            print(f"Timeframe: {signal['timeframe']}")
            print(f"Generated: {signal['timestamp']}")
            
    def portfolio_analysis(self):
        """Handle portfolio analysis"""
        print("\n💼 PORTFOLIO ANALYSIS")
        print("-" * 40)
        print("Enter your portfolio (one per line, format: SYMBOL QUANTITY PRICE)")
        print("Example: AAPL 100 150.50")
        print("Type 'done' when finished")
        
        portfolio = []
        while True:
            entry = input("➕ Add position (or 'done'): ").strip()
            if entry.lower() == 'done':
                break
            
            parts = entry.split()
            if len(parts) >= 2:
                portfolio.append({
                    "asset": parts[0].upper(),
                    "quantity": float(parts[1]),
                    "avg_price": float(parts[2]) if len(parts) > 2 else None
                })
                
        if portfolio:
            print(f"\n⏳ Analyzing portfolio with {len(portfolio)} positions...")
            analysis = self.system.portfolio_analysis(portfolio)
            
            print("\n📊 PORTFOLIO ANALYSIS:")
            print("="*60)
            for expert, insight in analysis["expert_insights"].items():
                print(f"\n🤖 {expert}:")
                print(f"   {insight[:300]}...")
                
    def economic_outlook(self):
        """Handle economic outlook request"""
        print("\n🌍 ECONOMIC OUTLOOK")
        print("-" * 40)
        region = input("Enter region (Global, US, Europe, Asia, Brazil) [default: Global]: ").strip() or "Global"
        
        print(f"\n⏳ Generating economic outlook for {region}...")
        outlook = self.system.economic_outlook(region)
        
        print("\n📊 ECONOMIC OUTLOOK:")
        print("="*60)
        print(f"Region: {outlook['region']}")
        print(f"Generated: {outlook['timestamp']}")
        
        for expert, view in outlook["expert_views"].items():
            print(f"\n🤖 {expert}:")
            print(f"   {view[:300]}...")
            
    def configure_agents(self):
        """Configure agent settings"""
        print("\n🔧 AGENT CONFIGURATION")
        print("-" * 40)
        print("Current agents:")
        for name, agent in self.system.agents.items():
            print(f"  • {name}")
            
        print("\n⚙️ Configuration options:")
        print("1. View agent details")
        print("2. Update LLM settings")
        print("3. Back to main menu")
        
        choice = input("\n👉 Select option: ").strip()
        if choice == "1":
            agent_name = input("Enter agent name: ").strip()
            if agent_name in self.system.agents:
                agent = self.system.agents[agent_name]
                print(f"\n🤖 {agent_name}:")
                print(f"System Message: {agent.system_message[:200]}...")
            else:
                print("❌ Agent not found")
                
    def show_help(self):
        """Show help information"""
        print("\n📚 HELP - EZIOFILHO AUTOGEN SYSTEM")
        print("="*60)
        print("""
This system uses multiple AI agents to provide comprehensive financial analysis:

🤖 AGENTS:
- EzioMaster: Main coordinator
- MarketAnalyst: Technical and fundamental analysis
- RiskManager: Risk assessment and portfolio protection
- DataCollector: Real-time data and sentiment
- StrategyExpert: Trading strategies and backtesting
- CryptoSpecialist: Cryptocurrency and DeFi
- MacroEconomist: Economic indicators and policy
- QuantAnalyst: Mathematical models and statistics

💡 TIPS:
- Set OPENAI_API_KEY environment variable for GPT-4
- Use specific queries for better results
- Combine multiple analyses for comprehensive insights

📊 EXAMPLES:
- "Analyze Bitcoin trend for next week"
- "Generate buy signal for AAPL"
- "What's the inflation outlook for 2025?"
- "Analyze my tech stock portfolio risk"
        """)
        input("\nPress Enter to continue...")


def main():
    """Main entry point"""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️ WARNING: OPENAI_API_KEY not set!")
        print("The system will work but with limited functionality.")
        print("Set it with: set OPENAI_API_KEY=your-api-key")
        input("\nPress Enter to continue anyway...")
    
    # Run CLI
    cli = EzioAutoGenCLI()
    cli.run()


if __name__ == "__main__":
    main()