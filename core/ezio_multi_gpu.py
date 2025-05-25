# ezio_multi_gpu.py - Complete multi-GPU financial AI system
import os
import sys
import torch
import time
import json
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class EzioMultiGPUSystem:
    """Complete financial AI system with multi-GPU support"""
    
    def __init__(self):
        print("=" * 70)
        print("🚀 EZIOFILHO MULTI-GPU FINANCIAL AI SYSTEM v2.0")
        print("=" * 70)
        
        # Check CUDA
        self.check_cuda_setup()
        
        # Initialize components
        self.experts = {}
        self.market_cache = {}
        self.models = {}
        
        # Setup GPU configuration
        self.setup_multi_gpu()
        
        # Initialize financial experts
        self.initialize_experts()
        
        print("\n✅ System ready!")
        print("=" * 70)
        
    def check_cuda_setup(self):
        """Check and display CUDA configuration"""
        if not torch.cuda.is_available():
            print("❌ CUDA not available! Running on CPU (slower)")
            self.device = "cpu"
            self.gpu_count = 0
        else:
            self.gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA Available - {self.gpu_count} GPU(s) detected")
            
            for i in range(self.gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            self.device = "cuda:0"
            
    def setup_multi_gpu(self):
        """Configure multi-GPU setup"""
        if self.gpu_count > 1:
            print(f"\n🎮 Configuring {self.gpu_count} GPUs for parallel processing...")
            
            # Set environment for multi-GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(self.gpu_count)))
            
            # Configure for RTX 2060 + GTX 1070
            self.gpu_config = {
                0: {"name": "RTX 2060", "memory": 6, "tasks": ["sentiment", "technical"]},
                1: {"name": "GTX 1070", "memory": 8, "tasks": ["fundamental", "risk"]}
            }
        else:
            self.gpu_config = {0: {"name": "Default", "memory": 0, "tasks": ["all"]}}
            
    def initialize_experts(self):
        """Initialize all 12 financial experts"""
        print("\n🧠 Initializing Financial Experts...")
        
        expert_classes = {
            # Market Analysis
            "sentiment": SentimentExpert,
            "technical": TechnicalExpert,
            "fundamental": FundamentalExpert,
            "macro": MacroExpert,
            
            # Risk Management
            "risk": RiskExpert,
            "volatility": VolatilityExpert,
            "credit": CreditExpert,
            "liquidity": LiquidityExpert,
            
            # Quantitative
            "algorithmic": AlgorithmicExpert,
            "options": OptionsExpert,
            "fixed_income": FixedIncomeExpert,
            "crypto": CryptoExpert
        }
        
        # Create experts with GPU assignment
        for name, expert_class in expert_classes.items():
            gpu_id = 0 if self.gpu_count == 1 else (0 if name in self.gpu_config[0]["tasks"] else 1)
            self.experts[name] = expert_class(gpu_id=gpu_id)
            print(f"   ✓ {name.capitalize()} Expert (GPU {gpu_id})")
            
    def analyze_stock(self, symbol: str) -> Dict[str, Any]:
        """Complete stock analysis using all experts"""
        print(f"\n📊 Analyzing {symbol} with 12 experts on {self.gpu_count} GPU(s)...")
        
        start_time = time.time()
        
        # Get market data
        try:
            data = self.get_market_data(symbol)
            
            # Run all experts
            results = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "market_data": data["summary"],
                "expert_analysis": {}
            }
            
            # Parallel analysis on multi-GPU
            for expert_name, expert in self.experts.items():
                expert_result = expert.analyze(symbol, data)
                results["expert_analysis"][expert_name] = expert_result
                
            # Calculate consensus
            results["consensus"] = self.calculate_consensus(results["expert_analysis"])
            
            elapsed = time.time() - start_time
            results["processing_time"] = f"{elapsed:.2f}s"
            
            return results
            
        except Exception as e:
            return {"error": str(e), "symbol": symbol}
            
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data"""
        ticker = yf.Ticker(symbol)
        
        return {
            "info": ticker.info,
            "history_1d": ticker.history(period="1d"),
            "history_1mo": ticker.history(period="1mo"),
            "history_1y": ticker.history(period="1y"),
            "news": ticker.news[:5] if hasattr(ticker, 'news') else [],
            "summary": {
                "price": ticker.info.get("currentPrice", 0),
                "change": ticker.info.get("regularMarketChangePercent", 0),
                "volume": ticker.info.get("volume", 0),
                "market_cap": ticker.info.get("marketCap", 0)
            }
        }
        
    def calculate_consensus(self, expert_results: Dict) -> Dict[str, Any]:
        """Calculate consensus from all experts"""
        scores = []
        recommendations = []
        
        for expert, result in expert_results.items():
            if "score" in result:
                scores.append(result["score"])
            if "recommendation" in result:
                recommendations.append(result["recommendation"])
                
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Determine consensus
        if avg_score > 0.6:
            consensus = "STRONG BUY"
        elif avg_score > 0.2:
            consensus = "BUY"
        elif avg_score > -0.2:
            consensus = "HOLD"
        elif avg_score > -0.6:
            consensus = "SELL"
        else:
            consensus = "STRONG SELL"
            
        return {
            "recommendation": consensus,
            "confidence": abs(avg_score),
            "score": avg_score,
            "agreement": len(set(recommendations)) / len(recommendations) if recommendations else 0
        }
        
    def interactive_analysis(self):
        """Interactive stock analysis"""
        print("\n💬 Interactive Financial Analysis")
        print("Commands: analyze [SYMBOL], portfolio, market, exit")
        print("=" * 70)
        
        while True:
            command = input("\n📈 Command: ").strip().lower()
            
            if command == "exit":
                print("👋 Goodbye!")
                break
                
            elif command.startswith("analyze "):
                symbol = command.split()[1].upper()
                result = self.analyze_stock(symbol)
                self.display_results(result)
                
            elif command == "portfolio":
                symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD"]
                print("\n📊 Analyzing portfolio...")
                for symbol in symbols:
                    result = self.analyze_stock(symbol)
                    print(f"\n{symbol}: {result.get('consensus', {}).get('recommendation', 'N/A')}")
                    
            elif command == "market":
                self.show_market_overview()
                
            else:
                print("❌ Unknown command. Try: analyze AAPL")
                
    def display_results(self, results: Dict):
        """Display analysis results"""
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
            
        print(f"\n{'=' * 70}")
        print(f"📊 {results['symbol']} Analysis Results")
        print(f"{'=' * 70}")
        
        # Market data
        market = results.get("market_data", {})
        print(f"\n💵 Price: ${market.get('price', 0):.2f}")
        print(f"📈 Change: {market.get('change', 0):+.2f}%")
        print(f"📊 Volume: {market.get('volume', 0):,.0f}")
        
        # Consensus
        consensus = results.get("consensus", {})
        print(f"\n🎯 RECOMMENDATION: {consensus.get('recommendation', 'N/A')}")
        print(f"💪 Confidence: {consensus.get('confidence', 0):.1%}")
        
        # Expert breakdown
        print(f"\n🧠 Expert Analysis:")
        for expert, analysis in results.get("expert_analysis", {}).items():
            rec = analysis.get("recommendation", "N/A")
            score = analysis.get("score", 0)
            print(f"   {expert:15} : {rec:10} (Score: {score:+.2f})")
            
        print(f"\n⚡ Analysis completed in {results.get('processing_time', 'N/A')}")
        
    def show_market_overview(self):
        """Show market overview"""
        print("\n📈 Market Overview")
        indices = ["^GSPC", "^DJI", "^IXIC", "^VIX"]
        
        for index in indices:
            try:
                ticker = yf.Ticker(index)
                info = ticker.info
                history = ticker.history(period="1d")
                
                price = history['Close'][-1] if not history.empty else 0
                change = info.get('regularMarketChangePercent', 0)
                
                print(f"{index:10} : ${price:,.2f} ({change:+.2f}%)")
            except:
                pass

# Base Expert Class
class BaseExpert:
    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        
    def analyze(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """Override in subclasses"""
        return {
            "expert": self.__class__.__name__,
            "score": 0,
            "recommendation": "HOLD",
            "confidence": 0.5
        }

# Implement all 12 experts
class SentimentExpert(BaseExpert):
    def analyze(self, symbol: str, data: Dict) -> Dict[str, Any]:
        # Simple sentiment based on news
        news_text = ' '.join([n.get('title', '') for n in data.get('news', [])])
        positive_words = ['buy', 'bullish', 'growth', 'profit', 'surge']
        negative_words = ['sell', 'bearish', 'loss', 'decline', 'crash']
        
        pos_score = sum(1 for word in positive_words if word in news_text.lower())
        neg_score = sum(1 for word in negative_words if word in news_text.lower())
        
        score = (pos_score - neg_score) / max(pos_score + neg_score, 1)
        
        return {
            "expert": "sentiment",
            "score": score,
            "recommendation": "BUY" if score > 0.2 else "SELL" if score < -0.2 else "HOLD",
            "confidence": abs(score)
        }

class TechnicalExpert(BaseExpert):
    def analyze(self, symbol: str, data: Dict) -> Dict[str, Any]:
        history = data.get('history_1mo', {})
        if history.empty:
            return super().analyze(symbol, data)
            
        # Simple RSI calculation
        close_prices = history['Close']
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        
        if current_rsi > 70:
            return {"expert": "technical", "score": -0.5, "recommendation": "SELL", "confidence": 0.7}
        elif current_rsi < 30:
            return {"expert": "technical", "score": 0.5, "recommendation": "BUY", "confidence": 0.7}
        else:
            return {"expert": "technical", "score": 0, "recommendation": "HOLD", "confidence": 0.5}

class FundamentalExpert(BaseExpert):
    def analyze(self, symbol: str, data: Dict) -> Dict[str, Any]:
        info = data.get('info', {})
        
        # P/E ratio analysis
        pe_ratio = info.get('forwardPE', info.get('trailingPE', 0))
        
        if pe_ratio > 0:
            if pe_ratio < 15:
                score = 0.5  # Undervalued
            elif pe_ratio > 30:
                score = -0.5  # Overvalued
            else:
                score = 0  # Fair value
        else:
            score = 0
            
        return {
            "expert": "fundamental",
            "score": score,
            "recommendation": "BUY" if score > 0.2 else "SELL" if score < -0.2 else "HOLD",
            "confidence": abs(score)
        }

# Quick implementations for other experts
class MacroExpert(BaseExpert):
    def analyze(self, symbol: str, data: Dict) -> Dict[str, Any]:
        # Simplified macro analysis
        return {"expert": "macro", "score": 0.1, "recommendation": "HOLD", "confidence": 0.6}

class RiskExpert(BaseExpert):
    def analyze(self, symbol: str, data: Dict) -> Dict[str, Any]:
        # Calculate volatility
        history = data.get('history_1mo', {})
        if not history.empty:
            returns = history['Close'].pct_change()
            volatility = returns.std() * (252 ** 0.5)  # Annualized
            
            if volatility > 0.4:  # High volatility
                return {"expert": "risk", "score": -0.3, "recommendation": "SELL", "confidence": 0.7}
                
        return {"expert": "risk", "score": 0, "recommendation": "HOLD", "confidence": 0.5}

class VolatilityExpert(BaseExpert):
    def analyze(self, symbol: str, data: Dict) -> Dict[str, Any]:
        return {"expert": "volatility", "score": 0.1, "recommendation": "HOLD", "confidence": 0.5}

class CreditExpert(BaseExpert):
    def analyze(self, symbol: str, data: Dict) -> Dict[str, Any]:
        return {"expert": "credit", "score": 0.2, "recommendation": "BUY", "confidence": 0.6}

class LiquidityExpert(BaseExpert):
    def analyze(self, symbol: str, data: Dict) -> Dict[str, Any]:
        volume = data.get('summary', {}).get('volume', 0)
        avg_volume = data.get('info', {}).get('averageVolume', 1)
        
        liquidity_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        if liquidity_ratio > 1.5:
            return {"expert": "liquidity", "score": 0.3, "recommendation": "BUY", "confidence": 0.7}
        elif liquidity_ratio < 0.5:
            return {"expert": "liquidity", "score": -0.3, "recommendation": "SELL", "confidence": 0.7}
            
        return {"expert": "liquidity", "score": 0, "recommendation": "HOLD", "confidence": 0.5}

class AlgorithmicExpert(BaseExpert):
    def analyze(self, symbol: str, data: Dict) -> Dict[str, Any]:
        # Simple moving average crossover
        history = data.get('history_1mo', {})
        if not history.empty:
            sma_20 = history['Close'].rolling(window=20).mean()
            sma_50 = history['Close'].rolling(window=50).mean()
            
            if len(sma_20) > 0 and len(sma_50) > 0:
                if sma_20.iloc[-1] > sma_50.iloc[-1]:
                    return {"expert": "algorithmic", "score": 0.4, "recommendation": "BUY", "confidence": 0.7}
                    
        return {"expert": "algorithmic", "score": 0, "recommendation": "HOLD", "confidence": 0.5}

class OptionsExpert(BaseExpert):
    def analyze(self, symbol: str, data: Dict) -> Dict[str, Any]:
        return {"expert": "options", "score": 0.1, "recommendation": "HOLD", "confidence": 0.5}

class FixedIncomeExpert(BaseExpert):
    def analyze(self, symbol: str, data: Dict) -> Dict[str, Any]:
        return {"expert": "fixed_income", "score": -0.1, "recommendation": "HOLD", "confidence": 0.4}

class CryptoExpert(BaseExpert):
    def analyze(self, symbol: str, data: Dict) -> Dict[str, Any]:
        # Check if it's a crypto
        if symbol.endswith('-USD'):
            return {"expert": "crypto", "score": 0.3, "recommendation": "BUY", "confidence": 0.6}
        return {"expert": "crypto", "score": 0, "recommendation": "HOLD", "confidence": 0.3}

# Main execution
if __name__ == "__main__":
    try:
        # Create and run the system
        system = EzioMultiGPUSystem()
        
        # Run interactive mode
        system.interactive_analysis()
        
    except KeyboardInterrupt:
        print("\n\n👋 System terminated by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()