# ezio_complete_system.py - Complete Financial AI System with All APIs

import os
import sys
import time
import json
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config/.env')

# Import multilingual responses
from core.multilingual_responses import FACTUAL_ANSWERS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class EzioCompleteSystem:
    """Complete Financial AI System with Multi-API Integration"""
    
    def __init__(self):
        print("=" * 80)
        print("🚀 EZIOFILHO COMPLETE FINANCIAL AI SYSTEM v4.0")
        print("🌍 Multilingual Support: PT, EN, FR")
        print("=" * 80)
        
        # Load all API keys
        self.api_keys = {
            "twelve_data": os.getenv("TWELVE_DATA_API_KEY"),
            "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
            "huggingface": os.getenv("HUGGINGFACE_API_KEY"),
            "wolfram": os.getenv("WOLFRAM_API_KEY"),
            "newsapi": os.getenv("NEWSAPI_KEY"),
            "coingecko": os.getenv("COINGECKO_API_KEY"),
            "youtube": os.getenv("YOUTUBE_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "the_odds": os.getenv("THE_ODDS_API_KEY")
        }
        
        # Initialize components
        self.current_language = "pt"  # Default to Portuguese
        self.market_cache = {}
        self.news_cache = {}
        
        # Load factual answers
        self.factual_answers = FACTUAL_ANSWERS
        
        # Initialize all API clients
        self.initialize_apis()
        
        print("\n✅ System ready with all APIs integrated!")
        print("=" * 80)
        
    def initialize_apis(self):
        """Initialize all API connections"""
        print("\n🔌 Initializing APIs:")
        
        # Check each API
        for api_name, api_key in self.api_keys.items():
            if api_key:
                print(f"   ✓ {api_name.upper()} API: Ready")
            else:
                print(f"   ✗ {api_name.upper()} API: Missing key")
                
    def detect_language(self, text: str) -> str:
        """Detect language from text"""
        text_lower = text.lower()
        
        # Portuguese indicators
        pt_words = ["você", "olá", "oi", "ajuda", "quem", "criou", "boa noite", "bom dia"]
        # English indicators
        en_words = ["you", "hello", "hi", "help", "who", "created", "good night", "good morning"]
        # French indicators
        fr_words = ["vous", "bonjour", "salut", "aide", "qui", "créé", "bonne nuit"]
        
        pt_count = sum(1 for word in pt_words if word in text_lower)
        en_count = sum(1 for word in en_words if word in text_lower)
        fr_count = sum(1 for word in fr_words if word in text_lower)
        
        if pt_count > en_count and pt_count > fr_count:
            return "pt"
        elif en_count > fr_count:
            return "en"
        else:
            return "fr"
            
    def get_factual_answer(self, query: str, language: str) -> Optional[str]:
        """Get factual answer if available"""
        query_lower = query.lower().strip()
        
        # Check in the specific language
        if language in self.factual_answers:
            for key, answer in self.factual_answers[language].items():
                if key in query_lower or query_lower in key:
                    return answer
                    
        return None
        
    # API Integration Methods
    
    def get_stock_data_twelve(self, symbol: str) -> Dict[str, Any]:
        """Get stock data from Twelve Data API"""
        if not self.api_keys["twelve_data"]:
            return {"error": "Twelve Data API key not found"}
            
        url = f"https://api.twelvedata.com/price"
        params = {
            "symbol": symbol,
            "apikey": self.api_keys["twelve_data"]
        }
        
        try:
            response = requests.get(url, params=params)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
            
    def get_crypto_data_coingecko(self, coin_id: str = "bitcoin") -> Dict[str, Any]:
        """Get cryptocurrency data from CoinGecko"""
        if not self.api_keys["coingecko"]:
            return {"error": "CoinGecko API key not found"}
            
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        headers = {
            "x-cg-demo-api-key": self.api_keys["coingecko"]
        }
        
        try:
            response = requests.get(url, headers=headers)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
            
    def get_financial_news(self, query: str = "finance") -> List[Dict[str, Any]]:
        """Get financial news from NewsAPI"""
        if not self.api_keys["newsapi"]:
            return [{"error": "NewsAPI key not found"}]
            
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": self.api_keys["newsapi"],
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 5
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            return data.get("articles", [])
        except Exception as e:
            return [{"error": str(e)}]
            
    def process_query(self, query: str) -> str:
        """Process user query with multilingual support"""
        # Detect language
        language = self.detect_language(query)
        self.current_language = language
        
        # Check for factual answer first
        factual_answer = self.get_factual_answer(query, language)
        if factual_answer:
            return factual_answer
            
        # Process based on query content
        query_lower = query.lower()
        
        # Stock market queries
        if any(word in query_lower for word in ["stock", "ação", "action", "market", "mercado", "marché"]):
            return self.handle_stock_query(query, language)
            
        # Cryptocurrency queries
        elif any(word in query_lower for word in ["bitcoin", "crypto", "ethereum", "btc", "eth", "criptomoeda", "cryptomonnaie"]):
            return self.handle_crypto_query(query, language)
            
        # News queries
        elif any(word in query_lower for word in ["news", "notícias", "nouvelles", "hoje", "today", "aujourd'hui"]):
            return self.handle_news_query(query, language)
            
        # Default response
        else:
            return self.get_default_response(language)
            
    def handle_stock_query(self, query: str, language: str) -> str:
        """Handle stock market queries"""
        # Get some stock data
        symbols = ["AAPL", "GOOGL", "MSFT"]
        responses = {
            "pt": "📊 Aqui estão os dados do mercado:\n\n",
            "en": "📊 Here's the market data:\n\n",
            "fr": "📊 Voici les données du marché:\n\n"
        }
        
        response = responses.get(language, responses["en"])
        
        for symbol in symbols:
            data = self.get_stock_data_twelve(symbol)
            if "price" in data:
                response += f"{symbol}: ${data['price']}\n"
                
        return response
        
    def handle_crypto_query(self, query: str, language: str) -> str:
        """Handle cryptocurrency queries"""
        # Get Bitcoin data
        btc_data = self.get_crypto_data_coingecko("bitcoin")
        
        if "error" not in btc_data:
            price = btc_data.get("market_data", {}).get("current_price", {}).get("usd", 0)
            change_24h = btc_data.get("market_data", {}).get("price_change_percentage_24h", 0)
            
            responses = {
                "pt": f"🪙 Bitcoin (BTC)\nPreço: ${price:,.2f}\nVariação 24h: {change_24h:.2f}%",
                "en": f"🪙 Bitcoin (BTC)\nPrice: ${price:,.2f}\n24h Change: {change_24h:.2f}%",
                "fr": f"🪙 Bitcoin (BTC)\nPrix: ${price:,.2f}\nVariation 24h: {change_24h:.2f}%"
            }
            
            return responses.get(language, responses["en"])
            
        return "Error fetching crypto data"
        
    def handle_news_query(self, query: str, language: str) -> str:
        """Handle news queries"""
        news = self.get_financial_news()
        
        if news and "error" not in news[0]:
            responses = {
                "pt": "📰 Últimas notícias financeiras:\n\n",
                "en": "📰 Latest financial news:\n\n",
                "fr": "📰 Dernières nouvelles financières:\n\n"
            }
            
            response = responses.get(language, responses["en"])
            
            for i, article in enumerate(news[:3], 1):
                title = article.get("title", "")
                response += f"{i}. {title}\n"
                
            return response
            
        return "Error fetching news"
        
    def get_default_response(self, language: str) -> str:
        """Get default response in the appropriate language"""
        responses = {
            "pt": "Desculpe, não entendi. Você pode perguntar sobre ações, criptomoedas ou notícias financeiras.",
            "en": "Sorry, I didn't understand. You can ask about stocks, cryptocurrencies, or financial news.",
            "fr": "Désolé, je n'ai pas compris. Vous pouvez demander des actions, des cryptomonnaies ou des nouvelles financières."
        }
        
        return responses.get(language, responses["en"])
        
    def interactive_chat(self):
        """Run interactive multilingual chat"""
        print("\n💬 SuperEzio Multilingual Chat")
        print("🌍 Languages: PT, EN, FR")
        print("Type 'exit' to quit")
        print("=" * 80)
        
        while True:
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'sair', 'sortir']:
                print("\n👋 Goodbye! Tchau! Au revoir!")
                break
                
            if not user_input:
                continue
                
            # Process query
            response = self.process_query(user_input)
            print(f"\n🤖 SuperEzio: {response}")

# Main execution
if __name__ == "__main__":
    try:
        # Create and run the system
        system = EzioCompleteSystem()
        
        # Run interactive chat
        system.interactive_chat()
        
    except KeyboardInterrupt:
        print("\n\n👋 System terminated by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()