# ezio_complete_system_fixed.py - Fixed Complete Financial AI System

import os
import sys
import time
import json
import requests
import re
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
        print("🚀 EZIOFILHO COMPLETE FINANCIAL AI SYSTEM v4.1")
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
                
    def normalize_text(self, text: str) -> str:
        """Normalize text for matching"""
        # Remove accents and special characters
        text = text.lower().strip()
        # Remove punctuation
        text = re.sub(r'[?!.,;:\'"¿¡]', '', text)
        # Remove extra spaces
        text = ' '.join(text.split())
        # Common replacements
        replacements = {
            'voce': 'você',
            'vc': 'você',
            'eh': 'é',
            'ta': 'está',
            'pq': 'porque',
            'tb': 'também'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
        
    def detect_language(self, text: str) -> str:
        """Detect language from text"""
        text_lower = self.normalize_text(text)
        
        # Portuguese indicators
        pt_words = ["você", "olá", "oi", "ajuda", "quem", "criou", "boa", "noite", "dia", "tarde", "é"]
        # English indicators
        en_words = ["you", "hello", "hi", "help", "who", "created", "good", "night", "morning", "are"]
        # French indicators
        fr_words = ["vous", "bonjour", "salut", "aide", "qui", "créé", "bonne", "nuit", "es", "tu"]
        
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
        query_normalized = self.normalize_text(query)
        
        # Debug print
        print(f"[DEBUG] Normalized query: '{query_normalized}'")
        print(f"[DEBUG] Detected language: {language}")
        
        # Check in the specific language
        if language in self.factual_answers:
            for key, answer in self.factual_answers[language].items():
                key_normalized = self.normalize_text(key)
                print(f"[DEBUG] Checking: '{key_normalized}'")
                
                # Check exact match
                if key_normalized == query_normalized:
                    print(f"[DEBUG] Exact match found!")
                    return answer
                    
                # Check if query contains key or vice versa
                if key_normalized in query_normalized or query_normalized in key_normalized:
                    print(f"[DEBUG] Partial match found!")
                    return answer
                    
                # Check word-by-word match
                query_words = query_normalized.split()
                key_words = key_normalized.split()
                if all(word in query_words for word in key_words):
                    print(f"[DEBUG] Word match found!")
                    return answer
                    
        # Try all languages if not found
        for lang in ["pt", "en", "fr"]:
            if lang != language and lang in self.factual_answers:
                for key, answer in self.factual_answers[lang].items():
                    key_normalized = self.normalize_text(key)
                    if key_normalized == query_normalized or key_normalized in query_normalized:
                        print(f"[DEBUG] Found in {lang} language!")
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
            response = requests.get(url, params=params, timeout=5)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
            
    def get_crypto_data_coingecko(self, coin_id: str = "bitcoin") -> Dict[str, Any]:
        """Get cryptocurrency data from CoinGecko"""
        if not self.api_keys["coingecko"]:
            return {"error": "CoinGecko API key not found"}
            
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": "usd,brl,eur",
            "include_24hr_change": "true",
            "x_cg_demo_api_key": self.api_keys["coingecko"]
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
            
    def get_financial_news(self, query: str = "finance cryptocurrency") -> List[Dict[str, Any]]:
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
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            return data.get("articles", [])
        except Exception as e:
            return [{"error": str(e)}]
            
    def process_query(self, query: str) -> str:
        """Process user query with multilingual support"""
        # Detect language
        language = self.detect_language(query)
        self.current_language = language
        
        print(f"\n[DEBUG] Processing query: '{query}'")
        print(f"[DEBUG] Language detected: {language}")
        
        # Check for factual answer first
        factual_answer = self.get_factual_answer(query, language)
        if factual_answer:
            return factual_answer
            
        # Process based on query content
        query_lower = self.normalize_text(query)
        
        # Stock market queries
        if any(word in query_lower for word in ["stock", "ação", "ações", "action", "market", "mercado", "marché", "bolsa"]):
            return self.handle_stock_query(query, language)
            
        # Cryptocurrency queries
        elif any(word in query_lower for word in ["bitcoin", "crypto", "ethereum", "btc", "eth", "criptomoeda", "cryptomonnaie", "cripto"]):
            return self.handle_crypto_query(query, language)
            
        # News queries
        elif any(word in query_lower for word in ["news", "notícias", "noticia", "nouvelles", "hoje", "today", "aujourd'hui"]):
            return self.handle_news_query(query, language)
            
        # Default response
        else:
            return self.get_default_response(language)
            
    def handle_stock_query(self, query: str, language: str) -> str:
        """Handle stock market queries"""
        # Get some stock data
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        responses = {
            "pt": "📊 Aqui estão os dados do mercado de ações:\n\n",
            "en": "📊 Here's the stock market data:\n\n",
            "fr": "📊 Voici les données du marché boursier:\n\n"
        }
        
        response = responses.get(language, responses["en"])
        
        for symbol in symbols[:3]:  # Show only top 3
            data = self.get_stock_data_twelve(symbol)
            if "price" in data:
                price = float(data['price'])
                response += f"**{symbol}**: ${price:.2f}\n"
            else:
                response += f"**{symbol}**: Dados indisponíveis\n"
                
        return response
        
    def handle_crypto_query(self, query: str, language: str) -> str:
        """Handle cryptocurrency queries"""
        # Get crypto data
        cryptos = ["bitcoin", "ethereum", "binancecoin"]
        crypto_names = {
            "bitcoin": "Bitcoin (BTC)",
            "ethereum": "Ethereum (ETH)",
            "binancecoin": "BNB"
        }
        
        responses = {
            "pt": "🪙 **Dados de Criptomoedas:**\n\n",
            "en": "🪙 **Cryptocurrency Data:**\n\n",
            "fr": "🪙 **Données des Cryptomonnaies:**\n\n"
        }
        
        response = responses.get(language, responses["en"])
        
        for crypto in cryptos:
            data = self.get_crypto_data_coingecko(crypto)
            if crypto in data and "error" not in data:
                crypto_data = data[crypto]
                price_usd = crypto_data.get("usd", 0)
                change_24h = crypto_data.get("usd_24h_change", 0)
                
                response += f"**{crypto_names[crypto]}**\n"
                response += f"   💵 USD: ${price_usd:,.2f} ({change_24h:+.2f}%)\n"
                
                if language == "pt" and "brl" in crypto_data:
                    price_brl = crypto_data.get("brl", 0)
                    response += f"   💵 BRL: R$ {price_brl:,.2f}\n"
                elif language == "fr" and "eur" in crypto_data:
                    price_eur = crypto_data.get("eur", 0)
                    response += f"   💵 EUR: €{price_eur:,.2f}\n"
                    
                response += "\n"
                
        return response
        
    def handle_news_query(self, query: str, language: str) -> str:
        """Handle news queries"""
        news = self.get_financial_news()
        
        if news and "error" not in news[0]:
            responses = {
                "pt": "📰 **Últimas Notícias Financeiras:**\n\n",
                "en": "📰 **Latest Financial News:**\n\n",
                "fr": "📰 **Dernières Nouvelles Financières:**\n\n"
            }
            
            response = responses.get(language, responses["en"])
            
            for i, article in enumerate(news[:3], 1):
                title = article.get("title", "")
                source = article.get("source", {}).get("name", "Unknown")
                response += f"{i}. **{title}**\n   _Fonte: {source}_\n\n"
                
            return response
        else:
            error_responses = {
                "pt": "❌ Erro ao buscar notícias. Tente novamente mais tarde.",
                "en": "❌ Error fetching news. Please try again later.",
                "fr": "❌ Erreur lors de la récupération des nouvelles. Veuillez réessayer plus tard."
            }
            return error_responses.get(language, error_responses["en"])
        
    def get_default_response(self, language: str) -> str:
        """Get default response in the appropriate language"""
        responses = {
            "pt": (
                "🤔 Desculpe, não entendi sua pergunta.\n\n"
                "Você pode me perguntar sobre:\n"
                "• 📊 Ações (ex: 'mostre as ações')\n"
                "• 🪙 Criptomoedas (ex: 'preço do bitcoin')\n"
                "• 📰 Notícias (ex: 'notícias de hoje')\n"
                "• ❓ Quem sou eu (ex: 'quem é você?')"
            ),
            "en": (
                "🤔 Sorry, I didn't understand your question.\n\n"
                "You can ask me about:\n"
                "• 📊 Stocks (e.g., 'show stocks')\n"
                "• 🪙 Cryptocurrencies (e.g., 'bitcoin price')\n"
                "• 📰 News (e.g., 'today's news')\n"
                "• ❓ Who I am (e.g., 'who are you?')"
            ),
            "fr": (
                "🤔 Désolé, je n'ai pas compris votre question.\n\n"
                "Vous pouvez me demander:\n"
                "• 📊 Actions (ex: 'montrer les actions')\n"
                "• 🪙 Cryptomonnaies (ex: 'prix du bitcoin')\n"
                "• 📰 Nouvelles (ex: 'nouvelles d'aujourd'hui')\n"
                "• ❓ Qui je suis (ex: 'qui es-tu?')"
            )
        }
        
        return responses.get(language, responses["en"])
        
    def interactive_chat(self):
        """Run interactive multilingual chat"""
        print("\n💬 SuperEzio Multilingual Chat")
        print("🌍 Languages: PT, EN, FR (Auto-detected)")
        print("Type 'exit' to quit")
        print("=" * 80)
        
        # Show initial help
        print("\n🤖 SuperEzio: Olá! Sou SuperEzio, seu assistente financeiro multilíngue!")
        print("Hello! I'm SuperEzio, your multilingual financial assistant!")
        print("Bonjour! Je suis SuperEzio, votre assistant financier multilingue!")
        
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