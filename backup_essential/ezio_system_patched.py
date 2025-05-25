# ezio_system_patched.py - Patched version with better language detection
# Audit Mode: Fixed language detection and crypto queries
# Path: C:\Users\anapa\eziofilho-unified
# User: marcobarreto007
# Date: 2025-05-24 16:22:00 UTC

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

# Configure logging with less verbosity
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class EzioCompleteSystem:
    """Complete Financial AI System with Multi-API Integration"""
    
    def __init__(self):
        print("=" * 80)
        print("🚀 EZIOFILHO COMPLETE FINANCIAL AI SYSTEM v4.2")
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
            'tb': 'também',
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'preco': 'preço',
            'preços': 'preço'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
        
    def detect_language(self, text: str) -> str:
        """Improved language detection"""
        text_normalized = self.normalize_text(text)
        
        # Strong Portuguese indicators (expanded)
        pt_strong = ["você", "voce", "vc", "é", "está", "preço", "preco", "hoje", 
                     "qual", "quanto", "como", "onde", "quando", "por que", "porque",
                     "olá", "oi", "bom dia", "boa tarde", "boa noite", "tchau",
                     "obrigado", "ajuda", "ajudar", "pode", "quero", "preciso",
                     "bitcoin", "ação", "ações", "mercado", "valor", "comprar", "vender"]
        
        # Strong English indicators
        en_strong = ["you", "are", "is", "what", "how", "when", "where", "why",
                     "hello", "hi", "good", "morning", "afternoon", "evening", "night",
                     "thanks", "help", "need", "want", "can", "will", "would",
                     "price", "today", "stock", "market", "buy", "sell", "trade"]
        
        # Strong French indicators
        fr_strong = ["vous", "êtes", "est", "quel", "quoi", "comment", "où", "quand",
                     "pourquoi", "bonjour", "bonsoir", "salut", "merci", "aide",
                     "besoin", "veux", "peux", "prix", "aujourd'hui", "marché"]
        
        # Count matches
        pt_count = sum(1 for word in pt_strong if word in text_normalized)
        en_count = sum(1 for word in en_strong if word in text_normalized)
        fr_count = sum(1 for word in fr_strong if word in text_normalized)
        
        # Debug info (disabled in production)
        # print(f"[LANG DEBUG] PT:{pt_count} EN:{en_count} FR:{fr_count}")
        
        # Determine language with better logic
        if pt_count > 0 and pt_count >= en_count and pt_count >= fr_count:
            return "pt"
        elif en_count > fr_count:
            return "en"
        elif fr_count > 0:
            return "fr"
        else:
            # Default to Portuguese for Brazilian users
            return "pt"
            
    def get_factual_answer(self, query: str, language: str) -> Optional[str]:
        """Get factual answer if available"""
        query_normalized = self.normalize_text(query)
        
        # Check in the specific language
        if language in self.factual_answers:
            for key, answer in self.factual_answers[language].items():
                key_normalized = self.normalize_text(key)
                
                # Check exact match
                if key_normalized == query_normalized:
                    return answer
                    
                # Check if query contains key or vice versa
                if key_normalized in query_normalized or query_normalized in key_normalized:
                    return answer
                    
                # Check word-by-word match
                query_words = query_normalized.split()
                key_words = key_normalized.split()
                if all(word in query_words for word in key_words):
                    return answer
                    
        return None
        
    # API Integration Methods
    
    def get_crypto_data_coingecko(self, coin_id: str = "bitcoin") -> Dict[str, Any]:
        """Get cryptocurrency data from CoinGecko"""
        if not self.api_keys["coingecko"]:
            return {"error": "CoinGecko API key not found"}
            
        # Map common symbols to CoinGecko IDs
        coin_map = {
            "btc": "bitcoin",
            "bitcoin": "bitcoin",
            "eth": "ethereum",
            "ethereum": "ethereum",
            "bnb": "binancecoin",
            "sol": "solana",
            "ada": "cardano",
            "xrp": "ripple"
        }
        
        coin_id = coin_map.get(coin_id.lower(), coin_id.lower())
        
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": "usd,brl,eur",
            "include_24hr_change": "true",
            "include_market_cap": "true",
            "x_cg_demo_api_key": self.api_keys["coingecko"]
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            return data
        except Exception as e:
            return {"error": str(e)}
            
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
            
    def get_financial_news(self, query: str = "finance cryptocurrency", language: str = "pt") -> List[Dict[str, Any]]:
        """Get financial news from NewsAPI"""
        if not self.api_keys["newsapi"]:
            return [{"error": "NewsAPI key not found"}]
            
        # Map language to NewsAPI language codes
        lang_map = {
            "pt": "pt",
            "en": "en",
            "fr": "fr"
        }
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": self.api_keys["newsapi"],
            "language": lang_map.get(language, "en"),
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
        
        # Check for factual answer first
        factual_answer = self.get_factual_answer(query, language)
        if factual_answer:
            return factual_answer
            
        # Normalize query for analysis
        query_normalized = self.normalize_text(query)
        
        # Cryptocurrency queries (improved detection)
        crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "crypto", "criptomoeda", 
                          "cripto", "moeda digital", "altcoin", "bnb", "sol", "ada", "xrp"]
        
        if any(keyword in query_normalized for keyword in crypto_keywords):
            return self.handle_crypto_query(query, language)
            
        # Stock market queries
        elif any(word in query_normalized for word in ["stock", "ação", "ações", "action", 
                                                       "market", "mercado", "marché", "bolsa",
                                                       "bovespa", "nasdaq", "dow jones"]):
            return self.handle_stock_query(query, language)
            
        # News queries
        elif any(word in query_normalized for word in ["news", "notícias", "noticia", 
                                                       "nouvelles", "hoje", "today", 
                                                       "aujourd'hui", "novidades"]):
            return self.handle_news_query(query, language)
            
        # Price queries (general)
        elif any(word in query_normalized for word in ["preço", "preco", "price", "prix",
                                                       "valor", "cotação", "cotacao",
                                                       "quanto", "custa"]):
            # Try to identify what they're asking about
            if "bitcoin" in query_normalized or "btc" in query_normalized:
                return self.handle_crypto_query(query, language)
            else:
                return self.handle_stock_query(query, language)
            
        # Default response
        else:
            return self.get_default_response(language)
            
    def handle_crypto_query(self, query: str, language: str) -> str:
        """Handle cryptocurrency queries"""
        # Extract crypto symbol from query
        query_lower = self.normalize_text(query)
        
        # Default to Bitcoin if no specific crypto mentioned
        crypto_id = "bitcoin"
        for crypto in ["bitcoin", "btc", "ethereum", "eth", "bnb", "sol", "ada", "xrp"]:
            if crypto in query_lower:
                crypto_id = crypto
                break
        
        # Get crypto data
        data = self.get_crypto_data_coingecko(crypto_id)
        
        if data and crypto_id in data and "error" not in data:
            crypto_data = data[crypto_id]
            
            # Format response based on language
            if language == "pt":
                response = f"🪙 **{crypto_id.upper()} - Cotação Atual:**\n\n"
                
                if "usd" in crypto_data:
                    response += f"💵 **USD**: ${crypto_data['usd']:,.2f}\n"
                    if "usd_24h_change" in crypto_data:
                        change = crypto_data['usd_24h_change']
                        emoji = "📈" if change > 0 else "📉"
                        response += f"   {emoji} Variação 24h: {change:+.2f}%\n\n"
                
                if "brl" in crypto_data:
                    response += f"💵 **BRL**: R$ {crypto_data['brl']:,.2f}\n"
                    if "brl_24h_change" in crypto_data:
                        change = crypto_data['brl_24h_change']
                        emoji = "📈" if change > 0 else "📉"
                        response += f"   {emoji} Variação 24h: {change:+.2f}%\n\n"
                
                if "usd_market_cap" in crypto_data:
                    market_cap = crypto_data['usd_market_cap']
                    response += f"📊 **Market Cap**: ${market_cap:,.0f}\n"
                    
                response += f"\n⏰ Atualizado: {datetime.now().strftime('%H:%M:%S')}"
                
            elif language == "en":
                response = f"🪙 **{crypto_id.upper()} - Current Price:**\n\n"
                
                if "usd" in crypto_data:
                    response += f"💵 **USD**: ${crypto_data['usd']:,.2f}\n"
                    if "usd_24h_change" in crypto_data:
                        change = crypto_data['usd_24h_change']
                        emoji = "📈" if change > 0 else "📉"
                        response += f"   {emoji} 24h Change: {change:+.2f}%\n\n"
                
                if "eur" in crypto_data:
                    response += f"💵 **EUR**: €{crypto_data['eur']:,.2f}\n"
                    if "eur_24h_change" in crypto_data:
                        change = crypto_data['eur_24h_change']
                        emoji = "📈" if change > 0 else "📉"
                        response += f"   {emoji} 24h Change: {change:+.2f}%\n\n"
                
                if "usd_market_cap" in crypto_data:
                    market_cap = crypto_data['usd_market_cap']
                    response += f"📊 **Market Cap**: ${market_cap:,.0f}\n"
                    
                response += f"\n⏰ Updated: {datetime.now().strftime('%H:%M:%S')}"
                
            else:  # French
                response = f"🪙 **{crypto_id.upper()} - Prix Actuel:**\n\n"
                
                if "eur" in crypto_data:
                    response += f"💵 **EUR**: €{crypto_data['eur']:,.2f}\n"
                    if "eur_24h_change" in crypto_data:
                        change = crypto_data['eur_24h_change']
                        emoji = "📈" if change > 0 else "📉"
                        response += f"   {emoji} Variation 24h: {change:+.2f}%\n\n"
                
                if "usd" in crypto_data:
                    response += f"💵 **USD**: ${crypto_data['usd']:,.2f}\n"
                    if "usd_24h_change" in crypto_data:
                        change = crypto_data['usd_24h_change']
                        emoji = "📈" if change > 0 else "📉"
                        response += f"   {emoji} Variation 24h: {change:+.2f}%\n\n"
                
                response += f"\n⏰ Mis à jour: {datetime.now().strftime('%H:%M:%S')}"
            
            return response
        else:
            error_messages = {
                "pt": "❌ Erro ao buscar dados da criptomoeda. Tente novamente.",
                "en": "❌ Error fetching cryptocurrency data. Please try again.",
                "fr": "❌ Erreur lors de la récupération des données. Veuillez réessayer."
            }
            return error_messages.get(language, error_messages["en"])
            
    def handle_stock_query(self, query: str, language: str) -> str:
        """Handle stock market queries"""
        # Get some stock data
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        
        responses = {
            "pt": "📊 **Principais Ações do Mercado:**\n\n",
            "en": "📊 **Top Market Stocks:**\n\n",
            "fr": "📊 **Principales Actions du Marché:**\n\n"
        }
        
        response = responses.get(language, responses["en"])
        
        for symbol in symbols[:3]:  # Show only top 3
            data = self.get_stock_data_twelve(symbol)
            if "price" in data:
                price = float(data['price'])
                response += f"**{symbol}**: ${price:.2f}\n"
            else:
                if language == "pt":
                    response += f"**{symbol}**: Dados indisponíveis\n"
                elif language == "en":
                    response += f"**{symbol}**: Data unavailable\n"
                else:
                    response += f"**{symbol}**: Données non disponibles\n"
                    
        return response
        
    def handle_news_query(self, query: str, language: str) -> str:
        """Handle news queries"""
        news = self.get_financial_news("cryptocurrency bitcoin finance", language)
        
        if news and isinstance(news, list) and len(news) > 0 and "error" not in news[0]:
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
                "fr": "❌ Erreur lors de la récupération des nouvelles."
            }
            return error_responses.get(language, error_responses["en"])
        
    def get_default_response(self, language: str) -> str:
        """Get default response in the appropriate language"""
        responses = {
            "pt": (
                "🤔 Desculpe, não entendi sua pergunta.\n\n"
                "Você pode me perguntar sobre:\n"
                "• 📊 **Ações** (ex: 'mostre as ações')\n"
                "• 🪙 **Criptomoedas** (ex: 'preço do bitcoin')\n"
                "• 📰 **Notícias** (ex: 'notícias de hoje')\n"
                "• ❓ **Quem sou eu** (ex: 'quem é você?')\n\n"
                "Exemplos:\n"
                "- 'Qual o preço do BTC?'\n"
                "- 'Mostre as principais ações'\n"
                "- 'Notícias sobre crypto'"
            ),
            "en": (
                "🤔 Sorry, I didn't understand your question.\n\n"
                "You can ask me about:\n"
                "• 📊 **Stocks** (e.g., 'show stocks')\n"
                "• 🪙 **Cryptocurrencies** (e.g., 'bitcoin price')\n"
                "• 📰 **News** (e.g., 'today's news')\n"
                "• ❓ **Who I am** (e.g., 'who are you?')\n\n"
                "Examples:\n"
                "- 'What's the price of BTC?'\n"
                "- 'Show top stocks'\n"
                "- 'Crypto news'"
            ),
            "fr": (
                "🤔 Désolé, je n'ai pas compris votre question.\n\n"
                "Vous pouvez me demander:\n"
                "• 📊 **Actions** (ex: 'montrer les actions')\n"
                "• 🪙 **Cryptomonnaies** (ex: 'prix du bitcoin')\n"
                "• 📰 **Nouvelles** (ex: 'nouvelles d'aujourd'hui')\n"
                "• ❓ **Qui je suis** (ex: 'qui es-tu?')"
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