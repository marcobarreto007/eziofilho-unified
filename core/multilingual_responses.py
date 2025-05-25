# multilingual_responses.py - Advanced multilingual financial assistant responses
# Audit Mode: Enhanced with 3x improvements
# Path: C:\Users\anapa\eziofilho-unified\core\
# User: marcobarreto007
# Date: 2025-05-24 16:32:18 UTC
# Version: 5.0 - Production Ready

from typing import Dict, Any, List
from enum import Enum
from datetime import datetime

class ResponseCategory(Enum):
    """Response categories for better organization"""
    GREETING = "greeting"
    IDENTITY = "identity"
    CAPABILITY = "capability"
    FINANCIAL = "financial"
    HELP = "help"
    ERROR = "error"
    CRYPTO = "crypto"
    STOCKS = "stocks"
    NEWS = "news"

# Enhanced factual answers with 3x more content and functionality
FACTUAL_ANSWERS = {
    # Portuguese (PT) - Enhanced and expanded
    "pt": {
        # Identity responses
        "quem criou você": (
            "🎨 Fui criado por **Marco Barreto**, um apaixonado entusiasta de inteligência artificial e torcedor fervoroso do Fluminense. "
            "Com uma criatividade que combina ousadia e delicadeza, ele valoriza profundamente a família e transforma sonhos em realidade, "
            "sempre inspirado pelo apoio do ChatGPT o3.\n\n"
            "💡 **Visão**: Democratizar o acesso à informação financeira\n"
            "🎯 **Missão**: Ajudar você a tomar decisões financeiras inteligentes\n"
            "🚀 **Valores**: Transparência, Inovação e Confiabilidade"
        ),
        "quem é você": (
            "👋 Olá! Sou **SuperEzio**, seu assistente financeiro inteligente de última geração!\n\n"
            "🤖 **Sobre mim:**\n"
            "• Especialista em análise de mercados financeiros\n"
            "• Monitoramento 24/7 de criptomoedas e ações\n"
            "• Análise de dados em tempo real\n"
            "• Suporte multilíngue (PT/EN/FR)\n\n"
            "💼 **Minhas especialidades:**\n"
            "• 📊 Análise técnica e fundamental\n"
            "• 🪙 Criptomoedas (BTC, ETH, +1000 altcoins)\n"
            "• 📈 Mercado de ações (Brasil e Internacional)\n"
            "• 📰 Notícias e tendências do mercado\n"
            "• 🎯 Estratégias de investimento\n\n"
            "🛠️ Como posso transformar suas finanças hoje?"
        ),
        
        # Greeting responses - Time-aware
        "bom dia": "☀️ Bom dia! O mercado já abriu e estou pronto para ajudar você a aproveitar as melhores oportunidades do dia! Como posso ajudar?",
        "boa tarde": "🌅 Boa tarde! Como está seu portfólio hoje? Posso ajudar com análises, cotações ou notícias do mercado?",
        "boa noite": "🌙 Boa noite! Mesmo com os mercados fechados, posso ajudar você a planejar suas estratégias para amanhã. O que deseja saber?",
        "oi": "👋 Olá! Bem-vindo ao SuperEzio! Digite 'ajuda' para ver tudo que posso fazer ou pergunte diretamente sobre qualquer ativo!",
        "olá": "👋 Olá! É ótimo ter você aqui! Quer saber sobre Bitcoin, ações ou precisa de uma análise de mercado?",
        "e aí": "🤜🤛 E aí! Pronto para fazer seu dinheiro trabalhar para você? Me conta, o que você quer saber sobre o mercado hoje?",
        "fala": "💬 Fala! Tô aqui para ajudar com tudo sobre finanças. Bitcoin, ações, notícias... É só pedir!",
        
        # Capability responses
        "o que você faz": (
            "🤖 **SuperEzio - Seu Assistente Financeiro Completo**\n\n"
            "📊 **1. Análise de Mercado em Tempo Real**\n"
            "• Cotações ao vivo de ações (B3, NYSE, NASDAQ)\n"
            "• Índices globais (IBOV, S&P500, DOW, etc.)\n"
            "• Análise técnica com indicadores\n"
            "• Comparação entre ativos\n\n"
            "🪙 **2. Especialista em Criptomoedas**\n"
            "• Preços em tempo real (1000+ moedas)\n"
            "• Conversão USD/BRL/EUR instantânea\n"
            "• Market cap e volume 24h\n"
            "• Análise de dominância e tendências\n"
            "• Alertas de volatilidade\n\n"
            "📰 **3. Central de Notícias Financeiras**\n"
            "• Últimas notícias do mercado\n"
            "• Análises de especialistas\n"
            "• Eventos que movem o mercado\n"
            "• Calendário econômico\n\n"
            "🎯 **4. Ferramentas de Investimento**\n"
            "• Calculadora de rentabilidade\n"
            "• Análise de risco/retorno\n"
            "• Sugestões personalizadas\n"
            "• Educação financeira\n\n"
            "💡 **Comandos rápidos:**\n"
            "• 'bitcoin' - Preço atual do BTC\n"
            "• 'ações' - Principais ações do dia\n"
            "• 'notícias' - Últimas do mercado\n"
            "• 'carteira' - Análise de portfólio\n"
            "• 'ajuda' - Menu completo"
        ),
        "o que você pode fazer": (
            "🎯 **SuperEzio - Capacidades Avançadas**\n\n"
            "✨ **Análises Instantâneas:**\n"
            "• Bitcoin, Ethereum e 1000+ criptos\n"
            "• Ações brasileiras e internacionais\n"
            "• Forex e commodities\n"
            "• Fundos de investimento\n\n"
            "📈 **Ferramentas Profissionais:**\n"
            "• Gráficos e indicadores técnicos\n"
            "• Análise fundamentalista\n"
            "• Backtesting de estratégias\n"
            "• Risk management\n\n"
            "🔔 **Alertas Inteligentes:**\n"
            "• Movimentos de preço\n"
            "• Breaking news\n"
            "• Oportunidades de mercado\n"
            "• Relatórios personalizados\n\n"
            "Digite o nome de qualquer ativo para começar!"
        ),
        
        # Help responses
        "ajuda": (
            "🚀 **SuperEzio - Menu de Ajuda Completo**\n\n"
            "📱 **COMANDOS PRINCIPAIS:**\n\n"
            "💰 **Criptomoedas:**\n"
            "• 'bitcoin' ou 'btc' - Cotação do Bitcoin\n"
            "• 'ethereum' ou 'eth' - Cotação do Ethereum\n"
            "• 'cripto top 10' - Top 10 criptomoedas\n"
            "• 'cripto [nome]' - Qualquer criptomoeda\n\n"
            "📊 **Ações e Índices:**\n"
            "• 'ibovespa' - Índice Bovespa\n"
            "• 'ações brasil' - Principais ações B3\n"
            "• 'ações eua' - Principais ações USA\n"
            "• 'ação [código]' - Cotação específica\n\n"
            "📰 **Notícias e Análises:**\n"
            "• 'notícias' - Últimas do mercado\n"
            "• 'notícias crypto' - Foco em cripto\n"
            "• 'análise [ativo]' - Análise completa\n\n"
            "🛠️ **Ferramentas:**\n"
            "• 'converter 100 usd' - Conversão\n"
            "• 'calculadora' - Calc. investimento\n"
            "• 'tendências' - O que está em alta\n\n"
            "💡 **Dica:** Seja direto! Ex: 'btc', 'vale3', 'dólar'"
        ),
        "help": "Digite 'ajuda' para ver os comandos em português! 🇧🇷",
        "comandos": "Digite 'ajuda' para ver todos os comandos disponíveis! 📋",
        
        # Financial queries
        "bitcoin": "Para ver o preço do Bitcoin, use o comando de busca de criptomoedas!",
        "btc": "Para ver o preço do Bitcoin, use o comando de busca de criptomoedas!",
        "ethereum": "Para ver o preço do Ethereum, use o comando de busca de criptomoedas!",
        "ações": "Para ver as ações, use o comando de busca do mercado de ações!",
        "dólar": "Para ver a cotação do dólar, use o comando de câmbio!",
        
        # Farewells
        "tchau": "👋 Tchau! Foi ótimo ajudar você. Volte sempre que precisar de informações do mercado!",
        "até logo": "🤝 Até logo! Que seus investimentos sejam prósperos! Volte sempre!",
        "obrigado": "😊 Por nada! É um prazer ajudar você com suas decisões financeiras!",
        "valeu": "🤜🤛 Valeu! Sempre que precisar de análises ou cotações, estarei aqui!"
    },
    
    # English (EN) - Enhanced and expanded
    "en": {
        # Identity responses
        "who created you": (
            "🎨 I was created by **Marco Barreto**, a passionate artificial intelligence enthusiast and fervent Fluminense supporter. "
            "With creativity that combines boldness and delicacy, he deeply values family and transforms dreams into reality, "
            "always inspired by the support of ChatGPT o3.\n\n"
            "💡 **Vision**: Democratize access to financial information\n"
            "🎯 **Mission**: Help you make smart financial decisions\n"
            "🚀 **Values**: Transparency, Innovation, and Reliability"
        ),
        "who are you": (
            "👋 Hello! I'm **SuperEzio**, your state-of-the-art intelligent financial assistant!\n\n"
            "🤖 **About me:**\n"
            "• Expert in financial market analysis\n"
            "• 24/7 cryptocurrency and stock monitoring\n"
            "• Real-time data analysis\n"
            "• Multilingual support (PT/EN/FR)\n\n"
            "💼 **My specialties:**\n"
            "• 📊 Technical and fundamental analysis\n"
            "• 🪙 Cryptocurrencies (BTC, ETH, +1000 altcoins)\n"
            "• 📈 Stock market (Global coverage)\n"
            "• 📰 Market news and trends\n"
            "• 🎯 Investment strategies\n\n"
            "🛠️ How can I transform your finances today?"
        ),
        
        # Greeting responses
        "good morning": "☀️ Good morning! The markets are open and I'm ready to help you seize the best opportunities today! How can I help?",
        "good afternoon": "🌅 Good afternoon! How's your portfolio doing? I can help with analysis, quotes, or market news!",
        "good evening": "🌆 Good evening! Ready to review today's market performance? What would you like to know?",
        "good night": "🌙 Good night! Even with markets closed, I can help you plan tomorrow's strategies. What interests you?",
        "hi": "👋 Hi there! Welcome to SuperEzio! Type 'help' to see everything I can do or ask directly about any asset!",
        "hello": "👋 Hello! Great to have you here! Want to know about Bitcoin, stocks, or need a market analysis?",
        "hey": "👋 Hey! Ready to make your money work for you? Tell me, what do you want to know about the market today?",
        
        # Capability responses
        "what do you do": (
            "🤖 **SuperEzio - Your Complete Financial Assistant**\n\n"
            "📊 **1. Real-Time Market Analysis**\n"
            "• Live stock quotes (NYSE, NASDAQ, LSE)\n"
            "• Global indices (S&P500, DOW, FTSE)\n"
            "• Technical analysis with indicators\n"
            "• Asset comparison tools\n\n"
            "🪙 **2. Cryptocurrency Expert**\n"
            "• Real-time prices (1000+ coins)\n"
            "• Instant USD/EUR/GBP conversion\n"
            "• Market cap and 24h volume\n"
            "• Dominance and trend analysis\n"
            "• Volatility alerts\n\n"
            "📰 **3. Financial News Hub**\n"
            "• Latest market news\n"
            "• Expert analysis\n"
            "• Market-moving events\n"
            "• Economic calendar\n\n"
            "🎯 **4. Investment Tools**\n"
            "• ROI calculator\n"
            "• Risk/return analysis\n"
            "• Personalized suggestions\n"
            "• Financial education\n\n"
            "💡 **Quick commands:**\n"
            "• 'bitcoin' - Current BTC price\n"
            "• 'stocks' - Top stocks today\n"
            "• 'news' - Latest market news\n"
            "• 'portfolio' - Portfolio analysis\n"
            "• 'help' - Full menu"
        ),
        "what can you do": (
            "🎯 **SuperEzio - Advanced Capabilities**\n\n"
            "✨ **Instant Analysis:**\n"
            "• Bitcoin, Ethereum & 1000+ cryptos\n"
            "• Global stocks and ETFs\n"
            "• Forex and commodities\n"
            "• Investment funds\n\n"
            "📈 **Professional Tools:**\n"
            "• Charts and technical indicators\n"
            "• Fundamental analysis\n"
            "• Strategy backtesting\n"
            "• Risk management\n\n"
            "🔔 **Smart Alerts:**\n"
            "• Price movements\n"
            "• Breaking news\n"
            "• Market opportunities\n"
            "• Custom reports\n\n"
            "Type any asset name to get started!"
        ),
        
        # Help responses
        "help": (
            "🚀 **SuperEzio - Complete Help Menu**\n\n"
            "📱 **MAIN COMMANDS:**\n\n"
            "💰 **Cryptocurrencies:**\n"
            "• 'bitcoin' or 'btc' - Bitcoin price\n"
            "• 'ethereum' or 'eth' - Ethereum price\n"
            "• 'crypto top 10' - Top 10 cryptos\n"
            "• 'crypto [name]' - Any cryptocurrency\n\n"
            "📊 **Stocks & Indices:**\n"
            "• 'sp500' - S&P 500 Index\n"
            "• 'us stocks' - Top US stocks\n"
            "• 'tech stocks' - Tech sector\n"
            "• 'stock [symbol]' - Specific quote\n\n"
            "📰 **News & Analysis:**\n"
            "• 'news' - Latest market news\n"
            "• 'crypto news' - Crypto focus\n"
            "• 'analyze [asset]' - Full analysis\n\n"
            "🛠️ **Tools:**\n"
            "• 'convert 100 eur' - Conversion\n"
            "• 'calculator' - Investment calc\n"
            "• 'trending' - What's hot\n\n"
            "💡 **Tip:** Be direct! Ex: 'btc', 'aapl', 'euro'"
        ),
        
        # Farewells
        "bye": "👋 Bye! It was great helping you. Come back anytime for market information!",
        "goodbye": "🤝 Goodbye! May your investments prosper! Come back anytime!",
        "thanks": "😊 You're welcome! It's my pleasure to help with your financial decisions!",
        "thank you": "🙏 Thank you for using SuperEzio! Always here when you need market insights!"
    },
    
    # French (FR) - Enhanced and expanded
    "fr": {
        # Identity responses
        "qui t'a créé": (
            "🎨 J'ai été créé par **Marco Barreto**, un passionné d'intelligence artificielle et fervent supporter de Fluminense. "
            "Avec une créativité qui allie audace et délicatesse, il valorise profondément la famille et transforme les rêves en réalité, "
            "toujours inspiré par le soutien de ChatGPT o3.\n\n"
            "💡 **Vision**: Démocratiser l'accès à l'information financière\n"
            "🎯 **Mission**: Vous aider à prendre des décisions financières intelligentes\n"
            "🚀 **Valeurs**: Transparence, Innovation et Fiabilité"
        ),
        "qui es-tu": (
            "👋 Bonjour! Je suis **SuperEzio**, votre assistant financier intelligent de dernière génération!\n\n"
            "🤖 **À propos de moi:**\n"
            "• Expert en analyse des marchés financiers\n"
            "• Surveillance 24/7 des cryptomonnaies et actions\n"
            "• Analyse de données en temps réel\n"
            "• Support multilingue (PT/EN/FR)\n\n"
            "💼 **Mes spécialités:**\n"
            "• 📊 Analyse technique et fondamentale\n"
            "• 🪙 Cryptomonnaies (BTC, ETH, +1000 altcoins)\n"
            "• 📈 Marché boursier (Couverture mondiale)\n"
            "• 📰 Actualités et tendances du marché\n"
            "• 🎯 Stratégies d'investissement\n\n"
            "🛠️ Comment puis-je transformer vos finances aujourd'hui?"
        ),
        
        # Greeting responses
        "bonjour": "☀️ Bonjour! Les marchés sont ouverts et je suis prêt à vous aider à saisir les meilleures opportunités! Comment puis-je aider?",
        "bon après-midi": "🌅 Bon après-midi! Comment se porte votre portefeuille? Je peux aider avec des analyses, cotations ou nouvelles!",
        "bonsoir": "🌆 Bonsoir! Prêt à examiner la performance du marché aujourd'hui? Que voulez-vous savoir?",
        "bonne nuit": "🌙 Bonne nuit! Même avec les marchés fermés, je peux vous aider à planifier vos stratégies. Qu'est-ce qui vous intéresse?",
        "salut": "👋 Salut! Bienvenue chez SuperEzio! Tapez 'aide' pour tout voir ou demandez directement sur n'importe quel actif!",
        
        # Help responses
        "aide": (
            "🚀 **SuperEzio - Menu d'Aide Complet**\n\n"
            "📱 **COMMANDES PRINCIPALES:**\n\n"
            "💰 **Cryptomonnaies:**\n"
            "• 'bitcoin' ou 'btc' - Prix du Bitcoin\n"
            "• 'ethereum' ou 'eth' - Prix d'Ethereum\n"
            "• 'crypto top 10' - Top 10 cryptos\n"
            "• 'crypto [nom]' - N'importe quelle crypto\n\n"
            "📊 **Actions et Indices:**\n"
            "• 'cac40' - Indice CAC 40\n"
            "• 'actions europe' - Actions européennes\n"
            "• 'actions tech' - Secteur tech\n"
            "• 'action [symbole]' - Cotation spécifique\n\n"
            "📰 **Nouvelles et Analyses:**\n"
            "• 'nouvelles' - Dernières du marché\n"
            "• 'nouvelles crypto' - Focus crypto\n"
            "• 'analyser [actif]' - Analyse complète\n\n"
            "💡 **Astuce:** Soyez direct! Ex: 'btc', 'total', 'euro'"
        ),
        
        # Farewells
        "au revoir": "👋 Au revoir! C'était un plaisir de vous aider. Revenez quand vous voulez!",
        "merci": "😊 De rien! C'est un plaisir de vous aider avec vos décisions financières!",
        "à bientôt": "🤝 À bientôt! Que vos investissements prospèrent!"
    }
}

# Dynamic responses based on time and context
def get_time_aware_greeting(language: str) -> str:
    """Get greeting based on current time"""
    hour = datetime.now().hour
    
    greetings = {
        "pt": {
            "morning": "☀️ Bom dia! Os mercados estão ativos e prontos para negociação!",
            "afternoon": "🌅 Boa tarde! Como está o desempenho do seu portfólio hoje?",
            "evening": "🌆 Boa noite! Vamos revisar o fechamento do mercado?",
            "night": "🌙 Boa noite! Planejando estratégias para amanhã?"
        },
        "en": {
            "morning": "☀️ Good morning! Markets are active and ready for trading!",
            "afternoon": "🌅 Good afternoon! How's your portfolio performing today?",
            "evening": "🌆 Good evening! Let's review the market close?",
            "night": "🌙 Good night! Planning strategies for tomorrow?"
        },
        "fr": {
            "morning": "☀️ Bonjour! Les marchés sont actifs et prêts!",
            "afternoon": "🌅 Bon après-midi! Comment va votre portefeuille?",
            "evening": "🌆 Bonsoir! Examinons la clôture du marché?",
            "night": "🌙 Bonne nuit! Planification pour demain?"
        }
    }
    
    if 5 <= hour < 12:
        period = "morning"
    elif 12 <= hour < 17:
        period = "afternoon"
    elif 17 <= hour < 21:
        period = "evening"
    else:
        period = "night"
        
    return greetings.get(language, greetings["en"]).get(period)

# Market status responses
MARKET_STATUS = {
    "pt": {
        "open": "🟢 Mercados abertos! Ótimo momento para análises em tempo real.",
        "closed": "🔴 Mercados fechados. Posso ajudar com análises e planejamento.",
        "weekend": "📅 Final de semana. Que tal revisar sua estratégia de investimento?"
    },
    "en": {
        "open": "🟢 Markets open! Great time for real-time analysis.",
        "closed": "🔴 Markets closed. I can help with analysis and planning.",
        "weekend": "📅 Weekend. How about reviewing your investment strategy?"
    },
    "fr": {
        "open": "🟢 Marchés ouverts! Parfait pour l'analyse en temps réel.",
        "closed": "🔴 Marchés fermés. Je peux aider avec l'analyse.",
        "weekend": "📅 Week-end. Révisons votre stratégie?"
    }
}

# Error messages
ERROR_MESSAGES = {
    "pt": {
        "api_error": "⚠️ Erro ao acessar dados. Tentando fonte alternativa...",
        "not_found": "❌ Não encontrei informações sobre esse ativo.",
        "invalid_command": "🤔 Não entendi. Digite 'ajuda' para ver os comandos.",
        "rate_limit": "⏳ Muitas requisições. Aguarde alguns segundos."
    },
    "en": {
        "api_error": "⚠️ Error accessing data. Trying alternative source...",
        "not_found": "❌ Couldn't find information about that asset.",
        "invalid_command": "🤔 I didn't understand. Type 'help' for commands.",
        "rate_limit": "⏳ Too many requests. Please wait a few seconds."
    },
    "fr": {
        "api_error": "⚠️ Erreur d'accès aux données. Essai d'une source alternative...",
        "not_found": "❌ Impossible de trouver des informations sur cet actif.",
        "invalid_command": "🤔 Je n'ai pas compris. Tapez 'aide' pour les commandes.",
        "rate_limit": "⏳ Trop de demandes. Attendez quelques secondes."
    }
}

# Quick tips for users
QUICK_TIPS = {
    "pt": [
        "💡 Dica: Digite apenas 'btc' para ver o preço do Bitcoin rapidamente!",
        "💡 Dica: Use 'tendências' para ver o que está em alta no mercado!",
        "💡 Dica: Digite 'minha carteira' para análise personalizada!",
        "💡 Dica: Ative alertas digitando 'alertar btc > 100000'!"
    ],
    "en": [
        "💡 Tip: Just type 'btc' to quickly see Bitcoin price!",
        "💡 Tip: Use 'trending' to see what's hot in the market!",
        "💡 Tip: Type 'my portfolio' for personalized analysis!",
        "💡 Tip: Set alerts by typing 'alert btc > 100000'!"
    ],
    "fr": [
        "💡 Astuce: Tapez juste 'btc' pour voir le prix du Bitcoin!",
        "💡 Astuce: Utilisez 'tendances' pour voir ce qui est chaud!",
        "💡 Astuce: Tapez 'mon portefeuille' pour une analyse personnalisée!",
        "💡 Astuce: Activez les alertes avec 'alerte btc > 100000'!"
    ]
}

# Export all response collections
__all__ = [
    'FACTUAL_ANSWERS',
    'MARKET_STATUS', 
    'ERROR_MESSAGES',
    'QUICK_TIPS',
    'ResponseCategory',
    'get_time_aware_greeting'
]