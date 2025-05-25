import sys
import os
import time
import json
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '01_core_system'))

from core.local_model_wrapper import LocalModelWrapper

class MultilingualChat:
    def __init__(self):
        print("🚀 Initializing EzioFilho Multilingual LLM System...")
        print("⏳ Loading Phi-2 model...")
        
        self.model = LocalModelWrapper(
            model_path=r"C:\Users\anapa\.cache\models\phi-2.gguf",
            model_type="gguf",
            chat_enabled=True
        )
        
        # Initialize factual answers in 3 languages
        self.init_factual_answers()
        
        # Load conversation history
        self.history_file = "chat_history.json"
        self.load_history()
        
        print("✅ Model loaded successfully!")
        print("🌍 Languages supported: EN, FR, PT")
        print("=" * 60)
        
    def init_factual_answers(self):
        """Initialize factual answers in English, French and Portuguese"""
        self.factual_answers = {
            # English
            "en": {
                "who created you": (
                    "I was created by Marco Barreto, a passionate artificial intelligence enthusiast and fervent Fluminense supporter. "
                    "With creativity that combines boldness and delicacy, he deeply values family and transforms dreams into reality, "
                    "always inspired by the support of ChatGPT o3."
                ),
                "who are you": (
                    "👋 Hello! I'm **SuperEzio**, an intelligent financial assistant!\n\n"
                    "💰 My goal is to help you understand the financial market and cryptocurrencies.\n\n"
                    "📈 I can provide data on Bitcoin, Ethereum, investments and much more!\n\n"
                    "🛠️ How can I help you today?"
                ),
                "good night": "🌙 Good night! I hope you have a great rest. How can I help?",
                "hi": "👋 Hello! How can I help you today with financial market information?",
                "hello": "👋 Hello! How can I help you today with financial market information?",
                "good morning": "☀️ Good morning! How can I help you today?",
                "good afternoon": "🌅 Good afternoon! How can I help you today?",
                "help": (
                    "🤝 I can help you with:\n\n"
                    "1. 📊 Stock market analysis\n"
                    "2. 🪙 Cryptocurrency information\n"
                    "3. 📈 Real-time market data\n\n"
                    "What would you like to know?"
                )
            },
            
            # French
            "fr": {
                "qui t'a créé": (
                    "J'ai été créé par Marco Barreto, un passionné d'intelligence artificielle et fervent supporter de Fluminense. "
                    "Avec une créativité qui allie audace et délicatesse, il valorise profondément la famille et transforme les rêves en réalité, "
                    "toujours inspiré par le soutien de ChatGPT o3."
                ),
                "qui es-tu": (
                    "👋 Bonjour! Je suis **SuperEzio**, un assistant financier intelligent!\n\n"
                    "💰 Mon objectif est de vous aider à comprendre le marché financier et les cryptomonnaies.\n\n"
                    "📈 Je peux fournir des données sur Bitcoin, Ethereum, les investissements et bien plus!\n\n"
                    "🛠️ Comment puis-je vous aider aujourd'hui?"
                ),
                "bonne nuit": "🌙 Bonne nuit! J'espère que vous aurez un bon repos. Comment puis-je aider?",
                "salut": "👋 Bonjour! Comment puis-je vous aider aujourd'hui avec des informations sur le marché financier?",
                "bonjour": "👋 Bonjour! Comment puis-je vous aider aujourd'hui avec des informations sur le marché financier?",
                "aide": (
                    "🤝 Je peux vous aider avec:\n\n"
                    "1. 📊 Analyse du marché boursier\n"
                    "2. 🪙 Informations sur les cryptomonnaies\n"
                    "3. 📈 Données de marché en temps réel\n\n"
                    "Que voulez-vous savoir?"
                )
            },
            
            # Portuguese
            "pt": {
                "quem criou você": (
                    "Fui criado por Marco Barreto, um apaixonado entusiasta de inteligência artificial e torcedor fervoroso do Fluminense. "
                    "Com uma criatividade que combina ousadia e delicadeza, ele valoriza profundamente a família e transforma sonhos em realidade, "
                    "sempre inspirado pelo apoio do ChatGPT o3."
                ),
                "quem é você": (
                    "👋 Olá! Sou **SuperEzio**, um assistente financeiro inteligente!\n\n"
                    "💰 Meu objetivo é ajudar você a entender o mercado financeiro e as criptomoedas.\n\n"
                    "📈 Posso fornecer dados sobre Bitcoin, Ethereum, investimentos e muito mais!\n\n"
                    "🛠️ Como posso te ajudar hoje?"
                ),
                "boa noite": "🌙 Boa noite! Espero que tenha um ótimo descanso. Como posso ajudar?",
                "oi": "👋 Olá! Como posso ajudar você hoje com informações sobre mercado financeiro?",
                "olá": "👋 Olá! Como posso ajudar você hoje com informações sobre mercado financeiro?",
                "bom dia": "☀️ Bom dia! Como posso ajudar você hoje?",
                "boa tarde": "🌅 Boa tarde! Como posso ajudar você hoje?",
                "ajuda": (
                    "🤝 Posso ajudar você com:\n\n"
                    "1. 📊 Análise de mercado de ações\n"
                    "2. 🪙 Informações sobre criptomoedas\n"
                    "3. 📈 Dados de mercado em tempo real\n\n"
                    "O que você gostaria de saber?"
                )
            }
        }
    
    def detect_language(self, text):
        """Detect language based on keywords"""
        text_lower = text.lower()
        
        # French indicators
        if any(word in text_lower for word in ["bonjour", "salut", "qui es-tu", "aide", "bonne"]):
            return "fr"
        
        # Portuguese indicators
        elif any(word in text_lower for word in ["olá", "oi", "quem é você", "ajuda", "bom dia", "boa"]):
            return "pt"
        
        # Default to English
        else:
            return "en"
    
    def normalize_text(self, text):
        """Normalize text for matching"""
        return text.lower().strip()
    
    def get_factual_answer(self, user_input):
        """Check if input matches a factual answer"""
        normalized_input = self.normalize_text(user_input)
        language = self.detect_language(user_input)
        
        # Check in the detected language
        if language in self.factual_answers:
            for key, response in self.factual_answers[language].items():
                if self.normalize_text(key) in normalized_input:
                    return response
        
        # Check all languages
        for lang in self.factual_answers:
            for key, response in self.factual_answers[lang].items():
                if self.normalize_text(key) in normalized_input:
                    return response
        
        return None
    
    def format_prompt(self, user_input, language="en"):
        """Format the prompt based on language"""
        prompts = {
            "en": "You are a helpful AI assistant. Respond in English, clearly and concisely.",
            "fr": "Vous êtes un assistant IA utile. Répondez en français, clairement et brièvement.",
            "pt": "Você é um assistente de IA útil. Responda em português, de forma clara e concisa."
        }
        
        system_prompt = prompts.get(language, prompts["en"])
        return f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
    
    def save_history(self, user_input, response, language):
        """Save conversation to history"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": response,
            "language": language
        }
        
        self.history.append(entry)
        
        # Keep only last 100 entries
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        # Save to file
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def load_history(self):
        """Load conversation history"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
        except:
            self.history = []
    
    def chat(self):
        print("💬 EzioFilho Multilingual AI Chat")
        print("🌍 Languages: English, Français, Português")
        print("📝 Commands: 'exit', 'clear', 'help', 'history'")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                
                # Handle commands
                if user_input.lower() in ['exit', 'quit', 'bye', 'sair', 'quitter']:
                    print("👋 Goodbye! Thanks for chatting!")
                    print("👋 Au revoir! Merci d'avoir discuté!")
                    print("👋 Tchau! Obrigado por conversar!")
                    break
                    
                elif user_input.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("💬 Chat cleared. Continue chatting...")
                    continue
                    
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                    
                elif user_input.lower() in ['help', 'aide', 'ajuda']:
                    self.show_help()
                    continue
                    
                elif not user_input:
                    continue
                
                # Detect language
                language = self.detect_language(user_input)
                
                # Check for factual answer first
                factual_response = self.get_factual_answer(user_input)
                
                if factual_response:
                    print(f"\n🤖 SuperEzio: {factual_response}")
                    self.save_history(user_input, factual_response, language)
                else:
                    # Generate response with AI
                    print(f"\n🤖 SuperEzio ({language.upper()}): ", end="", flush=True)
                    
                    start_time = time.time()
                    formatted_prompt = self.format_prompt(user_input, language)
                    response = self.model.generate(
                        formatted_prompt,
                        max_tokens=150,
                        temperature=0.7
                    )
                    
                    # Clean response
                    response = response.replace(formatted_prompt, "").strip()
                    response = response.split("User:")[0].strip()
                    
                    print(response)
                    
                    # Show generation stats
                    elapsed = time.time() - start_time
                    print(f"\n⚡ Generated in {elapsed:.1f}s")
                    
                    self.save_history(user_input, response, language)
                
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    def show_help(self):
        print("\n📚 HELP MENU / MENU D'AIDE / MENU DE AJUDA:")
        print("=" * 50)
        print("Commands / Commandes / Comandos:")
        print("  exit/quit/sair/quitter - Exit the chat")
        print("  clear - Clear the screen")
        print("  history - Show recent conversations")
        print("  help/aide/ajuda - Show this help menu")
        print("\n🌍 Supported Languages:")
        print("  English - Ask questions in English")
        print("  Français - Posez des questions en français")
        print("  Português - Faça perguntas em português")
        print("\n💡 The system automatically detects your language!")
        print("=" * 50)
    
    def show_history(self):
        print("\n📜 RECENT HISTORY:")
        print("=" * 50)
        if not self.history:
            print("No conversation history yet.")
        else:
            for entry in self.history[-5:]:  # Show last 5
                print(f"\n🕐 {entry['timestamp']} ({entry['language'].upper()})")
                print(f"You: {entry['user']}")
                print(f"SuperEzio: {entry['assistant'][:100]}...")
        print("=" * 50)

def main():
    print("=" * 60)
    print("🎯 EZIOFILHO MULTILINGUAL LLM SYSTEM")
    print("🌍 AI FINANCEIRO MULTILINGUE")
    print("=" * 60)
    
    try:
        chat = MultilingualChat()
        chat.chat()
    except Exception as e:
        print(f"❌ Failed to start: {str(e)}")
        print("\n💡 Make sure llama-cpp-python is installed:")
        print("   pip install llama-cpp-python")

if __name__ == "__main__":
    main()