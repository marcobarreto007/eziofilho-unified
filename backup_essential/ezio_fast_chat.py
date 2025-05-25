import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '01_core_system'))

from core.local_model_wrapper import LocalModelWrapper

class EzioFastChat:
    def __init__(self):
        print("🚀 Initializing EzioFilho Fast Chat...")
        print("⏳ Loading model...")
        
        self.model = LocalModelWrapper(
            model_path=r"C:\Users\anapa\.cache\models\phi-2.gguf",
            model_type="gguf",
            chat_enabled=True
        )
        
        # Quick responses for common questions
        self.quick_responses = {
            # Portuguese
            "quem é você": "👋 Olá! Sou SuperEzio, um assistente de IA criado por Marco Barreto. Posso ajudar com análises financeiras e conversas em português, inglês e francês!",
            "qual seu nome": "Meu nome é SuperEzio! 🤖",
            "você fala português": "Sim! Falo português, inglês e francês. Como posso ajudar? 🌍",
            "tudo bem": "Tudo ótimo! E com você? 😊",
            
            # English
            "who are you": "👋 Hello! I'm SuperEzio, an AI assistant created by Marco Barreto. I can help with financial analysis and chat in Portuguese, English, and French!",
            "what is your name": "My name is SuperEzio! 🤖",
            "do you speak portuguese": "Yes! I speak Portuguese, English, and French. How can I help? 🌍",
            "hello": "Hello! How can I help you today? 😊",
            
            # French
            "qui es-tu": "👋 Bonjour! Je suis SuperEzio, un assistant IA créé par Marco Barreto. Je peux aider avec l'analyse financière et discuter en portugais, anglais et français!",
            "comment tu t'appelles": "Je m'appelle SuperEzio! 🤖",
            "parles-tu portugais": "Oui! Je parle portugais, anglais et français. Comment puis-je aider? 🌍"
        }
        
        print("✅ System ready!")
        print("💡 Tip: I respond faster to common questions!")
        print("=" * 60)
    
    def normalize_text(self, text):
        """Normalize text for matching"""
        # Remove punctuation and extra spaces
        import re
        text = re.sub(r'[?!.,;:]', '', text.lower())
        text = ' '.join(text.split())
        return text
    
    def get_quick_response(self, user_input):
        """Check for quick responses first"""
        normalized = self.normalize_text(user_input)
        
        # Check exact matches
        for key, response in self.quick_responses.items():
            if self.normalize_text(key) == normalized:
                return response
        
        # Check partial matches
        for key, response in self.quick_responses.items():
            if self.normalize_text(key) in normalized or normalized in self.normalize_text(key):
                return response
        
        return None
    
    def chat(self):
        print("💬 SuperEzio Chat - Fast Mode")
        print("🌍 Languages: PT, EN, FR")
        print("Type 'exit' to quit")
        print("=" * 60)
        
        while True:
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'sair']:
                print("👋 Goodbye! Tchau! Au revoir!")
                break
            
            if not user_input:
                continue
            
            # Try quick response first
            quick_response = self.get_quick_response(user_input)
            
            if quick_response:
                print(f"🤖 SuperEzio: {quick_response}")
                print("⚡ Instant response!")
            else:
                # Use AI for complex questions
                print("🤖 SuperEzio: ", end="", flush=True)
                
                start_time = time.time()
                
                # Add context to prevent "OpenAI" responses
                prompt = f"""You are SuperEzio, an AI assistant created by Marco Barreto. 
                Never say you are OpenAI or ChatGPT.
                Respond briefly and helpfully.
                
                User: {user_input}
                SuperEzio:"""
                
                try:
                    response = self.model.generate(
                        prompt,
                        max_tokens=100,  # Shorter for faster responses
                        temperature=0.7
                    )
                    
                    # Clean response
                    response = response.replace(prompt, "").strip()
                    response = response.split("User:")[0].strip()
                    
                    print(response)
                    
                    elapsed = time.time() - start_time
                    print(f"⚡ Generated in {elapsed:.1f}s")
                    
                except Exception as e:
                    print(f"Sorry, I had an error: {str(e)}")

def main():
    print("=" * 60)
    print("🎯 EZIOFILHO FAST CHAT SYSTEM")
    print("=" * 60)
    
    chat = EzioFastChat()
    chat.chat()

if __name__ == "__main__":
    main()