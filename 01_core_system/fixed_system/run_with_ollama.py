#!/usr/bin/env python3
"""
Sistema AutoGen com Ollama
Mais fácil de configurar e usar!
"""

import os
import subprocess
import time
import logging
from typing import Optional

# Tenta importar bibliotecas
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent
except ImportError:
    print("❌ AutoGen não instalado!")
    print("Execute: pip install pyautogen==0.2.18")
    exit(1)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class OllamaAutoGenSystem:
    """Sistema integrado Ollama + AutoGen"""
    
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/v1"
        
    def check_ollama(self) -> bool:
        """Verifica se Ollama está rodando"""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def start_ollama(self):
        """Tenta iniciar Ollama"""
        logger.info("🚀 Iniciando Ollama...")
        try:
            # Windows
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            time.sleep(3)
            return True
        except:
            return False
    
    def pull_model(self):
        """Baixa modelo se necessário"""
        logger.info(f"📥 Verificando modelo {self.model_name}...")
        try:
            result = subprocess.run(
                ["ollama", "pull", self.model_name],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("✅ Modelo pronto!")
                return True
        except:
            pass
        return False
    
    def setup_autogen(self):
        """Configura AutoGen com Ollama"""
        # Configuração para Ollama
        config_list = [{
            "model": self.model_name,
            "base_url": self.ollama_url,
            "api_key": "ollama",  # Ollama não precisa de API key
            "api_type": "open_ai"
        }]
        
        # Configuração LLM
        llm_config = {
            "config_list": config_list,
            "temperature": 0.7,
            "cache_seed": None
        }
        
        # Cria assistente
        self.assistant = AssistantAgent(
            name="Assistente_1000",
            llm_config=llm_config,
            system_message="""Você é um assistente AI super inteligente e útil.
            Suas respostas são sempre claras, precisas e úteis.
            Você é criativo mas também prático."""
        )
        
        # Cria proxy do usuário
        self.user = UserProxyAgent(
            name="Usuario",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False
        )
        
        logger.info("✅ AutoGen configurado!")
    
    def chat(self, message: str):
        """Conversa com o assistente"""
        logger.info("\n💬 Enviando mensagem...")
        try:
            self.user.initiate_chat(
                self.assistant,
                message=message
            )
        except Exception as e:
            logger.error(f"❌ Erro: {e}")
    
    def run(self):
        """Executa o sistema completo"""
        logger.info("=== SISTEMA AUTOGEN 1000 COM OLLAMA ===\n")
        
        # 1. Verifica/Inicia Ollama
        if not self.check_ollama():
            logger.info("⚠️ Ollama não está rodando.")
            if not self.start_ollama():
                logger.error("❌ Não foi possível iniciar Ollama!")
                logger.info("\n📌 Instale Ollama:")
                logger.info("   https://ollama.ai/download")
                return
        
        logger.info("✅ Ollama está rodando!")
        
        # 2. Baixa modelo se necessário
        if not self.pull_model():
            logger.warning("⚠️ Não foi possível verificar o modelo")
        
        # 3. Configura AutoGen
        self.setup_autogen()
        
        # 4. Loop interativo
        logger.info("\n🎯 Sistema pronto! Digite 'sair' para encerrar.\n")
        
        while True:
            try:
                prompt = input("👤 Você: ").strip()
                if prompt.lower() in ['sair', 'exit', 'quit']:
                    break
                if prompt:
                    self.chat(prompt)
                    print()  # Linha em branco
            except KeyboardInterrupt:
                break
        
        logger.info("\n👋 Até logo!")

def test_quick():
    """Teste rápido do sistema"""
    system = OllamaAutoGenSystem("mistral")
    
    # Verifica componentes
    logger.info("🔍 Verificando componentes...\n")
    
    # AutoGen
    logger.info(f"✅ AutoGen {autogen.__version__}")
    
    # Ollama
    if system.check_ollama():
        logger.info("✅ Ollama rodando")
    else:
        logger.info("❌ Ollama não encontrado")
        logger.info("   Instale: https://ollama.ai/download")
    
    logger.info("\n✨ Para iniciar o sistema:")
    logger.info("   python run_with_ollama.py")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_quick()
    else:
        system = OllamaAutoGenSystem()
        system.run()