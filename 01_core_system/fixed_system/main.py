#!/usr/bin/env python3
"""
Sistema AutoGen 1000 - Versão Otimizada
Compatível com pyautogen 0.2.18
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Importações do AutoGen
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
    logger.info(f"✅ AutoGen {autogen.__version__} carregado com sucesso!")
except ImportError as e:
    logger.error(f"❌ Erro ao importar AutoGen: {e}")
    sys.exit(1)

class AutoGenSystem1000:
    """Sistema AutoGen otimizado e funcional"""
    
    def __init__(self):
        self.models_path = Path.home() / ".cache" / "models"
        self.config = self._create_config()
        self.agents = {}
        
    def _create_config(self) -> list:
        """Cria configuração para modelos locais"""
        config_list = []
        
        # Configuração para LM Studio (porta padrão 1234)
        lm_studio_config = {
            "model": "local-model",
            "base_url": "http://localhost:1234/v1",
            "api_key": "not-needed",
            "api_type": "open_ai"
        }
        
        # Configuração para Ollama (porta padrão 11434)
        ollama_config = {
            "model": "mistral",
            "base_url": "http://localhost:11434/v1",
            "api_key": "not-needed",
            "api_type": "open_ai"
        }
        
        # Configuração para API OpenAI (se tiver chave)
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            openai_config = {
                "model": "gpt-3.5-turbo",
                "api_key": openai_key
            }
            config_list.append(openai_config)
            logger.info("✅ OpenAI API configurada")
        
        # Adiciona configurações locais
        config_list.extend([lm_studio_config, ollama_config])
        
        return config_list
    
    def setup_agents(self):
        """Configura os agentes do AutoGen"""
        
        # Configuração LLM
        llm_config = {
            "config_list": self.config,
            "temperature": 0.7,
            "timeout": 120,
            "cache_seed": None  # Desabilita cache para testes
        }
        
        # Assistente AI
        self.agents["assistant"] = AssistantAgent(
            name="assistant",
            llm_config=llm_config,
            system_message="""Você é um assistente AI útil e inteligente.
            Responda de forma clara, concisa e precisa.
            Se não souber algo, diga honestamente."""
        )
        
        # Proxy do usuário
        self.agents["user"] = UserProxyAgent(
            name="user",
            human_input_mode="NEVER",  # Modo automático
            max_consecutive_auto_reply=0,
            code_execution_config=False  # Desabilita execução de código por segurança
        )
        
        logger.info("✅ Agentes configurados com sucesso!")
    
    def chat(self, message: str):
        """Inicia uma conversa com o assistente"""
        try:
            logger.info(f"💬 Enviando mensagem: {message[:50]}...")
            
            # Inicia a conversa
            self.agents["user"].initiate_chat(
                self.agents["assistant"],
                message=message
            )
            
            logger.info("✅ Conversa concluída!")
            
        except Exception as e:
            logger.error(f"❌ Erro durante conversa: {e}")
            
    def test_connection(self):
        """Testa conexão com os modelos"""
        logger.info("🔍 Testando conexões...")
        
        test_message = "Responda apenas 'OK' se você está funcionando."
        
        for i, config in enumerate(self.config):
            logger.info(f"Testando config {i+1}: {config.get('base_url', 'OpenAI API')}")
            
            try:
                # Cria agente temporário para teste
                test_agent = AssistantAgent(
                    name=f"test_agent_{i}",
                    llm_config={
                        "config_list": [config],
                        "timeout": 30
                    }
                )
                
                # Tenta gerar resposta
                response = test_agent.generate_reply(
                    messages=[{"role": "user", "content": test_message}]
                )
                
                if response:
                    logger.info(f"✅ Config {i+1} funcionando!")
                else:
                    logger.warning(f"⚠️ Config {i+1} sem resposta")
                    
            except Exception as e:
                logger.error(f"❌ Config {i+1} falhou: {e}")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Sistema AutoGen 1000")
    parser.add_argument("--test", action="store_true", help="Testa conexões")
    parser.add_argument("--chat", type=str, help="Inicia chat com mensagem")
    parser.add_argument("--interactive", action="store_true", help="Modo interativo")
    
    args = parser.parse_args()
    
    # Cria sistema
    logger.info("🚀 Iniciando Sistema AutoGen 1000...")
    system = AutoGenSystem1000()
    system.setup_agents()
    
    # Executa ação solicitada
    if args.test:
        system.test_connection()
    elif args.chat:
        system.chat(args.chat)
    elif args.interactive:
        logger.info("💬 Modo interativo. Digite 'sair' para encerrar.")
        while True:
            try:
                message = input("\n👤 Você: ")
                if message.lower() in ['sair', 'exit', 'quit']:
                    break
                system.chat(message)
            except KeyboardInterrupt:
                break
        logger.info("👋 Encerrando...")
    else:
        # Teste padrão
        system.chat("Olá! Me explique em 2 linhas o que é inteligência artificial.")

if __name__ == "__main__":
    main()