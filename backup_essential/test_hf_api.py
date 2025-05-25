#!/usr/bin/env python3
"""
Teste da API HuggingFace com AutoGen
"""

import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_huggingface_api():
    """Testa conexão com HuggingFace API"""
    try:
        import autogen
        
        # Token HuggingFace configurado
        HF_TOKEN = "os.getenv("HUGGINGFACE_TOKEN", "your_token_here")"
        
        # Configuração para HuggingFace
        config_list = [{
            "model": "microsoft/DialoGPT-medium",
            "api_key": HF_TOKEN,
            "base_url": "https://api-inference.huggingface.co/models",
            "api_type": "huggingface"
        }]
        
        # Criar agentes
        assistant = autogen.AssistantAgent(
            name="assistant",
            system_message="Você é um assistente útil.",
            llm_config={
                "config_list": config_list,
                "temperature": 0.7,
                "timeout": 30
            }
        )
        
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False
        )
        
        logger.info("Testando HuggingFace API...")
        
        # Teste simples
        user_proxy.initiate_chat(
            assistant,
            message="Olá! Diga apenas 'funcionando' se você conseguir me responder."
        )
        
        logger.info("✅ Teste concluído!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no teste: {e}")
        return False

if __name__ == "__main__":
    test_huggingface_api()