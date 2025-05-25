#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sistema Simplificado de Chat com Modelos Locais
----------------------------------------------
Um sistema de chat simples que usa modelos locais através do AutoGen.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s"
)
logger = logging.getLogger("autogen_system")

def setup_environment():
    """Configura o ambiente Python."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    return current_dir

def import_autogen():
    """Importa o framework AutoGen."""
    try:
        import autogen
        logger.info(f"AutoGen importado com sucesso")
        return autogen
    except ImportError as e:
        logger.error(f"Erro ao importar AutoGen: {e}")
        sys.exit(1)

def load_models_config(config_path):
    """Carrega a configuração dos modelos."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if "models" in config:
            logger.info(f"Configuração carregada: {len(config['models'])} modelos")
            return config
        elif isinstance(config, list):
            config = {"models": config}
            logger.info(f"Formato antigo detectado: {len(config['models'])} modelos")
            return config
        else:
            logger.warning("Formato de configuração não reconhecido")
            return {"models": []}
    except Exception as e:
        logger.error(f"Erro ao carregar configuração: {e}")
        return {"models": []}

def setup_completion_function(model_path):
    """Configura uma função de completion simples usando um único modelo."""
    try:
        # Importamos diretamente do local_model_wrapper.py
        from core.local_model_wrapper import LocalModelWrapper
        
        # Inicializamos um único modelo para simplificar
        logger.info(f"Inicializando modelo: {os.path.basename(model_path)}")
        model = LocalModelWrapper(
            model_path=model_path,
            temperature=0.7,
            max_tokens=2048,
            use_gpu=True
        )
        
        def model_completion(prompt, **kwargs):
            """Função de completion simplificada."""
            try:
                logger.info(f"Gerando resposta para prompt: '{prompt[:30]}...'")
                start_time = time.time()
                response = model.generate(prompt)
                elapsed = time.time() - start_time
                logger.info(f"Resposta gerada em {elapsed:.2f}s")
                return {"content": response, "model": os.path.basename(model_path)}
            except Exception as e:
                logger.error(f"Erro na geração: {e}")
                return {"content": f"Erro: {str(e)}", "model": "error"}
        
        return model_completion
    except Exception as e:
        logger.error(f"Erro ao configurar função de completion: {e}")
        return None

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Sistema simplificado de chat")
    parser.add_argument("--config", default="models_config.json", help="Arquivo de configuração")
    parser.add_argument("--model", help="Nome específico do modelo")
    parser.add_argument("--prompt", help="Prompt inicial")
    args = parser.parse_args()
    
    # Configura ambiente
    setup_environment()
    
    # Carrega configuração
    config = load_models_config(args.config)
    if not config["models"]:
        logger.error("Nenhum modelo encontrado na configuração")
        return 1
    
    # Seleciona um modelo (o primeiro ou o especificado)
    model_config = None
    if args.model:
        # Busca pelo nome especificado
        for m in config["models"]:
            if args.model.lower() in m.get("name", "").lower():
                model_config = m
                break
        
        if not model_config:
            logger.warning(f"Modelo '{args.model}' não encontrado, usando o primeiro disponível")
            model_config = config["models"][0]
    else:
        # Usa o primeiro modelo disponível
        model_config = config["models"][0]
    
    model_path = model_config.get("path")
    model_name = model_config.get("name")
    
    logger.info(f"Usando modelo: {model_name} ({model_path})")
    
    # Verifica se o modelo existe
    if not os.path.exists(model_path):
        logger.error(f"Arquivo/diretório de modelo não encontrado: {model_path}")
        return 1
    
    # Configura função de completion
    completion_func = setup_completion_function(model_path)
    if not completion_func:
        logger.error("Falha ao configurar função de completion")
        return 1
    
    # Importa AutoGen
    autogen = import_autogen()
    
    # Registra a função personalizada
    autogen.register_llm_provider(
        model_type="local-model",
        completion_func=completion_func
    )
    
    # Configura agente assistente
    assistant = autogen.AssistantAgent(
        name="Assistente",
        system_message="""Você é um assistente IA útil e versátil.
Responda em português do Brasil de forma clara, precisa e concisa.
Quando a resposta for extensa, use formatação em Markdown para melhorar a legibilidade.""",
        llm_config={
            "config_list": [{"model": "local-model", "api_key": "not-needed"}],
            "temperature": 0.7,
        }
    )
    
    # Configura agente usuário
    user = autogen.UserProxyAgent(
        name="Usuario",
        human_input_mode="NEVER",
        is_termination_msg=lambda x: False,
        code_execution_config=False,
    )
    
    # Obtém o prompt se não fornecido
    prompt = args.prompt
    if not prompt:
        print("\n===== Sistema de Chat com Modelo Local =====")
        print(f"Modelo: {model_name}")
        print("============================================\n")
        prompt = input("Digite seu prompt: ")
    
    # Inicia a conversa
    logger.info(f"Iniciando conversa com prompt: '{prompt[:50]}...'")
    try:
        user.initiate_chat(
            assistant,
            message=prompt
        )
        logger.info("Conversa concluída com sucesso")
    except Exception as e:
        logger.error(f"Erro durante conversa: {e}")
        print(f"\nErro: {str(e)}")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Operação interrompida pelo usuário")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erro não tratado: {e}")
        sys.exit(1)