#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sistema de Chat com Roteador Inteligente de Modelos
---------------------------------------------------
Este script integra o LocalModelWrapper com o AutoGen para criar
um sistema de chat alimentado por modelos locais.

Uso:
  python run_system.py --config models_config.json --prompt "Seu prompt aqui"
  python run_system.py --model phi-3-mini --prompt "Seu prompt aqui"
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s"
)
logger = logging.getLogger("model_system")

# ==============================================================================
# Funções de utilidade e preparação do ambiente
# ==============================================================================

def setup_environment():
    """Prepara o ambiente para execução."""
    # Adiciona o diretório atual ao PYTHONPATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Configura caminhos importantes
    core_dir = os.path.join(current_dir, "core")
    if os.path.exists(core_dir) and os.path.isdir(core_dir):
        logger.info(f"Diretório core encontrado: {core_dir}")
    else:
        logger.warning(f"Diretório core não encontrado: {core_dir}")
    
    return current_dir

def import_autogen():
    """Importa a biblioteca AutoGen."""
    try:
        import autogen
        logger.info(f"AutoGen importado: {autogen.__version__}")
        return autogen
    except ImportError as e:
        logger.error(f"Erro ao importar AutoGen: {e}")
        logger.info("Instale com: pip install pyautogen")
        sys.exit(1)

def import_wrapper_and_router():
    """Importa o LocalModelWrapper e ModelRouter."""
    try:
        from core.local_model_wrapper import LocalModelWrapper
        logger.info("LocalModelWrapper importado com sucesso")
        
        try:
            from core.model_router import ModelRouter
            logger.info("ModelRouter importado com sucesso")
            return LocalModelWrapper, ModelRouter
        except ImportError as e:
            logger.error(f"Erro ao importar ModelRouter: {e}")
            
            # Implementação básica de ModelRouter se a importação falhar
            class SimpleModelRouter:
                def __init__(self, models, **kwargs):
                    self.models = models
                    self.last_used_model = None
                    self.default_model = list(models.keys())[0] if models else None
                
                def route_and_generate(self, prompt, **kwargs):
                    """Seleciona modelo e gera resposta."""
                    # Seleciona o melhor modelo (simplesmente o primeiro disponível)
                    model_name = self.default_model
                    self.last_used_model = model_name
                    
                    if not self.models or not model_name:
                        return f"Erro: Nenhum modelo disponível para processar: '{prompt[:30]}...'"
                    
                    # Usa o modelo para gerar resposta
                    return self.models[model_name].generate(prompt, **kwargs)
            
            logger.info("Usando implementação básica de ModelRouter")
            return LocalModelWrapper, SimpleModelRouter
        
    except ImportError as e:
        logger.error(f"Erro crítico ao importar LocalModelWrapper: {e}")
        logger.error("Verifique se o arquivo local_model_wrapper.py existe no diretório core/")
        sys.exit(1)

# ==============================================================================
# Funções para carregamento e gestão de modelos
# ==============================================================================

def load_config(config_path):
    """Carrega a configuração de modelos."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # Verifica formato da configuração
        if "models" in config:
            logger.info(f"Configuração carregada: {len(config['models'])} modelos encontrados")
            return config
        elif isinstance(config, list):
            logger.info(f"Configuração carregada (formato antigo): {len(config)} modelos encontrados")
            return {"models": config}
        else:
            logger.error(f"Formato de configuração não reconhecido")
            return {"models": []}
            
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Erro ao carregar configuração de {config_path}: {e}")
        return {"models": []}

def initialize_models(config, model_filter=None):
    """
    Inicializa os modelos definidos na configuração.
    
    Args:
        config: Configuração com lista de modelos
        model_filter: Nome específico de modelo para carregar
        
    Returns:
        Dicionário de {nome_modelo: instância_modelo}
    """
    LocalModelWrapper, _ = import_wrapper_and_router()
    
    models = {}
    valid_models = config.get("models", [])
    
    # Filtra por nome específico se solicitado
    if model_filter:
        valid_models = [m for m in valid_models if model_filter.lower() in m.get("name", "").lower()]
        if not valid_models:
            logger.error(f"Nenhum modelo encontrado com o nome '{model_filter}'")
            return {}
    
    # Inicializa cada modelo
    for model_config in valid_models:
        name = model_config.get("name")
        path = model_config.get("path")
        model_type = model_config.get("type")
        
        if not name or not path:
            logger.warning(f"Configuração inválida para modelo: {model_config}")
            continue
        
        # Verifica se o caminho existe
        if not os.path.exists(path):
            logger.warning(f"Caminho não encontrado para modelo {name}: {path}")
            continue
        
        try:
            logger.info(f"Inicializando modelo: {name} ({model_type})")
            
            # Configura parâmetros específicos para o modelo
            params = {
                "model_path": path,
                "model_type": model_type,
                "context_length": model_config.get("context_length", 4096),
                "temperature": 0.7,
                "top_p": 0.95,
                "use_gpu": True,  # Tenta usar GPU por padrão
                "trust_remote_code": True,  # Necessário para alguns modelos
            }
            
            # Inicializa o modelo
            model = LocalModelWrapper(**params)
            models[name] = model
            
            logger.info(f"✓ Modelo {name} inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar modelo {name}: {e}")
    
    return models

def setup_router(models):
    """Sets up the model router with available models."""
    # Import the necessary components
    _, ModelRouter = import_wrapper_and_router()
    from core.model_router import create_model_router
    
    # Create model configurations for the router
    model_configs = []
    
    for name, wrapper in models.items():
        # Determine capabilities based on model name
        capabilities = ["general"]  # Default capability
        
        if "phi" in name.lower():
            capabilities.extend(["fast", "creative"])
        elif "mistral" in name.lower():
            capabilities.extend(["precise", "code"])
        elif "dialogpt" in name.lower():
            capabilities.extend(["creative", "conversation"])
        
        # Create model configuration
        config = {
            "name": name,
            "path": wrapper.model_path if hasattr(wrapper, 'model_path') else name,
            "model_type": wrapper.model_type if hasattr(wrapper, 'model_type') else "gguf",
            "capabilities": capabilities,
            "min_prompt_tokens": 0,
            "max_prompt_tokens": 2048
        }
        
        model_configs.append(config)
    
    # Create router using the factory function
    router = create_model_router(
        model_configs=model_configs,
        default_model=list(models.keys())[0] if models else None
    )
    
    logger.info(f"Router configured with {len(models)} models")
    return router
# ==============================================================================
# Integração com AutoGen
# ==============================================================================

def setup_autogen_agents(router):
    """
    Configura os agentes AutoGen com o roteador.
    
    Args:
        router: Roteador de modelos
        
    Returns:
        Tupla (agente_assistente, agente_usuário)
    """
    autogen = import_autogen()
    
    # Função de LLM que usa o roteador
    def router_llm_completion(prompt, **kwargs):
        """Função de completion usando o roteador."""
        try:
            response = router.route_and_generate(prompt)
            model_name = router.last_used_model or "unknown"
            logger.info(f"Resposta gerada usando modelo: {model_name}")
            return {"content": response, "model": model_name}
        except Exception as e:
            logger.error(f"Erro durante a geração: {e}")
            return {"content": f"Erro: {str(e)}", "model": "error"}
    
    # Registra o provedor LLM personalizado
    autogen.register_llm_provider(
        model_type="local-router",
        completion_func=router_llm_completion
    )
    
    # Configura o agente assistente
    assistant = autogen.AssistantAgent(
        name="Assistente",
        system_message="""Você é um assistente IA útil e versátil.
Responda em português do Brasil de forma clara, precisa e concisa.
Quando a resposta for extensa, use formatação em Markdown para melhorar a legibilidade.
Cite fontes quando necessário e seja útil com exemplos relevantes.
""",
        llm_config={
            "config_list": [{"model": "local-router", "api_key": "not-needed"}],
            "temperature": 0.7,
            "cache_seed": None,  # Desativa cache para permitir respostas dinâmicas
        }
    )
    
    # Configura o agente usuário
    user = autogen.UserProxyAgent(
        name="Usuario",
        human_input_mode="NEVER",  # Não pede entrada interativa, responde só uma vez
        is_termination_msg=lambda x: False,  # Não encerra automaticamente
    )
    
    return assistant, user

def run_chat(assistant, user, prompt):
    """
    Executa uma conversa entre usuário e assistente.
    
    Args:
        assistant: Agente assistente AutoGen
        user: Agente usuário AutoGen
        prompt: Prompt inicial para a conversa
    """
    # Inicia a conversa com o prompt
    user.initiate_chat(
        assistant,
        message=prompt
    )

# ==============================================================================
# Função principal
# ==============================================================================

def main():
    """Função principal do script."""
    parser = argparse.ArgumentParser(description="Sistema de chat com modelos locais")
    parser.add_argument("--config", default="models_config.json",
                       help="Arquivo de configuração dos modelos")
    parser.add_argument("--model", help="Nome específico do modelo a usar")
    parser.add_argument("--prompt", help="Prompt inicial para a conversa")
    args = parser.parse_args()
    
    # Configura ambiente
    setup_environment()
    
    # Carrega configuração
    config = load_config(args.config)
    
    # Inicializa modelos
    models = initialize_models(config, args.model)
    if not models:
        logger.error("Nenhum modelo pôde ser inicializado. Verifique a configuração.")
        return 1
    
    # Configura roteador
    router = setup_router(models)
    if not router:
        logger.error("Falha ao configurar roteador de modelos.")
        return 1
    
    # Configura agentes AutoGen
    assistant, user = setup_autogen_agents(router)
    
    # Se não há prompt, solicita interativamente
    prompt = args.prompt
    if not prompt:
        prompt = input("\nDigite seu prompt: ")
    
    # Executa chat
    logger.info(f"Iniciando conversa com prompt: {prompt[:50]}...")
    run_chat(assistant, user, prompt)
    
    # Limpa recursos
    for model in models.values():
        try:
            model.unload()
        except:
            pass
    
    return 0

if __name__ == "__main__":
    sys.exit(main())