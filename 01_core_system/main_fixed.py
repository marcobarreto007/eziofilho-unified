#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integração AutoGen com Roteador de Modelos
------------------------------------------
Sistema que integra AutoGen com um roteador inteligente de modelos locais.
Suporta modelos nos formatos GGUF (llama.cpp) e Transformers (HuggingFace).
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union, Set

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s"
)
logger = logging.getLogger("autogen_main")

# ===========================================================================
# Funções utilitárias
# ===========================================================================

def require(condition: bool, msg: str) -> None:
    """Verifica uma condição e lança um erro se falsa."""
    if not condition:
        raise RuntimeError(msg)

def setup_project() -> str:
    """
    Configura o ambiente do projeto e retorna o diretório raiz.
    """
    # Encontra o diretório do projeto
    project_dir = os.path.abspath(os.path.dirname(__file__))
    logger.info(f"Diretório do projeto: {project_dir}")
    
    # Adiciona diretório do projeto ao caminho de busca Python
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    
    # Lista os arquivos Python em core/
    core_dir = os.path.join(project_dir, "core")
    if os.path.exists(core_dir) and os.path.isdir(core_dir):
        py_files = [f for f in os.listdir(core_dir) if f.endswith(".py")]
        logger.info(f"Arquivos Python em core/: {py_files}")
    
    return project_dir

def import_autogen() -> Any:
    """
    Importa e verifica a instalação do AutoGen.
    """
    try:
        import autogen
        logger.info(f"AutoGen encontrado: {autogen.__name__}")
        return autogen
    except ImportError:
        logger.error("AutoGen não encontrado! Instale com 'pip install pyautogen'")
        raise

def import_modules() -> tuple:
    """
    Importa os módulos core necessários (model_router e local_model_wrapper).
    """
    try:
        from core.model_router import ModelRouter
        from core.local_model_wrapper import LocalModelWrapper
        logger.info("✓ Módulos importados com sucesso!")
        return ModelRouter, LocalModelWrapper
    except ImportError as e:
        logger.error(f"Erro ao importar módulos core: {e}")
        try:
            # Tenta resolver problemas de importação
            from fix_imports import fix_imports
            fix_imports()
            from core.model_router import ModelRouter
            from core.local_model_wrapper import LocalModelWrapper
            logger.info("✓ Módulos importados após correção!")
            return ModelRouter, LocalModelWrapper
        except Exception as e2:
            logger.error(f"Falha mesmo após tentar correção: {e2}")
            raise

# ===========================================================================
# Funções para contrução do roteador de modelos
# ===========================================================================

def build_router(config_path: str) -> Any:
    """
    Constrói o roteador de modelos a partir de um arquivo de configuração.
    
    Args:
        config_path: Caminho para o arquivo de configuração JSON
        
    Returns:
        Instância de ModelRouter configurada
    """
    ModelRouter, LocalModelWrapper = import_modules()
    
    logger.info("Construindo roteador de modelos...")
    
    # Carrega o arquivo de configuração
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Erro ao carregar configuração de {config_path}: {e}")
        # Se o arquivo não for encontrado, tentamos criar uma configuração padrão
        fallback_config = {
            "models": [
                {
                    "name": "phi2",
                    "path": str(Path.home() / ".cache" / "models" / "phi-2.gguf"),
                    "type": "gguf",
                    "context_length": 4096,
                    "capabilities": ["general"],
                    "performance": 5,
                    "priority": 70
                },
                {
                    "name": "mistral",
                    "path": str(Path.home() / ".cache" / "models" / "mistral-7b.gguf"),
                    "type": "gguf",
                    "context_length": 8192,
                    "capabilities": ["general", "chat"],
                    "performance": 7,
                    "priority": 50
                }
            ]
        }
        config_data = fallback_config
    
    # Verifica se temos a lista de modelos na configuração
    model_configs = config_data.get("models", [])
    if not model_configs and isinstance(config_data, list):
        # Compatibilidade com formato antigo (lista direta de modelos)
        model_configs = config_data
        
    # Valida cada configuração de modelo
    valid_configs = []
    for model_config in model_configs:
        model_name = model_config.get("name")
        model_path = model_config.get("path")
        model_type = model_config.get("type", "")
        
        if not model_name or not model_path:
            logger.warning(f"Configuração de modelo inválida: {model_config}")
            continue
            
        # Verifica se o arquivo/diretório do modelo existe
        path = Path(model_path)
        if not path.exists():
            logger.warning(f"✗ Modelo não encontrado: {model_name} em {model_path}")
            continue
            
        valid_configs.append(model_config)
    
    # Garante que temos pelo menos um modelo válido
    require(len(valid_configs) > 0, "No valid model files found")
    
    # Inicializa os wrappers de modelo e o roteador
    model_wrappers = {}
    for config in valid_configs:
        model_name = config["name"]
        model_path = config["path"]
        model_type = config["type"]
        
        try:
            logger.info(f"Inicializando modelo: {model_name}")
            wrapper = LocalModelWrapper(
                model_path=model_path, 
                model_type=model_type,
                context_length=config.get("context_length", 4096),
                allow_downloads=False  # Não permitir downloads durante inicialização
            )
            model_wrappers[model_name] = wrapper
        except Exception as e:
            logger.error(f"Erro ao inicializar modelo {model_name}: {e}")
    
    # Garante que temos pelo menos um modelo inicializado
    require(len(model_wrappers) > 0, "No models could be initialized")
    
    # Cria o roteador com os modelos configurados
    router = ModelRouter(
        models=model_wrappers,
        capabilities={name: config.get("capabilities", ["general"]) 
                    for name, config in zip([m["name"] for m in valid_configs], valid_configs)},
        performance_scores={name: config.get("performance", 5) 
                          for name, config in zip([m["name"] for m in valid_configs], valid_configs)},
        default_model=list(model_wrappers.keys())[0]  # Primeiro modelo como padrão
    )
    
    return router

# ===========================================================================
# Integração com AutoGen
# ===========================================================================

def setup_autogen_agent(router, autogen_module, max_consecutive_auto_reply=10):
    """
    Configura um agente AutoGen que utiliza o roteador de modelos.
    
    Args:
        router: Instância de ModelRouter
        autogen_module: Módulo AutoGen importado
        max_consecutive_auto_reply: Limite de respostas automáticas consecutivas
        
    Returns:
        Instância de agente AutoGen configurada
    """
    # Função de completion que utilizará o roteador
    def router_completion(
        prompt: str,
        context: Any = None, 
        model: Optional[str] = None,
        **kwargs
    ) -> dict:
        """Função de completion personalizada que utiliza o roteador de modelos."""
        try:
            # Se um modelo específico foi solicitado e está disponível, use-o
            if model and model in router.models:
                response = router.models[model].generate(prompt)
            else:
                # Caso contrário, use o roteador para selecionar o melhor modelo
                response = router.route_and_generate(prompt)
                
            return {"content": response, "model": router.last_used_model}
        except Exception as e:
            logger.error(f"Erro na completion: {e}")
            return {"content": f"Erro no processamento da solicitação: {str(e)}", "model": "error"}
    
    # Configuração do agente
    config = {
        "llm_config": {
            "config_list": [{"model": "local-router", "api_key": "not-needed"}],
            "temperature": 0.7,
            "max_tokens": 2000,
            "cache_seed": None,  # Desativa cache para permitir respostas dinâmicas
            "timeout": 120,  # 2 minutos de timeout
        },
        "human_input_mode": "TERMINATE",  # Termina conversa quando o humano intervém
        "max_consecutive_auto_reply": max_consecutive_auto_reply,
    }
    
    # Registra a função de completion personalizada
    autogen_module.register_llm_provider(
        model_type="local-router",
        completion_func=router_completion
    )
    
    # Cria o agente usando a configuração
    agent = autogen_module.AssistantAgent(
        name="Generalist",
        system_message="""Você é um assistente IA útil e versátil.
Responda em português do Brasil de forma clara, precisa e concisa.
Quando a resposta for extensa, use formatação em Markdown para melhorar a legibilidade.
""",
        llm_config=config["llm_config"],
        human_input_mode=config["human_input_mode"],
        max_consecutive_auto_reply=config["max_consecutive_auto_reply"],
    )
    
    return agent

def chat_with_agent(agent, autogen_module, prompt: str):
    """
    Inicia uma conversa com o agente usando o prompt fornecido.
    
    Args:
        agent: Agente AutoGen configurado
        autogen_module: Módulo AutoGen importado
        prompt: Prompt inicial para a conversa
    """
    # Criar agente de usuário
    user = autogen_module.UserProxyAgent(
        name="Usuario",
        human_input_mode="NEVER",  # Não solicitar entrada do usuário
        is_termination_msg=lambda x: False,  # Nunca terminar por conta própria
        code_execution_config=False,  # Desabilitar execução de código
    )
    
    # Iniciar a conversa
    user.initiate_chat(agent, message=prompt)

# ===========================================================================
# Função principal
# ===========================================================================

def main():
    """Função principal do script."""
    parser = argparse.ArgumentParser(description="Sistema AutoGen com Roteador de Modelos")
    parser.add_argument("--config", default="default_config.json", help="Arquivo de configuração de modelos")
    parser.add_argument("--prompt", help="Prompt para enviar ao sistema")
    parser.add_argument("--self-test", action="store_true", help="Executar auto-teste")
    parser.add_argument("--debug", action="store_true", help="Ativar modo de depuração")
    
    ns = parser.parse_args()
    
    # Configura o nível de logging se em modo debug
    if ns.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Executa auto-teste se solicitado
    if ns.self_test:
        logger.info("Executando auto-teste...")
        
        # Verifica instalação do AutoGen
        autogen = import_autogen()
        
        # Configura o ambiente do projeto
        project_dir = setup_project()
        
        # Tenta importar os módulos core
        ModelRouter, LocalModelWrapper = import_modules()
        
        logger.info("Self-test OK")
        
        # Encerra se for apenas self-test
        if not ns.prompt:
            return 0
    
    try:
        # Importa o AutoGen
        autogen = import_autogen()
        
        # Configura o ambiente do projeto
        project_dir = setup_project()
        
        # Se não há um prompt, exibe o menu interativo
        if not ns.prompt:
            ns.prompt = input("\nDigite seu prompt (ou 'sair' para encerrar): ")
            if ns.prompt.lower() in ["sair", "exit", "quit"]:
                return 0
        
        # Constrói o roteador de modelos
        router = build_router(ns.config)
        
        # Configura o agente AutoGen
        agent = setup_autogen_agent(router, autogen)
        
        # Inicia a conversa
        chat_with_agent(agent, autogen, ns.prompt)
        
        return 0
        
    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        if ns.debug:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())