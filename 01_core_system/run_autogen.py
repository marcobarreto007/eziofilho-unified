#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sistema AutoGen com Roteador de Modelos Locais
----------------------------------------------
Conecta modelos locais de IA com o framework AutoGen da Microsoft,
permitindo chat, agentes inteligentes e ferramentas LLM locais.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s"
)
logger = logging.getLogger("autogen_system")

# ==============================================================================
# Funções utilitárias
# ==============================================================================

def add_paths():
    """Adiciona diretórios ao caminho Python."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    return current_dir

def import_autogen():
    """Importa AutoGen e confirma disponibilidade."""
    try:
        import autogen
        logger.info(f"AutoGen importado com sucesso: {autogen.__name__}")
        return autogen
    except ImportError as e:
        logger.error(f"Erro ao importar AutoGen: {e}")
        logger.error("Instale com: pip install pyautogen")
        sys.exit(1)

def load_models_config(config_path: str) -> dict:
    """Carrega configuração de modelos do arquivo JSON."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Verifica se tem o formato correto
        if "models" in config:
            model_count = len(config["models"])
            logger.info(f"Configuração carregada: {model_count} modelos encontrados")
        elif isinstance(config, list):
            # Formato antigo (lista direta)
            config = {"models": config}
            logger.info(f"Formato antigo detectado, {len(config['models'])} modelos")
        else:
            logger.warning("Formato de configuração desconhecido")
            config = {"models": []}
        
        return config
    except Exception as e:
        logger.error(f"Erro ao carregar configuração: {e}")
        return {"models": []}

# ==============================================================================
# Classes principais
# ==============================================================================

class ModelManager:
    """Gerencia a carga e acesso a modelos locais."""
    
    def __init__(self, config_path: str, model_filter: Optional[str] = None):
        """
        Inicializa o gerenciador de modelos.
        
        Args:
            config_path: Caminho para arquivo de configuração JSON
            model_filter: Filtro opcional para carregar apenas modelos específicos
        """
        self.config_path = config_path
        self.model_filter = model_filter
        self.config = load_models_config(config_path)
        self.models = {}  # {nome: instância}
        self.initialized = False
    
    def initialize(self):
        """Inicializa os modelos do sistema."""
        if self.initialized:
            return True
        
        try:
            # Importa o wrapper de modelo
            from core.local_model_wrapper import create_model_wrapper
            
            # Filtra modelos se necessário
            model_configs = self.config.get("models", [])
            if self.model_filter:
                filter_lower = self.model_filter.lower()
                model_configs = [m for m in model_configs 
                               if filter_lower in m.get("name", "").lower()]
                logger.info(f"Filtro aplicado: {len(model_configs)} modelos correspondem a '{self.model_filter}'")
            
            if not model_configs:
                logger.warning("Nenhum modelo encontrado na configuração")
                return False
            
            # Inicializa cada modelo
            for model_config in model_configs:
                name = model_config.get("name")
                path = model_config.get("path")
                
                if not name or not path or not os.path.exists(path):
                    logger.warning(f"Modelo inválido ou não encontrado: {name} - {path}")
                    continue
                
                # Cria instância do wrapper
                logger.info(f"Inicializando modelo: {name}")
                wrapper = create_model_wrapper(
                    model_path=path,
                    model_type=model_config.get("type"),
                    context_length=model_config.get("context_length", 4096),
                    temperature=0.7,
                    max_tokens=2048,
                    use_gpu=True
                )
                
                self.models[name] = wrapper
                logger.info(f"✓ Modelo {name} inicializado")
            
            self.initialized = len(self.models) > 0
            if self.initialized:
                logger.info(f"Gerenciador inicializado com {len(self.models)} modelos")
            
            return self.initialized
            
        except Exception as e:
            logger.error(f"Erro ao inicializar modelos: {e}")
            return False
    
    def get_model(self, name: str):
        """Obtém um modelo específico pelo nome."""
        return self.models.get(name)
    
    def get_model_names(self) -> List[str]:
        """Retorna lista de nomes de modelos disponíveis."""
        return list(self.models.keys())
    
    def cleanup(self):
        """Libera recursos de todos os modelos."""
        for name, model in self.models.items():
            try:
                logger.info(f"Descarregando modelo: {name}")
                model.unload()
            except Exception as e:
                logger.warning(f"Erro ao descarregar {name}: {e}")


class ModelRouter:
    """Roteador inteligente de modelos."""
    
    def __init__(self, manager: ModelManager):
        """
        Inicializa o roteador.
        
        Args:
            manager: Gerenciador de modelos contendo os modelos disponíveis
        """
        self.manager = manager
        self.model_capabilities = self._build_capabilities()
        
        # Estado interno
        self.last_used_model = None
        self.default_model = self._select_default_model()
    
    def _build_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Constrói mapa de capacidades para cada modelo."""
        capabilities = {}
        
        for name in self.manager.get_model_names():
            model = self.manager.get_model(name)
            model_path = model.model_path.lower()
            
            # Detecta capacidades baseadas no nome/caminho
            caps = {"general": True}
            
            # Capacidade de chat
            if any(x in model_path for x in ["chat", "instruct", "assistant", "phi", "llama", "falcon"]):
                caps["chat"] = True
            
            # Capacidade de código
            if any(x in model_path for x in ["code", "starcoder", "phi"]):
                caps["code"] = True
            
            # Capacidade matemática
            if any(x in model_path for x in ["math", "qwen"]):
                caps["math"] = True
            
            # Tamanho do modelo (heurística)
            if "7b" in model_path:
                caps["size"] = "medium"
                caps["performance"] = 7
            elif "13b" in model_path:
                caps["size"] = "large"
                caps["performance"] = 8
            elif any(x in model_path for x in ["30b", "33b", "40b", "65b", "70b"]):
                caps["size"] = "xlarge"
                caps["performance"] = 9
            else:
                caps["size"] = "small"
                caps["performance"] = 5
            
            # Boost para modelos específicos de alta qualidade
            if "phi-3" in model_path:
                caps["performance"] += 2
            elif "phi-2" in model_path:
                caps["performance"] += 1
            elif "qwen" in model_path:
                caps["performance"] += 1
            
            capabilities[name] = caps
        
        return capabilities
    
    def _select_default_model(self) -> Optional[str]:
        """Seleciona o modelo padrão baseado nas capacidades."""
        if not self.manager.get_model_names():
            return None
        
        # Prioriza modelos de chat com boa performance
        chat_models = [(name, caps) for name, caps in self.model_capabilities.items() 
                      if caps.get("chat", False)]
        
        if chat_models:
            # Ordena por performance
            chat_models.sort(key=lambda x: x[1].get("performance", 0), reverse=True)
            return chat_models[0][0]
        
        # Se não houver modelos de chat, usa o primeiro disponível
        return self.manager.get_model_names()[0]
    
    def _select_model_for_prompt(self, prompt: str) -> str:
        """
        Seleciona o melhor modelo para o prompt específico.
        
        Args:
            prompt: O prompt de entrada
            
        Returns:
            Nome do modelo selecionado
        """
        # Determina características do prompt
        prompt_lower = prompt.lower()
        
        # Detecta se é uma tarefa de código
        is_code_task = any(x in prompt_lower for x in [
            "código", "codigo", "code", "programa", "function", "função", "programação",
            "python", "javascript", "java", "c++", "html", "css", "sql"
        ])
        
        # Detecta se é uma tarefa matemática
        is_math_task = any(x in prompt_lower for x in [
            "matemática", "calculo", "cálculo", "equação", "equation", "math", 
            "algebra", "álgebra", "calculate", "solve"
        ])
        
        # Detecta se é uma tarefa complexa ou longa
        is_complex = len(prompt) > 500 or "explique detalhadamente" in prompt_lower
        
        # Seleciona modelo apropriado
        candidates = []
        
        if is_code_task:
            # Prioriza modelos com capacidade de código
            candidates = [(name, caps) for name, caps in self.model_capabilities.items() 
                         if caps.get("code", False)]
        elif is_math_task:
            # Prioriza modelos com capacidade matemática
            candidates = [(name, caps) for name, caps in self.model_capabilities.items() 
                         if caps.get("math", False)]
        else:
            # Para tarefas gerais, usa todos os modelos
            candidates = list(self.model_capabilities.items())
        
        if not candidates:
            # Fallback para o modelo padrão
            return self.default_model
        
        # Ajusta pontuação baseada nas características do prompt
        scored_candidates = []
        for name, caps in candidates:
            score = caps.get("performance", 5)
            
            # Bônus para tarefas complexas em modelos maiores
            if is_complex and caps.get("size") in ["large", "xlarge"]:
                score += 2
            
            # Bônus para prompts curtos em modelos pequenos (resposta mais rápida)
            if len(prompt) < 100 and caps.get("size") == "small":
                score += 1
            
            # Sempre prefere modelos de chat para interações conversacionais
            if "chat" in caps and "responda" in prompt_lower:
                score += 1
            
            scored_candidates.append((name, score))
        
        # Seleciona o modelo com maior pontuação
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_model = scored_candidates[0][0]
        
        logger.info(f"Modelo selecionado: {selected_model} (pontuação: {scored_candidates[0][1]})")
        return selected_model
    
    def route_and_generate(self, prompt: str, **kwargs) -> str:
        """
        Seleciona o melhor modelo e gera uma resposta.
        
        Args:
            prompt: Prompt de entrada
            **kwargs: Parâmetros adicionais para geração
            
        Returns:
            Texto gerado pelo modelo selecionado
        """
        if not prompt or not prompt.strip():
            return ""
        
        if not self.manager.get_model_names():
            return "Erro: Nenhum modelo disponível"
        
        # Seleciona o modelo
        model_name = kwargs.get("model_name") or self._select_model_for_prompt(prompt)
        model = self.manager.get_model(model_name)
        
        if not model:
            logger.warning(f"Modelo não encontrado: {model_name}, usando fallback")
            model_name = self.default_model
            model = self.manager.get_model(model_name)
            
            if not model:
                return "Erro: Modelo selecionado não disponível"
        
        # Registra o modelo usado
        self.last_used_model = model_name
        
        # Gera a resposta
        logger.info(f"Gerando resposta com modelo: {model_name}")
        start_time = time.time()
        response = model.generate(prompt, **kwargs)
        elapsed = time.time() - start_time
        
        logger.info(f"Resposta gerada em {elapsed:.2f}s")
        return response


class AutoGenSystem:
    """Sistema para integrar modelos locais com o framework AutoGen."""
    
    def __init__(self, router: ModelRouter):
        """
        Inicializa o sistema AutoGen.
        
        Args:
            router: Roteador de modelos configurado
        """
        self.router = router
        self.autogen = import_autogen()
        self.assistant = None
        self.user_proxy = None
        self.initialized = False
    
    def initialize(self):
        """Configura os agentes do AutoGen."""
        if self.initialized:
            return True
        
        try:
            # Define a função de completion personalizada
            def router_completion(prompt, **kwargs):
                """Função de completion que usa o roteador de modelos."""
                try:
                    response = self.router.route_and_generate(prompt, **kwargs)
                    model_name = self.router.last_used_model or "unknown"
                    
                    return {
                        "content": response,
                        "model": model_name
                    }
                except Exception as e:
                    logger.error(f"Erro na função de completion: {e}")
                    return {
                        "content": f"Erro durante processamento: {str(e)}",
                        "model": "error"
                    }
            
            # Registra o provedor LLM
            self.autogen.register_llm_provider(
                model_type="local-router",
                completion_func=router_completion
            )
            
            # Configuração de LLM para os agentes
            llm_config = {
                "config_list": [{"model": "local-router", "api_key": "not-needed"}],
                "temperature": 0.7,
                "timeout": 300,
                "cache_seed": None,  # Desabilita cache para respostas dinâmicas
            }
            
            # Cria o agente assistente
            self.assistant = self.autogen.AssistantAgent(
                name="Assistente",
                system_message="""Você é um assistente IA útil, preciso e amigável.
Responda sempre em português do Brasil de forma clara e direta.
Use formatação Markdown para melhorar a legibilidade quando apropriado.
Seja útil, conciso e informativo.""",
                llm_config=llm_config
            )
            
            # Cria o agente do usuário
            self.user_proxy = self.autogen.UserProxyAgent(
                name="Usuario",
                human_input_mode="NEVER",  # Não solicita entrada interativa
                is_termination_msg=lambda x: False,  # Não termina automaticamente
                code_execution_config=False,  # Não executa código
            )
            
            self.initialized = True
            logger.info("Sistema AutoGen inicializado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar AutoGen: {e}")
            return False
    
    def run_conversation(self, prompt: str):
        """
        Executa uma conversa usando o prompt fornecido.
        
        Args:
            prompt: Prompt inicial para a conversa
        """
        if not self.initialized:
            self.initialize()
        
        logger.info(f"Iniciando conversa com prompt: '{prompt[:50]}...'")
        
        try:
            # Inicia a conversa
            self.user_proxy.initiate_chat(
                self.assistant,
                message=prompt
            )
            
            logger.info("Conversa concluída com sucesso")
            
        except Exception as e:
            logger.error(f"Erro durante conversa: {e}")
            print(f"\nErro: {str(e)}")

# ==============================================================================
# Função principal
# ==============================================================================

def main():
    """Função principal do script."""
    # Configura argumentos de linha de comando
    parser = argparse.ArgumentParser(description="Sistema de chat com modelos locais e AutoGen")
    parser.add_argument("--config", default="models_config.json", help="Arquivo de configuração dos modelos")
    parser.add_argument("--model", help="Filtro de modelo específico a usar")
    parser.add_argument("--prompt", help="Prompt inicial da conversa")
    args = parser.parse_args()
    
    # Configura ambiente
    add_paths()
    
    # Inicializa gerenciador de modelos
    logger.info("Inicializando gerenciador de modelos...")
    manager = ModelManager(args.config, args.model)
    if not manager.initialize():
        logger.error("Falha ao inicializar modelos")
        return 1
    
    # Configura roteador
    logger.info("Configurando roteador de modelos...")
    router = ModelRouter(manager)
    
    # Configura sistema AutoGen
    logger.info("Configurando sistema AutoGen...")
    system = AutoGenSystem(router)
    if not system.initialize():
        logger.error("Falha ao inicializar sistema AutoGen")
        return 1
    
    # Obtém prompt do usuário se não fornecido
    prompt = args.prompt
    if not prompt:
        print("\n===== Sistema de Chat com Modelos Locais =====")
        print(f"Modelos disponíveis: {', '.join(manager.get_model_names())}")
        print("==============================================\n")
        prompt = input("Digite seu prompt: ")
    
    # Inicia conversa
    try:
        system.run_conversation(prompt)
    finally:
        # Limpa recursos
        manager.cleanup()
    
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