#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ModelAutoDiscovery - Sistema de Autodescoberta e Configuração para EzioFilho_LLMGraph
-----------------------------------------------------------------------------------
Sistema principal que integra os componentes de descoberta, wrapper e configuração
para modelos de linguagem em aplicações financeiras.

Este script fornece:
1. Interface unificada para descoberta e configuração de modelos
2. Detecção automática de modelos, capacidades e mapeamento para especialistas financeiros
3. Geração de configurações otimizadas para diferentes funções financeiras
4. Gerenciamento inteligente de modelos com carregamento preguiçoso e otimização

Autor: EzioFilho LLMGraph Team
Data: Maio/2025
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union

# Importações dos componentes principais
from core.model_discovery import ModelDiscovery, ModelInfo
from core.universal_model_wrapper import UniversalModelWrapper, get_optimal_wrapper_params
from core.auto_configuration import AutoConfiguration, ExpertConfig
from core.gpu_monitor import get_gpu_monitor, GPUMonitor

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ModelAutoDiscovery")

class ModelAutoDiscovery:
    """
    Sistema principal de Autodescoberta e Configuração para EzioFilho_LLMGraph.
    
    Esta classe integra todos os componentes do sistema:
    - Descoberta automática de modelos
    - Wrapper universal para interface unificada
    - Autoconfiguração para especialistas financeiros
    """
    
    def __init__(
        self,
        base_dirs: Optional[List[Union[str, Path]]] = None,
        config_dir: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        gpu_ids: Optional[List[int]] = None,
        auto_init: bool = True
    ):
        """
        Inicializa o sistema de autodescoberta.
        
        Args:
            base_dirs: Lista de diretórios base para busca de modelos
            config_dir: Diretório para armazenar configurações
            config_path: Caminho para arquivo de configuração existente
            gpu_ids: Lista de IDs de GPUs disponíveis (None para auto-detecção)
            auto_init: Se deve inicializar automaticamente
        """
        # Configura componentes
        self.discovery = None
        self.auto_config = None
        self.config = None
        
        # Diretórios
        self.base_dirs = base_dirs
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self.config_path = Path(config_path) if config_path else None
        
        # Detectar GPUs disponíveis
        self.gpu_ids = gpu_ids
        self._detect_gpus()
        
        # Modelos e especialistas
        self.models = {}  # name -> UniversalModelWrapper
        self.experts_config = {}  # type -> ExpertConfig
        
        # Assegura que o diretório de configuração existe
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicialização automática
        if auto_init:
            self.initialize()
    
    def initialize(self) -> bool:
        """
        Inicializa o sistema completo.
        
        Returns:
            True se inicialização bem-sucedida
        """
        try:
            # Inicia descoberta de modelos
            self._initialize_discovery()
            
            # Verifica se devemos carregar configuração existente
            if self.config_path and self.config_path.exists():
                # Carrega configuração
                return self._load_configuration()
            else:
                # Gera nova configuração
                return self._generate_configuration()
                
        except Exception as e:
            logger.error(f"Erro ao inicializar sistema: {e}")
            return False
            
    def get_gpu_status(self) -> List[Dict[str, Any]]:
        """
        Obtém status atual das GPUs.
        
        Returns:
            Lista com informações das GPUs
        """
        if self.gpu_monitor:
            return self.gpu_monitor.get_gpu_info_summary()
        return []
    
    def _initialize_discovery(self):
        """Inicializa o sistema de descoberta de modelos."""
        start_time = time.time()
        logger.info("Inicializando sistema de descoberta de modelos...")
        
        self.discovery = ModelDiscovery(
            base_dirs=self.base_dirs,
            auto_scan=False,  # Não escaneia automaticamente
            gpu_ids=self.gpu_ids
        )
        
        # Inicializa autoconfiguração
        self.auto_config = AutoConfiguration(
            discovery=self.discovery,
            config_dir=self.config_dir,
            discovery_on_init=False,
            gpu_ids=self.gpu_ids
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Sistema de descoberta inicializado em {elapsed:.2f}s")
    
    def _load_configuration(self) -> bool:
        """
        Carrega configuração existente.
        
        Returns:
            True se carregamento bem-sucedido
        """
        try:
            logger.info(f"Carregando configuração: {self.config_path}")
            self.config = self.auto_config.load_config(self.config_path)
            
            # Carrega modelos e especialistas
            self._load_experts_config()
            logger.info(f"Configuração carregada: {len(self.experts_config)} especialistas configurados")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            
            # Tenta gerar nova configuração como fallback
            logger.info("Tentando gerar nova configuração como fallback...")
            return self._generate_configuration()
    
    def _generate_configuration(self) -> bool:
        """
        Gera nova configuração.
        
        Returns:
            True se geração bem-sucedida
        """
        try:
            start_time = time.time()
            logger.info("Gerando nova configuração...")
            
            # Executa descoberta de modelos
            self.discovery.scan_all()
            
            # Gera configurações
            self.auto_config.generate_model_configs()
            self.auto_config.generate_expert_configs()
            
            # Salva configuração
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            config_filename = f"eziofilho_config_{timestamp}.json"
            config_path = self.auto_config.save_config(config_filename)
            self.config_path = Path(config_path)
            self.config = self.auto_config.load_config(self.config_path)
            
            # Carrega especialistas
            self._load_experts_config()
            
            elapsed = time.time() - start_time
            logger.info(f"Configuração gerada em {elapsed:.2f}s: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao gerar configuração: {e}")
            return False
    
    def _load_experts_config(self):
        """Carrega configurações de especialistas."""
        self.experts_config = {}
        
        for expert_config in self.auto_config.expert_configs:
            if expert_config.enabled:
                self.experts_config[expert_config.type] = expert_config
    
    def discover_models(self, force: bool = False) -> Dict[str, Any]:
        """
        Executa a descoberta de modelos nos caminhos configurados.
        
        Args:
            force: Se deve forçar uma nova descoberta mesmo se já feita
            
        Returns:
            Dicionário com modelos descobertos
        """
        try:
            if force or not self.discovery.models:
                logger.info("Iniciando descoberta de modelos...")
                self.discovery.scan_all()
                
            return self.discovery.models
            
        except Exception as e:
            logger.error(f"Erro ao descobrir modelos: {e}")
            return {}
            
    def configure_models(self) -> bool:
        """
        Configura modelos descobertos para uso com especialistas.
        
        Returns:
            True se configuração bem-sucedida
        """
        try:
            if not self.discovery.models:
                logger.warning("Nenhum modelo descoberto para configurar")
                return False
                
            # Gera configurações para modelos e especialistas
            self.auto_config.generate_model_configs()
            self.auto_config.generate_expert_configs()
            
            # Carrega especialistas
            self._load_experts_config()
            
            # Marca modelos como configurados
            for model_id in self.discovery.models:
                self.discovery.models[model_id]["configured"] = True
                
            logger.info(f"Modelos configurados: {len(self.discovery.models)}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao configurar modelos: {e}")
            return False
            
    def get_discovered_models(self) -> Dict[str, Any]:
        """
        Retorna todos os modelos descobertos.
        
        Returns:
            Dicionário com informações de modelos
        """
        return self.discovery.models
        
    def get_expert_configs(self) -> Dict[str, ExpertConfig]:
        """
        Retorna configurações de especialistas.
        
        Returns:
            Dicionário com configurações de especialistas
        """
        return self.experts_config
    
    def set_search_paths(self, paths: List[Union[str, Path]]) -> None:
        """
        Define caminhos de busca para modelos.
        
        Args:
            paths: Lista de caminhos para busca
        """
        self.discovery.base_dirs = [Path(p) for p in paths]
        
    def set_model_patterns(self, patterns: List[str]) -> None:
        """
        Define padrões de busca para modelos.
        
        Args:
            patterns: Lista de padrões de arquivo
        """
        self.discovery.file_patterns = patterns
        
    def set_recursive_search(self, recursive: bool) -> None:
        """
        Define se a busca deve ser recursiva.
        
        Args:
            recursive: Se deve buscar em subdiretórios
        """
        self.discovery.recursive = recursive
    
    def get_model(self, model_name: str) -> Optional[UniversalModelWrapper]:
        """
        Obtém ou carrega um modelo pelo nome.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            Wrapper do modelo ou None
        """
        # Verifica se já está carregado
        if model_name in self.models:
            return self.models[model_name]
        
        # Procura o modelo nas configurações
        model_config = next((m for m in self.config.get("models", []) 
                          if m["name"] == model_name), None)
        
        if not model_config:
            logger.error(f"Modelo não encontrado: {model_name}")
            return None
        
        try:
            # Cria wrapper para o modelo
            wrapper = UniversalModelWrapper(
                model_path=model_config["path"],
                model_type=model_config["model_type"],
                model_name=model_config["name"],
                capabilities=model_config["capabilities"],
                parameters=model_config["parameters"],
                context_length=model_config["context_length"],
                lazy_load=True  # Carregamento preguiçoso
            )
            
            # Armazena no dicionário
            self.models[model_name] = wrapper
            return wrapper
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {model_name}: {e}")
            return None
    
    def get_expert_model(self, expert_type: str) -> Optional[UniversalModelWrapper]:
        """
        Obtém o modelo para um tipo de especialista.
        
        Args:
            expert_type: Tipo de especialista
            
        Returns:
            Wrapper do modelo ou None
        """
        # Procura configuração do especialista
        expert_config = self.experts_config.get(expert_type)
        
        if not expert_config:
            logger.warning(f"Especialista não configurado: {expert_type}")
            return None
        
        # Obtém modelo primário
        model = self.get_model(expert_config.model)
        
        # Se falhar, tenta modelo de fallback
        if not model and expert_config.fallback_model:
            logger.info(f"Usando modelo de fallback para {expert_type}: {expert_config.fallback_model}")
            model = self.get_model(expert_config.fallback_model)
        
        return model
    
    def get_default_model(self) -> Optional[UniversalModelWrapper]:
        """
        Obtém o modelo padrão do sistema.
        
        Returns:
            Wrapper do modelo padrão ou None
        """
        default_model_name = self.config.get("default_model")
        if not default_model_name:
            # Se não tiver modelo padrão na configuração, usa o primeiro
            models = self.config.get("models", [])
            if models:
                default_model_name = models[0]["name"]
            else:
                return None
        
        return self.get_model(default_model_name)
    
    def get_expert_config(self, expert_type: str) -> Optional[ExpertConfig]:
        """
        Obtém a configuração para um tipo de especialista.
        
        Args:
            expert_type: Tipo de especialista
            
        Returns:
            Configuração do especialista ou None
        """
        return self.experts_config.get(expert_type)
    
    def get_available_experts(self) -> List[str]:
        """
        Obtém a lista de tipos de especialistas disponíveis.
        
        Returns:
            Lista de tipos de especialistas
        """
        return list(self.experts_config.keys())
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Obtém a lista de modelos disponíveis.
        
        Returns:
            Lista de configurações de modelos
        """
        return self.config.get("models", [])
    
    def unload_all_models(self):
        """Descarrega todos os modelos carregados."""
        for model_name, model in list(self.models.items()):
            try:
                model.unload()
            except Exception as e:
                logger.warning(f"Erro ao descarregar modelo {model_name}: {e}")
        
        self.models = {}
    
    def refresh_configuration(self, force_discovery: bool = False) -> bool:
        """
        Atualiza a configuração do sistema.
        
        Args:
            force_discovery: Se deve forçar nova descoberta de modelos
            
        Returns:
            True se atualização bem-sucedida
        """
        # Descarrega modelos atuais
        self.unload_all_models()
        
        if force_discovery:
            # Executa nova descoberta completa
            return self._generate_configuration()
        else:
            # Recarrega configuração existente
            return self._load_configuration()
    
    def generate_expert(
        self, 
        expert_type: str,
        model_name: Optional[str] = None
    ) -> Tuple[Optional[ExpertConfig], Optional[UniversalModelWrapper]]:
        """
        Gera um novo especialista com modelo associado.
        
        Args:
            expert_type: Tipo de especialista
            model_name: Nome do modelo a ser usado (opcional)
            
        Returns:
            Tupla com (configuração, modelo) ou (None, None)
        """
        # Cria configuração
        expert_config = self.auto_config.create_expert_config(
            expert_type=expert_type,
            model_name=model_name
        )
        
        if not expert_config:
            return None, None
        
        # Adiciona à configuração
        self.experts_config[expert_config.type] = expert_config
        
        # Salva configuração atualizada
        self.auto_config.save_config(self.config_path.name)
        
        # Obtém o modelo
        model = self.get_model(expert_config.model)
        
        return expert_config, model

    def _detect_gpus(self):
        """Detecta as GPUs disponíveis no sistema e inicializa o monitor."""
        try:
            if self.gpu_ids is not None:
                logger.info(f"Usando GPUs especificadas: {self.gpu_ids}")
                # Inicializa o monitor com as GPUs especificadas
                self.gpu_monitor = get_gpu_monitor(gpu_ids=self.gpu_ids)
                return
                
            import torch
            
            if not torch.cuda.is_available():
                logger.warning("CUDA não disponível, usando apenas CPU")
                self.gpu_ids = []
                self.gpu_monitor = None
                return
                
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                logger.warning("Nenhuma GPU detectada, usando apenas CPU")
                self.gpu_ids = []
                self.gpu_monitor = None
                return
                
            # Detectar todas as GPUs disponíveis
            gpu_info = []
            for i in range(num_gpus):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                gpu_info.append({
                    "id": i,
                    "name": name,
                    "memory": mem,
                    "score": self._calculate_gpu_score(name, mem)
                })
                logger.info(f"GPU {i}: {name} ({mem:.2f} GB)")
            
            # Ordenar GPUs por pontuação (melhor primeiro)
            gpu_info.sort(key=lambda x: x["score"], reverse=True)
            self.gpu_ids = [gpu["id"] for gpu in gpu_info]
            
            # Detecção específica de modelos RTX 2060 e GTX 1070
            rtx_2060_id = None
            gtx_1070_id = None
            
            for gpu in gpu_info:
                if "2060" in gpu["name"]:
                    rtx_2060_id = gpu["id"]
                    logger.info(f"RTX 2060 detectada como GPU {rtx_2060_id}")
                elif "1070" in gpu["name"]:
                    gtx_1070_id = gpu["id"]
                    logger.info(f"GTX 1070 detectada como GPU {gtx_1070_id}")
            
            # Priorizar RTX 2060 para modelos mais recentes e complexos
            if rtx_2060_id is not None and gtx_1070_id is not None:
                self.gpu_ids = [rtx_2060_id, gtx_1070_id]
                logger.info(f"Usando GPUs na ordem: RTX 2060 (ID: {rtx_2060_id}), GTX 1070 (ID: {gtx_1070_id})")
            
            # Inicializa o monitor de GPU com as GPUs detectadas
            self.gpu_monitor = get_gpu_monitor(gpu_ids=self.gpu_ids)
            
        except Exception as e:
            logger.warning(f"Erro ao detectar GPUs: {e}")
            logger.warning("Usando apenas CPU como fallback")
            self.gpu_ids = []
            self.gpu_monitor = None
    
    def _calculate_gpu_score(self, name: str, memory: float) -> float:
        """
        Calcula uma pontuação para a GPU com base no nome e memória.
        
        Args:
            name: Nome da GPU
            memory: Memória da GPU em GB
            
        Returns:
            Pontuação da GPU (maior é melhor)
        """
        score = memory  # Base: quantidade de memória
        
        # Ajustes baseados na geração da GPU
        if "RTX" in name:
            score += 5.0  # RTX tem Tensor Cores
            
            # Gerações RTX
            if "4090" in name or "4080" in name or "4070" in name:
                score += 4.0  # Série 40
            elif "3090" in name or "3080" in name or "3070" in name:
                score += 3.0  # Série 30
            elif "2080" in name or "2070" in name:
                score += 2.0  # Série 20
            elif "2060" in name:
                score += 1.5  # RTX 2060
        
        # GPUs GTX
        elif "GTX" in name:
            if "1080" in name:
                score += 1.0
            elif "1070" in name:
                score += 0.8
            elif "1060" in name:
                score += 0.6
                
        return score


# Funções auxiliares

def create_auto_discovery_system(
    base_dirs: Optional[List[Union[str, Path]]] = None,
    config_path: Optional[Union[str, Path]] = None
) -> ModelAutoDiscovery:
    """
    Cria e inicializa um sistema de autodescoberta de modelos.
    
    Args:
        base_dirs: Lista de diretórios base para busca de modelos
        config_path: Caminho para arquivo de configuração existente
        
    Returns:
        Instância de ModelAutoDiscovery
    """
    return ModelAutoDiscovery(
        base_dirs=base_dirs,
        config_path=config_path,
        auto_init=True
    )

def get_system_with_experts() -> ModelAutoDiscovery:
    """
    Cria um sistema completo com todos os especialistas configurados.
    
    Returns:
        Instância de ModelAutoDiscovery inicializada
    """
    system = create_auto_discovery_system()
    
    # Pré-carrega configurações de especialistas para garantir disponibilidade
    for expert_type in system.get_available_experts():
        system.get_expert_config(expert_type)
    
    return system


# Uso como script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistema de Autodescoberta e Configuração para EzioFilho_LLMGraph")
    parser.add_argument("--config", "-c", help="Caminho para arquivo de configuração existente")
    parser.add_argument("--dir", "-d", action="append", help="Diretório para busca de modelos")
    parser.add_argument("--generate", "-g", action="store_true", help="Gera nova configuração mesmo se já existir")
    parser.add_argument("--test", "-t", action="store_true", help="Testa todos os modelos e especialistas")
    parser.add_argument("--prompt", "-p", default="Explique o que é uma análise financeira.", 
                      help="Prompt para teste dos modelos")
    parser.add_argument("--verbose", "-v", action="store_true", help="Modo verboso")
    
    args = parser.parse_args()
    
    # Configura logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Cria sistema
    system = ModelAutoDiscovery(
        base_dirs=args.dir,
        config_path=args.config,
        auto_init=not args.generate  # Não inicializa se for gerar nova config
    )
    
    # Gera nova configuração se solicitado
    if args.generate:
        system._generate_configuration()
    
    # Testa modelos e especialistas
    if args.test:
        # Testa modelo padrão
        default_model = system.get_default_model()
        if default_model:
            print(f"\n=== Teste do Modelo Padrão: {default_model.model_name} ===")
            response = default_model.generate(args.prompt, max_tokens=100)
            print(f"Prompt: {args.prompt}")
            print(f"Resposta: {response}")
        
        # Testa especialistas
        experts = system.get_available_experts()
        for expert_type in experts:
            print(f"\n=== Teste do Especialista: {expert_type} ===")
            expert_config = system.get_expert_config(expert_type)
            expert_model = system.get_expert_model(expert_type)
            
            if expert_model:
                print(f"Usando modelo: {expert_model.model_name}")
                
                # Cria prompt específico
                expert_prompt = f"Como especialista em {expert_type}, {args.prompt}"
                
                # Gera resposta
                response = expert_model.generate(expert_prompt, max_tokens=100)
                print(f"Prompt: {expert_prompt}")
                print(f"Resposta: {response}")
            else:
                print(f"Erro: Modelo não disponível para {expert_type}")
        
        # Descarrega todos os modelos ao finalizar
        system.unload_all_models()
    
    # Exibe informações
    available_models = system.get_available_models()
    available_experts = system.get_available_experts()
    
    print(f"\nModelos disponíveis: {len(available_models)}")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model['name']} ({model['model_type']})")
    
    print(f"\nEspecialistas configurados: {len(available_experts)}")
    for i, expert_type in enumerate(available_experts, 1):
        expert_config = system.get_expert_config(expert_type)
        print(f"{i}. {expert_type} - Modelo: {expert_config.model}")
    
    if system.config_path:
        print(f"\nArquivo de configuração: {system.config_path}")
