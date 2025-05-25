#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phi2Expert - Classe base para especialistas baseados em Phi-2
--------------------------------------------------------------
Fornece a estrutura básica para os especialistas financeiros baseados em Phi-2,
com otimização para os diferentes tipos de GPU (RTX 2060 e GTX 1070).

Autor: EzioFilho LLMGraph Team
Data: Maio/2025
"""

import os
import sys
import json
import time
import logging
import uuid
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional, Union, Callable

# Adicionar diretório pai ao path para importações relativas
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Importar componentes do sistema
from core.unified_base_expert import EzioBaseExpert
from core.universal_model_wrapper import UniversalModelWrapper
from core.multi_gpu_manager import get_multi_gpu_manager
from core.gpu_monitor import get_gpu_monitor

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Phi2Expert")

class Phi2Expert(EzioBaseExpert):
    """
    Especialista base para modelos Phi-2, especializado em tarefas financeiras.
    
    Fornece funcionalidades específicas para:
    - Otimização para GPUs RTX 2060 e GTX 1070
    - Alocação inteligente via MultiGPUManager
    - Formatos de entrada e saída padronizados para análise financeira
    """
    
    # Versão do framework de especialistas Phi-2
    VERSION = "1.0.0"
    
    # Modelo base usado por todos os especialistas Phi-2
    MODEL_NAME = "microsoft/phi-2"
    DEFAULT_QUANTIZATION = "6bit"  # Quantização padrão para melhor balancear desempenho vs qualidade
    
    def __init__(self, 
                expert_type: str,
                domain: str,
                specialization: str,
                config_path: Optional[Union[str, Path]] = None,
                gpu_id: Optional[int] = None,
                gpu_ids: Optional[List[int]] = None,
                model_path_override: Optional[str] = None,
                system_message: Optional[str] = None,
                quantization: Optional[str] = None,
                **kwargs):
        """
        Inicializa o especialista Phi-2
        
        Args:
            expert_type: Tipo do especialista (ex: "sentiment", "technical", etc.)
            domain: Domínio principal (ex: "market", "risk", "quant")
            specialization: Especialização específica (ex: "sentiment_analysis", "volatility_modeling")
            config_path: Caminho para arquivo de configuração (opcional)
            gpu_id: ID da GPU para usar (None = seleção automática)
            gpu_ids: Lista de IDs de GPUs disponíveis
            model_path_override: Caminho alternativo para o modelo
            system_message: Mensagem de sistema para contextualizar o modelo
            quantization: Nível de quantização do modelo
        """
        # Chamar construtor da classe pai
        super().__init__(
            expert_type=expert_type,
            config_path=config_path,
            gpu_id=gpu_id,
            gpu_ids=gpu_ids,
            system_message=system_message,
            quantization=quantization or self.DEFAULT_QUANTIZATION,
            **kwargs
        )
        
        # Atributos específicos de Phi-2
        self.domain = domain
        self.specialization = specialization
        self.model_size_mb = 1100  # Tamanho aproximado do Phi-2 em MB
        self.expert_id = f"phi2_{domain}_{specialization}"
        
        # Inicializar gerenciador Multi-GPU (se disponível)
        try:
            self.gpu_manager = get_multi_gpu_manager()
            self.has_gpu_manager = True
        except Exception as e:
            logger.warning(f"Gerenciador Multi-GPU não disponível: {e}")
            self.gpu_manager = None
            self.has_gpu_manager = False
        
        # Inicializar estado
        self.model = None
        self.model_loaded = False
        self.last_used = 0
        self.usage_count = 0
        
        # Informações do modelo
        model_path = model_path_override or self.MODEL_NAME
        self.model_info = {
            "name": model_path,
            "size_mb": self.model_size_mb,
            "expert_type": expert_type,
            "domain": domain,
            "specialization": specialization,
            "quantization": self.quantization,
            "version": self.VERSION
        }
        
        # Carregar configurações
        self._load_config(config_path)
        
        # Registrar no gerenciador Multi-GPU
        if self.has_gpu_manager and hasattr(self, "model"):
            self.gpu_manager.register_model(
                self.expert_id, 
                self.model, 
                self.model_size_mb
            )
    
    def _load_config(self, config_path: Optional[Union[str, Path]] = None):
        """
        Carrega configurações do especialista
        
        Args:
            config_path: Caminho para arquivo de configuração
        """
        # Configuração padrão
        self.config = {
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "prompt_template": "[INST] {system_message}\n\n{prompt} [/INST]",
            "system_message": self.system_message or (
                f"Você é um especialista em {self.specialization} "
                f"no domínio de {self.domain} dentro do sistema EzioFilho_LLMGraph. "
                f"Forneça análises precisas e objetivas."
            )
        }
        
        # Sobrescrever com configurações do arquivo (se fornecido)
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        custom_config = json.load(f)
                        self.config.update(custom_config)
                    logger.info(f"Configurações carregadas de {config_path}")
                except Exception as e:
                    logger.error(f"Erro ao carregar configurações: {e}")
    
    def load_model(self):
        """
        Carrega o modelo Phi-2 e prepara para inferência
        """
        if self.model_loaded:
            logger.debug(f"Modelo {self.expert_id} já está carregado")
            return
        
        logger.info(f"Carregando modelo {self.MODEL_NAME} para especialista {self.expert_id}")
        
        try:
            # Alocar GPU via gerenciador Multi-GPU (se disponível)
            gpu_id = None
            if self.has_gpu_manager:
                gpu_id = self.gpu_manager.allocate_model_to_gpu(self.expert_id)
                
            if gpu_id is not None:
                logger.info(f"Modelo {self.expert_id} alocado para GPU {gpu_id}")
                self.gpu_id = gpu_id
                device = f"cuda:{gpu_id}"
            elif self.gpu_id is not None:
                logger.info(f"Usando GPU {self.gpu_id} para modelo {self.expert_id}")
                device = f"cuda:{self.gpu_id}"
            elif torch.cuda.is_available():
                # Usar primeira GPU disponível
                self.gpu_id = 0
                device = "cuda:0"
                logger.info(f"Usando GPU default (0) para modelo {self.expert_id}")
            else:
                logger.warning(f"Nenhuma GPU disponível para modelo {self.expert_id}, usando CPU")
                device = "cpu"
                self.gpu_id = None
            
            # Criar wrapper universal para o modelo
            self.model = UniversalModelWrapper(
                model_name_or_path=self.MODEL_NAME,
                device=device,
                model_type="transformers",  # Phi-2 é baseado em transformers
                quantization=self.quantization,
                max_length=self.config.get("max_length", 2048),
                load_in_8bit=self.quantization == "8bit",
                load_in_4bit=self.quantization == "4bit"
            )
            
            self.model_loaded = True
            self.last_used = time.time()
            logger.info(f"Modelo {self.expert_id} carregado com sucesso em {device}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {self.expert_id}: {e}")
            self.model = None
            self.model_loaded = False
            raise
    
    def unload_model(self):
        """
        Descarrega o modelo da GPU para liberar memória
        """
        if not self.model_loaded or self.model is None:
            return
        
        try:
            # Descarregar do gerenciador Multi-GPU
            if self.has_gpu_manager:
                self.gpu_manager.unload_model(self.expert_id)
            
            # Liberar recursos do modelo
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.model_loaded = False
            logger.info(f"Modelo {self.expert_id} descarregado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao descarregar modelo {self.expert_id}: {e}")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Gera resposta para um prompt usando o modelo Phi-2
        
        Args:
            prompt: Texto de entrada para o modelo
            **kwargs: Parâmetros adicionais para geração
            
        Returns:
            Resposta gerada pelo modelo
        """
        if not self.model_loaded:
            self.load_model()
            
        # Atualizar contagem de uso
        self.usage_count += 1
        self.last_used = time.time()
        
        # Atualizar no gerenciador Multi-GPU
        if self.has_gpu_manager:
            self.gpu_manager.mark_model_used(self.expert_id)
        
        # Preparar prompt com template
        system_message = kwargs.pop("system_message", self.config.get("system_message", ""))
        formatted_prompt = self.config["prompt_template"].format(
            system_message=system_message,
            prompt=prompt
        )
        
        # Parâmetros de geração
        gen_params = {
            "temperature": self.config.get("temperature", 0.7),
            "top_p": self.config.get("top_p", 0.9),
            "max_tokens": kwargs.pop("max_tokens", self.config.get("max_length", 2048)),
        }
        
        # Atualizar com parâmetros customizados
        gen_params.update(kwargs)
        
        # Gerar resposta
        try:
            response = self.model.generate(formatted_prompt, **gen_params)
            return response
        except Exception as e:
            logger.error(f"Erro na geração de resposta: {e}")
            return f"Erro: {str(e)}"
    
    def analyze(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Método principal para análise, deve ser implementado por subclasses
        
        Args:
            input_data: Dados de entrada (texto ou dicionário)
            
        Returns:
            Resultados da análise
        """
        raise NotImplementedError("Subclasses devem implementar o método analyze()")
        
    def get_status(self) -> Dict[str, Any]:
        """
        Retorna status atual do especialista
        
        Returns:
            Dicionário com status
        """
        return {
            "expert_id": self.expert_id,
            "expert_type": self.expert_type,
            "domain": self.domain,
            "specialization": self.specialization,
            "model_loaded": self.model_loaded,
            "gpu_id": self.gpu_id,
            "last_used": self.last_used,
            "usage_count": self.usage_count,
            "version": self.VERSION
        }
    
    def __del__(self):
        """Descarregar modelo ao destruir a instância"""
        self.unload_model()
