#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EzioFilho_LLMGraph - Integração Phi-2 e Phi-3
---------------------------------------------
Integra os especialistas Phi-2 com o sistema central Phi-3.

Autor: EzioFilho LLMGraph Team
Data: Maio/2025
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Phi2Phi3Integration")

# Importar componentes do sistema
from core.phi2_experts import get_phi2_expert, get_available_phi2_experts
from core.multi_gpu_manager import get_multi_gpu_manager
from core.gpu_monitor import get_gpu_monitor

class Phi2Phi3Integrator:
    """
    Sistema de integração entre especialistas Phi-2 e o cérebro central Phi-3.
    
    Responsabilidades:
    - Roteamento de consultas para especialistas apropriados
    - Agregação de respostas de múltiplos especialistas
    - Gerenciamento de batching e balanceamento de carga
    - Comunicação com o cérebro central Phi-3
    """
    
    def __init__(
        self, 
        phi3_endpoint: Optional[str] = None,
        phi3_model_id: str = "phi3-small-128k",
        experts_prefix: str = "phi2"
    ):
        """
        Inicializa o integrador.
        
        Args:
            phi3_endpoint: Endpoint da API para o modelo Phi-3 (opcional)
            phi3_model_id: ID do modelo Phi-3 a ser usado
            experts_prefix: Prefixo para identificar especialistas (padrão: "phi2")
        """
        self.phi3_endpoint = phi3_endpoint
        self.phi3_model_id = phi3_model_id
        self.experts_prefix = experts_prefix
        
        # Inicializar gerenciador Multi-GPU
        self.gpu_manager = get_multi_gpu_manager()
        
        # Carregar especialistas disponíveis
        self.available_experts = get_available_phi2_experts()
        self.active_experts = {}
        
        logger.info(f"Integrador Phi-2/Phi-3 inicializado com {len(self.available_experts)} especialistas disponíveis")
    
    def load_expert(self, expert_type: str) -> bool:
        """
        Carrega um especialista na memória.
        
        Args:
            expert_type: Tipo de especialista a ser carregado
            
        Returns:
            True se o especialista foi carregado com sucesso, False caso contrário
        """
        if expert_type not in self.available_experts:
            logger.warning(f"Tipo de especialista desconhecido: {expert_type}")
            return False
            
        if expert_type in self.active_experts:
            logger.info(f"Especialista {expert_type} já está carregado")
            return True
            
        try:
            # Inicializar o especialista
            expert = get_phi2_expert(expert_type)
            
            # Tentar alocar para GPU adequada
            expert_id = f"{self.experts_prefix}_{expert_type}"
            self.gpu_manager.register_model(expert_id, expert, 3000)  # Tamanho estimado em MB
            gpu_id = self.gpu_manager.allocate_model_to_gpu(expert_id)
            
            if gpu_id is not None:
                logger.info(f"Especialista {expert_type} alocado para GPU {gpu_id}")
            else:
                logger.warning(f"Especialista {expert_type} não pôde ser alocado para GPU")
            
            # Registrar como especialista ativo
            self.active_experts[expert_type] = {
                "expert": expert,
                "expert_id": expert_id,
                "gpu_id": gpu_id,
                "loaded_at": time.time()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar especialista {expert_type}: {e}")
            return False
    
    def unload_expert(self, expert_type: str) -> bool:
        """
        Descarrega um especialista da memória.
        
        Args:
            expert_type: Tipo de especialista a ser descarregado
            
        Returns:
            True se o especialista foi descarregado com sucesso, False caso contrário
        """
        if expert_type not in self.active_experts:
            logger.warning(f"Especialista {expert_type} não está carregado")
            return False
            
        try:
            expert_info = self.active_experts[expert_type]
            expert_id = expert_info["expert_id"]
            
            # Descarregar da GPU
            self.gpu_manager.unload_model(expert_id)
            
            # Remover dos especialistas ativos
            del self.active_experts[expert_type]
            
            logger.info(f"Especialista {expert_type} descarregado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao descarregar especialista {expert_type}: {e}")
            return False
    
    def query_expert(self, expert_type: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consulta um especialista específico.
        
        Args:
            expert_type: Tipo de especialista a consultar
            query_data: Dados da consulta
            
        Returns:
            Resultado da consulta
        """
        # Verificar se o especialista está carregado
        if expert_type not in self.active_experts:
            logger.info(f"Especialista {expert_type} não está carregado, carregando agora")
            if not self.load_expert(expert_type):
                return {"error": f"Falha ao carregar especialista {expert_type}"}
        
        try:
            # Obter o especialista
            expert_info = self.active_experts[expert_type]
            expert = expert_info["expert"]
            expert_id = expert_info["expert_id"]
            
            # Marcar como utilizado para fins de gerenciamento de cache
            self.gpu_manager.mark_model_used(expert_id)
            
            # Executar análise
            start_time = time.time()
            result = expert.analyze(query_data)
            elapsed_time = time.time() - start_time
            
            # Adicionar metadados
            result["processing_time"] = elapsed_time
            result["expert_type"] = expert_type
            
            logger.info(f"Consulta ao especialista {expert_type} concluída em {elapsed_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Erro ao consultar especialista {expert_type}: {e}")
            return {"error": str(e), "expert_type": expert_type}
    
    def query_multiple_experts(
        self, 
        expert_types: List[str], 
        query_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Consulta múltiplos especialistas em sequência.
        
        Args:
            expert_types: Lista de tipos de especialistas a consultar
            query_data: Dados da consulta
            
        Returns:
            Dicionário com resultados de cada especialista
        """
        results = {}
        
        for expert_type in expert_types:
            results[expert_type] = self.query_expert(expert_type, query_data)
        
        # Verificar necessidade de rebalanceamento
        self.gpu_manager.rebalance_models()
        
        return results
    
    def query_phi3(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Consulta o modelo central Phi-3.
        
        Args:
            prompt: Prompt a ser enviado ao modelo
            context: Contexto adicional (opcional)
            
        Returns:
            Resposta do modelo Phi-3
        """
        # Placeholder - implementar comunicação real com Phi-3
        logger.info(f"Consulta ao Phi-3 ({self.phi3_model_id})")
        
        # Simular resposta
        result = {
            "response": "Esta é uma resposta simulada do modelo Phi-3. Implementar comunicação real.",
            "model": self.phi3_model_id,
            "processing_time": 1.5,
            "timestamp": time.time()
        }
        
        return result
    
    def analyze_with_full_system(
        self, 
        query_data: Dict[str, Any], 
        expert_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Realiza análise completa usando especialistas Phi-2 e cérebro Phi-3.
        
        Args:
            query_data: Dados da consulta
            expert_types: Tipos de especialistas a utilizar (ou None para usar todos)
            
        Returns:
            Resultado agregado da análise
        """
        # Se não especificado, usar todos os especialistas disponíveis
        if expert_types is None:
            expert_types = self.available_experts
        
        # Consultar especialistas em sequência
        start_time = time.time()
        expert_results = self.query_multiple_experts(expert_types, query_data)
        
        # Preparar contexto para o Phi-3
        phi3_context = {
            "query": query_data,
            "expert_analyses": expert_results
        }
        
        # Montar prompt para o Phi-3
        prompt = self._build_integration_prompt(query_data, expert_results)
        
        # Consultar Phi-3 com os resultados dos especialistas
        phi3_result = self.query_phi3(prompt, phi3_context)
        
        # Criar resultado integrado
        total_time = time.time() - start_time
        result = {
            "query": query_data,
            "expert_results": expert_results,
            "phi3_result": phi3_result,
            "total_processing_time": total_time,
            "timestamp": time.time()
        }
        
        logger.info(f"Análise completa concluída em {total_time:.2f}s utilizando {len(expert_types)} especialistas")
        return result
    
    def _build_integration_prompt(
        self, 
        query_data: Dict[str, Any], 
        expert_results: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Constrói um prompt para o Phi-3 baseado nos resultados dos especialistas.
        
        Args:
            query_data: Dados da consulta original
            expert_results: Resultados dos especialistas
            
        Returns:
            Prompt formatado para o Phi-3
        """
        # Formatar os dados de consulta
        query_str = json.dumps(query_data, indent=2, ensure_ascii=False)
        
        # Formatar resultados dos especialistas
        experts_str = ""
        for expert_type, result in expert_results.items():
            if "error" in result:
                experts_str += f"\n### Especialista: {expert_type}\nERRO: {result['error']}\n"
            else:
                result_str = json.dumps(result, indent=2, ensure_ascii=False)
                experts_str += f"\n### Especialista: {expert_type}\n{result_str}\n"
        
        # Montar prompt para o Phi-3
        prompt = f"""
        # Análise Integrada EzioFilho_LLMGraph
        
        ## Dados da Consulta:
        ```json
        {query_str}
        ```
        
        ## Análises dos Especialistas:
        {experts_str}
        
        ## Instruções:
        Com base nas análises dos especialistas Phi-2 acima, crie uma análise integrada e abrangente.
        Resolva quaisquer contradições entre os especialistas.
        Destaque as principais conclusões e recomendações.
        Organize sua resposta em seções claras com introdução, análise e conclusão.
        """
        
        return prompt

# Função auxiliar para obter instância do integrador
_INTEGRATOR_INSTANCE = None

def get_phi2_phi3_integrator(**kwargs):
    """
    Retorna a instância global do integrador Phi-2/Phi-3.
    
    Args:
        **kwargs: Parâmetros para inicialização (ignorados se já existir instância)
        
    Returns:
        Instância do integrador
    """
    global _INTEGRATOR_INSTANCE
    
    if _INTEGRATOR_INSTANCE is None:
        _INTEGRATOR_INSTANCE = Phi2Phi3Integrator(**kwargs)
        
    return _INTEGRATOR_INSTANCE
