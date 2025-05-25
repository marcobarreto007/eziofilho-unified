#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EzioFilho_LLMGraph - Sistema de Gestão de GPUs Múltiplas
Otimizado para RTX 2060 e GTX 1070
--------------------------------------------------------
Este módulo fornece um gerenciador otimizado para distribuir modelos
entre GPUs RTX 2060 e GTX 1070, levando em consideração suas
características específicas.

Autor: EzioFilho LLMGraph Team
Data: Maio/2025
"""

import os
import sys
import time
import json
import logging
import threading
import math
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from pathlib import Path

# Adicionar diretório pai ao path para importações relativas
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Importar módulos do projeto
from core.gpu_monitor import get_gpu_monitor, GPUMonitor

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("MultiGPUManager")

class MultiGPUManager:
    """
    Gerenciador especializado para balanceamento de modelos entre múltiplas GPUs.
    
    Características:
    - Reconhecimento específico de RTX 2060 e GTX 1070
    - Priorização inteligente baseada nas capacidades de cada GPU
    - Cache adaptativo para modelos frequentemente usados
    - Separação dos modelos por tamanho, arquitetura e características
    """
    
    def __init__(self):
        # Inicializar o monitor de GPU
        self.gpu_monitor = get_gpu_monitor()
        
        # Estado interno
        self._registered_models = {}  # modelo_id -> metadados
        self._model_assignments = {}  # modelo_id -> gpu_id
        self._model_usage_stats = {}  # modelo_id -> estatísticas de uso
        self._model_sizes = {}        # modelo_id -> tamanho em MB
        self._gpu_assignments = {}    # gpu_id -> conjunto de modelos
        self._lock = threading.RLock()
        
        # Inicializar gpus disponíveis
        self._initialize()
    
    def _initialize(self):
        """Inicializa o gerenciador detectando GPUs disponíveis."""
        # Detectar GPUs específicas (RTX 2060, GTX 1070)
        self.specific_gpus = self.gpu_monitor.detect_rtx2060_gtx1070()
        
        # Inicializar estruturas para cada GPU
        with self._lock:
            self._gpu_assignments = {gpu_id: set() for gpu_id in self.specific_gpus}
            
        if self.specific_gpus:
            gpu_types = [info["gpu_type"] for info in self.specific_gpus.values()]
            logger.info(f"Gerenciador Multi-GPU inicializado com {len(self.specific_gpus)} GPUs: {gpu_types}")
        else:
            logger.warning("Nenhuma GPU RTX 2060 ou GTX 1070 detectada. Usando configuração genérica.")
    
    def register_model(self, model_id: str, model_object: Any, model_size_mb: float) -> None:
        """
        Registra um modelo no gerenciador.
        
        Args:
            model_id: Identificador único do modelo
            model_object: Objeto do modelo (para referência)
            model_size_mb: Tamanho estimado do modelo em MB
        """
        with self._lock:
            self._registered_models[model_id] = {
                "model": model_object,
                "last_used": time.time(),
                "size_mb": model_size_mb,
                "usage_count": 0,
                "gpu_id": None
            }
            self._model_sizes[model_id] = model_size_mb
            self._model_usage_stats[model_id] = {
                "first_used": time.time(),
                "last_used": time.time(),
                "usage_count": 0,
                "average_duration": 0.0,
                "total_duration": 0.0
            }
            
        logger.info(f"Modelo registrado: {model_id} ({model_size_mb:.2f} MB)")
    
    def allocate_model_to_gpu(self, model_id: str) -> Optional[int]:
        """
        Aloca um modelo registrado para a GPU mais adequada.
        
        Args:
            model_id: Identificador do modelo
            
        Returns:
            ID da GPU onde o modelo foi alocado, ou None se não foi possível alocar
        """
        if model_id not in self._registered_models:
            logger.warning(f"Tentativa de alocar modelo não registrado: {model_id}")
            return None
            
        model_info = self._registered_models[model_id]
        model_size = model_info["size_mb"]
        
        # Se o modelo já estiver em uma GPU, retorna essa GPU
        if model_info["gpu_id"] is not None:
            logger.debug(f"Modelo {model_id} já está na GPU {model_info['gpu_id']}")
            return model_info["gpu_id"]
        
        # Determinar melhor GPU para este modelo
        best_gpu = self.gpu_monitor.get_best_gpu_for_model_by_type(model_id, model_size)
        
        if best_gpu is None:
            logger.warning(f"Não foi possível encontrar GPU adequada para o modelo {model_id}")
            return None
            
        # Atualizar registros
        with self._lock:
            self._registered_models[model_id]["gpu_id"] = best_gpu
            self._gpu_assignments.setdefault(best_gpu, set()).add(model_id)
            
        logger.info(f"Modelo {model_id} alocado para GPU {best_gpu}")
        return best_gpu
    
    def mark_model_used(self, model_id: str) -> None:
        """
        Marca um modelo como usado recentemente, atualizando estatísticas.
        
        Args:
            model_id: Identificador do modelo
        """
        if model_id not in self._registered_models:
            logger.warning(f"Tentativa de marcar uso de modelo não registrado: {model_id}")
            return
            
        current_time = time.time()
        
        with self._lock:
            self._registered_models[model_id]["last_used"] = current_time
            self._registered_models[model_id]["usage_count"] += 1
            
            # Atualizar estatísticas de uso
            stats = self._model_usage_stats[model_id]
            stats["last_used"] = current_time
            stats["usage_count"] += 1
    
    def rebalance_models(self) -> List[Dict[str, Any]]:
        """
        Verifica a necessidade de rebalanceamento de modelos entre GPUs
        e retorna recomendações de ações a serem tomadas.
        
        Returns:
            Lista de recomendações para descarregar ou mover modelos
        """
        # Usar o monitor de GPU para verificar sobrecarga
        recommendations = self.gpu_monitor.check_and_rebalance_models(self._registered_models)
        
        if recommendations:
            logger.info(f"Rebalanceamento recomendado: {len(recommendations)} ações")
            
        return recommendations
    
    def unload_model(self, model_id: str) -> bool:
        """
        Marca um modelo como descarregado da GPU.
        
        Args:
            model_id: Identificador do modelo
            
        Returns:
            True se o modelo foi marcado como descarregado, False caso contrário
        """
        if model_id not in self._registered_models:
            logger.warning(f"Tentativa de descarregar modelo não registrado: {model_id}")
            return False
            
        with self._lock:
            model_info = self._registered_models[model_id]
            
            if model_info["gpu_id"] is not None:
                gpu_id = model_info["gpu_id"]
                # Remover modelo do conjunto de modelos da GPU
                if gpu_id in self._gpu_assignments:
                    self._gpu_assignments[gpu_id].discard(model_id)
                    
                # Atualizar registro do modelo
                model_info["gpu_id"] = None
                logger.info(f"Modelo {model_id} marcado como descarregado")
                return True
            else:
                logger.debug(f"Modelo {model_id} já está descarregado")
                return False
    
    def move_model(self, model_id: str, target_gpu_id: int) -> bool:
        """
        Move um modelo de uma GPU para outra.
        
        Args:
            model_id: Identificador do modelo
            target_gpu_id: ID da GPU de destino
            
        Returns:
            True se o modelo foi movido, False caso contrário
        """
        if model_id not in self._registered_models:
            logger.warning(f"Tentativa de mover modelo não registrado: {model_id}")
            return False
            
        if target_gpu_id not in self.specific_gpus:
            logger.warning(f"GPU de destino inválida: {target_gpu_id}")
            return False
            
        with self._lock:
            model_info = self._registered_models[model_id]
            source_gpu_id = model_info["gpu_id"]
            
            if source_gpu_id == target_gpu_id:
                logger.debug(f"Modelo {model_id} já está na GPU {target_gpu_id}")
                return True
                
            # Remover modelo da GPU de origem
            if source_gpu_id is not None and source_gpu_id in self._gpu_assignments:
                self._gpu_assignments[source_gpu_id].discard(model_id)
                
            # Adicionar modelo à GPU de destino
            self._gpu_assignments.setdefault(target_gpu_id, set()).add(model_id)
            
            # Atualizar registro do modelo
            model_info["gpu_id"] = target_gpu_id
            
            logger.info(f"Modelo {model_id} movido da GPU {source_gpu_id} para GPU {target_gpu_id}")
            return True
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """
        Obtém status detalhado das GPUs e modelos alocados.
        
        Returns:
            Dicionário com informações detalhadas sobre GPUs e modelos
        """
        result = {
            "timestamp": time.time(),
            "gpus": {},
            "models": {}
        }
        
        # Obter métricas atuais
        gpu_metrics = self.gpu_monitor.get_current_metrics()
        
        # Informações das GPUs
        for gpu_id, specific_info in self.specific_gpus.items():
            metrics = gpu_metrics.get(gpu_id, {})
            
            # Lista de modelos nesta GPU
            models_in_gpu = list(self._gpu_assignments.get(gpu_id, set()))
            
            # Memória total usada por modelos nesta GPU
            memory_used_by_models = sum(
                self._model_sizes.get(model_id, 0)
                for model_id in models_in_gpu
            )
            
            # Preparar informações da GPU
            result["gpus"][gpu_id] = {
                "id": gpu_id,
                "name": specific_info["name"],
                "gpu_type": specific_info["gpu_type"],
                "total_memory_mb": metrics.get("total_memory_mb", 0),
                "used_memory_mb": metrics.get("mem_allocated_mb", 0),
                "free_memory_mb": metrics.get("memory_free_mb", 0),
                "utilization_percent": metrics.get("mem_utilization", 0),
                "models_count": len(models_in_gpu),
                "models_memory_mb": memory_used_by_models,
                "models": models_in_gpu
            }
            
        # Informações dos modelos
        for model_id, model_info in self._registered_models.items():
            gpu_id = model_info["gpu_id"]
            gpu_name = self.specific_gpus.get(gpu_id, {}).get("name", "Nenhuma") if gpu_id is not None else "Nenhuma"
            
            result["models"][model_id] = {
                "id": model_id,
                "size_mb": model_info["size_mb"],
                "gpu_id": gpu_id,
                "gpu_name": gpu_name,
                "last_used": model_info["last_used"],
                "usage_count": model_info["usage_count"],
                "time_since_last_use": time.time() - model_info["last_used"]
            }
            
        return result
    
    def optimize_for_phi3(self) -> None:
        """
        Otimiza a alocação de recursos específica para modelos Phi-3.
        Prioriza RTX 2060 para os modelos Phi-3 devido aos Tensor Cores.
        """
        rtx_2060_id = None
        gtx_1070_id = None
        
        # Identificar as GPUs específicas
        for gpu_id, info in self.specific_gpus.items():
            if info["gpu_type"] == "rtx_2060":
                rtx_2060_id = gpu_id
            elif info["gpu_type"] == "gtx_1070":
                gtx_1070_id = gpu_id
                
        if rtx_2060_id is None:
            logger.warning("RTX 2060 não detectada, otimização para Phi-3 não aplicável")
            return
            
        # Identificar modelos Phi-3
        phi3_models = []
        other_models = []
        
        for model_id, info in self._registered_models.items():
            if "phi-3" in model_id.lower() or "phi3" in model_id.lower():
                phi3_models.append((model_id, info))
            else:
                other_models.append((model_id, info))
                
        # Prioridade de movimentação:
        # 1. Mover Phi-3 para RTX 2060 (se possível)
        # 2. Se necessário, mover outros modelos para GTX 1070
        
        for model_id, info in phi3_models:
            current_gpu = info["gpu_id"]
            
            # Se não estiver na RTX 2060, tentar mover
            if current_gpu != rtx_2060_id:
                # Verificar se há espaço na RTX 2060
                metrics = self.gpu_monitor.get_current_metrics().get(rtx_2060_id, {})
                free_memory = metrics.get("memory_free_mb", 0)
                
                if free_memory >= info["size_mb"] * 1.2:  # 20% de margem
                    self.move_model(model_id, rtx_2060_id)
                    logger.info(f"Otimização Phi-3: Modelo {model_id} movido para RTX 2060")
                else:
                    logger.info(f"Otimização Phi-3: Memória insuficiente na RTX 2060 para mover {model_id}")
        
        # Se tivermos uma GTX 1070, podemos mover não-Phi3 da RTX 2060 para lá
        if gtx_1070_id is not None:
            # Identificar modelos não-Phi3 na RTX 2060
            rtx_models = [m for m in other_models if m[1]["gpu_id"] == rtx_2060_id]
            
            for model_id, info in rtx_models:
                # Verificar se há espaço na GTX 1070
                metrics = self.gpu_monitor.get_current_metrics().get(gtx_1070_id, {})
                free_memory = metrics.get("memory_free_mb", 0)
                
                if free_memory >= info["size_mb"] * 1.2:  # 20% de margem
                    self.move_model(model_id, gtx_1070_id)
                    logger.info(f"Otimização Phi-3: Modelo não-Phi3 {model_id} movido da RTX 2060 para GTX 1070")
                    
                    # Verificar se agora podemos mover algum Phi-3 para a RTX 2060
                    for phi_id, phi_info in phi3_models:
                        if phi_info["gpu_id"] != rtx_2060_id:
                            # Verificar espaço novamente
                            new_metrics = self.gpu_monitor.get_current_metrics().get(rtx_2060_id, {})
                            new_free = new_metrics.get("memory_free_mb", 0)
                            
                            if new_free >= phi_info["size_mb"] * 1.2:
                                self.move_model(phi_id, rtx_2060_id)
                                logger.info(f"Otimização Phi-3: Modelo {phi_id} movido para RTX 2060 após liberação")
                                break
    
# Criar instância singleton
_multi_gpu_manager_instance = None

def get_multi_gpu_manager() -> MultiGPUManager:
    """
    Obtém instância singleton do gerenciador Multi-GPU.
    
    Returns:
        Instância do gerenciador Multi-GPU
    """
    global _multi_gpu_manager_instance
    
    if _multi_gpu_manager_instance is None:
        _multi_gpu_manager_instance = MultiGPUManager()
        
    return _multi_gpu_manager_instance
