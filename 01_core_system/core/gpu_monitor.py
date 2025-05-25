#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU Monitor - Sistema de monitoramento de GPUs para EzioFilho_LLMGraph
----------------------------------------------------------------------
Este módulo fornece funcionalidades para:
- Monitorar uso de memória em múltiplas GPUs
- Coletar estatísticas de uso
- Implementar balanceamento de carga dinâmico entre GPUs
- Oferecer interface para monitoramento em tempo real

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
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("GPUMonitor")

class GPUMonitor:
    """
    Sistema de monitoramento de uso de GPUs.
    
    Esta classe fornece:
    - Monitoramento em tempo real do uso de memória de GPUs
    - Histórico de uso para análise de tendências
    - Métricas para balanceamento de carga
    - Detecção de sobrecarga e prevenção de OOM (Out of Memory)
    """
    
    def __init__(
        self, 
        gpu_ids: Optional[List[int]] = None,
        poll_interval: float = 1.0,
        history_size: int = 60,
        auto_start: bool = True
    ):
        """
        Inicializa o monitor de GPUs.
        
        Args:
            gpu_ids: Lista de IDs de GPUs para monitorar (None para todas)
            poll_interval: Intervalo de atualização em segundos
            history_size: Tamanho do histórico a manter
            auto_start: Se deve iniciar monitoramento automaticamente
        """
        self.poll_interval = poll_interval
        self.history_size = history_size
        self.gpu_ids = gpu_ids
        
        # Estado interno
        self.is_running = False
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.RLock()
        
        # Informações das GPUs
        self.gpu_info = {}
        self.history = {}
        
        # Tentar importar torch
        try:
            import torch
            self.torch = torch
            self._has_torch = True
            
            # Auto-detecção de GPUs se não especificadas
            if self.gpu_ids is None:
                self._detect_available_gpus()
                
            # Inicializa dados para cada GPU
            self._initialize_gpu_data()
            
        except ImportError:
            logger.warning("PyTorch não encontrado. Funcionalidade de GPU limitada.")
            self._has_torch = False
            self.torch = None
            
        # Se configurado para início automático
        if auto_start and self._has_torch:
            self.start()
    
    def _detect_available_gpus(self):
        """Detecta todas as GPUs disponíveis no sistema."""
        if not self._has_torch:
            self.gpu_ids = []
            return
            
        try:
            # Verifica se CUDA está disponível
            if not self.torch.cuda.is_available():
                logger.warning("CUDA não disponível")
                self.gpu_ids = []
                return
                
            # Obtém contagem de GPUs
            num_gpus = self.torch.cuda.device_count()
            if num_gpus == 0:
                logger.warning("Nenhuma GPU CUDA detectada")
                self.gpu_ids = []
                return
                
            # Lista todas as GPUs disponíveis
            self.gpu_ids = list(range(num_gpus))
            logger.info(f"Detectadas {num_gpus} GPUs: {self.gpu_ids}")
            
        except Exception as e:
            logger.error(f"Erro ao detectar GPUs: {e}")
            self.gpu_ids = []
    
    def _initialize_gpu_data(self):
        """Inicializa estruturas de dados para as GPUs."""
        if not self._has_torch or not self.gpu_ids:
            return
            
        with self._lock:
            self.gpu_info = {}
            self.history = {}
            
            for gpu_id in self.gpu_ids:
                try:
                    name = self.torch.cuda.get_device_name(gpu_id)
                    props = self.torch.cuda.get_device_properties(gpu_id)
                    
                    # Informações estáticas da GPU
                    self.gpu_info[gpu_id] = {
                        "id": gpu_id,
                        "name": name,
                        "total_memory": props.total_memory / (1024**2),  # MB
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multi_processor_count": props.multi_processor_count,
                        "supports_tensor_cores": bool(props.major >= 7),  # Tensor cores disponíveis a partir da arquitetura Volta (7.x)
                        "supports_fp16": bool(props.major >= 6),  # FP16 disponível a partir da arquitetura Pascal (6.x)
                    }
                    
                    # Inicializar histórico vazio
                    self.history[gpu_id] = []
                    
                    logger.info(f"GPU {gpu_id}: {name} ({self.gpu_info[gpu_id]['total_memory']:.2f} MB)")
                    
                except Exception as e:
                    logger.error(f"Erro ao inicializar dados para GPU {gpu_id}: {e}")
    
    def start(self):
        """Inicia o thread de monitoramento."""
        if self.is_running or not self._has_torch:
            return
            
        # Garante que temos GPUs para monitorar
        if not self.gpu_ids:
            logger.warning("Sem GPUs para monitorar")
            return
            
        # Reinicia o evento de parada
        self._stop_event.clear()
        
        # Inicia thread de monitoramento
        self._thread = threading.Thread(
            target=self._monitoring_loop,
            name="GPUMonitorThread",
            daemon=True
        )
        self._thread.start()
        self.is_running = True
        logger.info(f"Monitoramento de GPU iniciado para GPUs {self.gpu_ids}")
    
    def stop(self):
        """Para o thread de monitoramento."""
        if not self.is_running:
            return
            
        # Sinaliza parada
        self._stop_event.set()
        
        # Aguarda thread terminar
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            
        self.is_running = False
        logger.info("Monitoramento de GPU parado")
    
    def _monitoring_loop(self):
        """Loop principal de monitoramento."""
        while not self._stop_event.is_set():
            try:
                # Atualiza métricas de todas as GPUs
                for gpu_id in self.gpu_ids:
                    self._update_gpu_metrics(gpu_id)
                    
                # Aguarda próximo ciclo
                time.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Erro no ciclo de monitoramento: {e}")
                time.sleep(self.poll_interval)
    
    def _update_gpu_metrics(self, gpu_id: int):
        """
        Atualiza métricas para uma GPU específica.
        
        Args:
            gpu_id: ID da GPU a atualizar
        """
        if not self._has_torch:
            return
            
        try:
            # Estatísticas de memória
            mem_allocated = self.torch.cuda.memory_allocated(gpu_id) / (1024**2)  # MB
            mem_reserved = self.torch.cuda.memory_reserved(gpu_id) / (1024**2)    # MB
            total_memory = self.gpu_info[gpu_id]["total_memory"]
            mem_utilization = (mem_allocated / total_memory) * 100  # Percentual
            
            # Métricas deste ponto no tempo
            metrics = {
                "timestamp": time.time(),
                "mem_allocated_mb": mem_allocated,
                "mem_reserved_mb": mem_reserved,
                "mem_utilization": mem_utilization,
                "memory_free_mb": total_memory - mem_allocated,
            }
            
            # Adiciona ao histórico
            with self._lock:
                self.history[gpu_id].append(metrics)
                
                # Limita tamanho do histórico
                if len(self.history[gpu_id]) > self.history_size:
                    self.history[gpu_id].pop(0)
            
        except Exception as e:
            logger.error(f"Erro ao atualizar métricas para GPU {gpu_id}: {e}")
    
    def get_current_metrics(self) -> Dict[int, Dict[str, Any]]:
        """
        Obtém métricas atuais de todas as GPUs monitoradas.
        
        Returns:
            Dicionário com métricas atuais por GPU
        """
        result = {}
        
        with self._lock:
            for gpu_id in self.gpu_ids:
                # Informações básicas da GPU
                result[gpu_id] = {
                    "id": gpu_id,
                    "name": self.gpu_info[gpu_id]["name"]
                }
                
                # Adiciona métricas mais recentes se disponíveis
                if self.history[gpu_id]:
                    latest = self.history[gpu_id][-1]
                    result[gpu_id].update({
                        "mem_allocated_mb": latest["mem_allocated_mb"],
                        "mem_reserved_mb": latest["mem_reserved_mb"],
                        "mem_utilization": latest["mem_utilization"],
                        "memory_free_mb": latest["memory_free_mb"],
                        "total_memory_mb": self.gpu_info[gpu_id]["total_memory"]
                    })
                    
        return result
    
    def get_gpu_metrics_history(self, gpu_id: int) -> List[Dict[str, Any]]:
        """
        Obtém o histórico de métricas para uma GPU específica.
        
        Args:
            gpu_id: ID da GPU
            
        Returns:
            Lista com histórico de métricas
        """
        with self._lock:
            if gpu_id in self.history:
                return list(self.history[gpu_id])  # Retorna cópia para evitar race conditions
            return []
    
    def get_best_gpu_for_model(self, model_size_mb: float, prefer_stronger_gpu: bool = True) -> Optional[int]:
        """
        Determina a melhor GPU para carregar um modelo de um determinado tamanho.
        Prioriza GPUs com memória suficiente e, em seguida, opcionalmente, a GPU mais forte
        ou com menor utilização de memória.

        Args:
            model_size_mb: Tamanho estimado do modelo em MB.
            prefer_stronger_gpu: Se True, entre GPUs com memória suficiente, prefere a mais forte 
                                 (maior memória total como proxy).
                                 Se False, prefere a com menor utilização de memória atual.

        Returns:
            ID da GPU mais adequada, ou None se nenhuma tiver memória suficiente.
        """
        if not self._has_torch or not self.is_running:
            logger.warning("Monitor de GPU não está rodando ou PyTorch não disponível para get_best_gpu_for_model.")
            return None

        candidate_gpus = []
        # Obter métricas atuais uma vez para evitar múltiplas chamadas/locks
        current_metrics_all_gpus = self.get_current_metrics() 

        safety_margin_mb_default = 200 # Default safety margin if total_memory is not available for a GPU

        with self._lock: # Proteger acesso a self.gpu_info e self.gpu_ids
            if not self.gpu_ids:
                logger.warning("Nenhuma GPU ID configurada no monitor.")
                return None

            for gpu_id in self.gpu_ids:
                info = self.gpu_info.get(gpu_id)
                # Usar as métricas já obtidas fora do loop para esta GPU
                metrics = current_metrics_all_gpus.get(gpu_id)

                if not info or not metrics or "memory_free_mb" not in metrics or "total_memory_mb" not in metrics:
                    logger.debug(f"Dados de informação ou métricas insuficientes para GPU {gpu_id} em get_best_gpu_for_model.")
                    continue
                
                # Adicionar uma margem de segurança (ex: 5% da memória total da GPU ou um fixo mínimo)
                # para operações do sistema, fragmentação, etc.
                current_gpu_total_memory = info.get("total_memory_mb", info.get("total_memory")) # Handle both possible keys
                if current_gpu_total_memory is None:
                    logger.warning(f"Memória total não encontrada para GPU {gpu_id}, usando margem de segurança padrão.")
                    safety_margin_mb = safety_margin_mb_default
                else:
                    safety_margin_mb = max(safety_margin_mb_default, current_gpu_total_memory * 0.05)
                
                required_memory_with_margin = model_size_mb + safety_margin_mb

                if metrics["memory_free_mb"] >= required_memory_with_margin:
                    candidate_gpus.append({
                        "id": gpu_id,
                        "name": info.get("name", f"GPU {gpu_id}"), # Fallback para nome
                        "free_memory_mb": metrics["memory_free_mb"],
                        "total_memory_mb": current_gpu_total_memory, # Use the fetched total memory
                        "mem_utilization": metrics.get("mem_utilization", 100.0) 
                    })
                else:
                    logger.debug(
                        f"GPU {gpu_id} ({info.get('name', '')}) não tem memória suficiente. "
                        f"Necessário (com margem de {safety_margin_mb:.2f}MB): {required_memory_with_margin:.2f} MB, "
                        f"Disponível: {metrics['memory_free_mb']:.2f} MB"
                    )
        
        final_required_memory_note = model_size_mb + safety_margin_mb_default # For logging if no candidates
        if candidate_gpus: # If there are candidates, safety_margin_mb would be from the last checked GPU
            # This is a bit imprecise for logging if different GPUs had different safety margins,
            # but gives a general idea.
            # For a more precise log, we'd need to calculate the margin for the *best_gpu* if one is found.
            # Or simply state the range of margins if they vary.
            # For now, the debug log per GPU is more precise.
            pass


        if not candidate_gpus:
            logger.info(
                f"Nenhuma GPU com memória suficiente para modelo de {model_size_mb:.2f} MB "
                f"(necessidade aproximada com margem: {final_required_memory_note:.2f} MB, onde margem é ~max(200MB, 5% do total da GPU))."
            )
            return None

        # Ordenar candidatas:
        if prefer_stronger_gpu:
            # Prioridade 1: Maior memória total (proxy para "força")
            # Prioridade 2: Menor utilização de memória atual (para desempate)
            candidate_gpus.sort(key=lambda gpu: (-gpu["total_memory_mb"], gpu["mem_utilization"]))
        else:
            # Prioridade 1: Menor utilização de memória atual (para balanceamento)
            # Prioridade 2: Maior memória livre (para desempate)
            candidate_gpus.sort(key=lambda gpu: (gpu["mem_utilization"], -gpu["free_memory_mb"]))
        
        best_gpu = candidate_gpus[0]
        # Recalculate safety margin for the chosen best_gpu for accurate logging
        best_gpu_info = self.gpu_info.get(best_gpu['id'])
        best_gpu_total_mem = best_gpu_info.get("total_memory_mb", best_gpu_info.get("total_memory", 0))
        best_gpu_safety_margin = max(safety_margin_mb_default, best_gpu_total_mem * 0.05)

        logger.info(
            f"Melhor GPU selecionada para modelo de {model_size_mb:.2f}MB (necessário com margem ~{model_size_mb + best_gpu_safety_margin:.2f}MB): "
            f"GPU {best_gpu['id']} ({best_gpu['name']}) - Livre: {best_gpu['free_memory_mb']:.2f}MB, "
            f"Total: {best_gpu['total_memory_mb']:.2f}MB, Util: {best_gpu['mem_utilization']:.2f}%. "
            f"(Preferiu GPU mais forte: {prefer_stronger_gpu})"
        )
        return best_gpu["id"]

    def get_best_gpu_by_usage_pattern(self, model_name: str, model_size_mb: float, 
                                      last_used_timestamp: Optional[float] = None) -> Optional[int]:
        """
        Seleciona a melhor GPU baseada no padrão de uso histórico e tamanho do modelo.
        Esta função considera tanto memória disponível quanto frequência de uso do modelo.
        
        Args:
            model_name: Nome do modelo
            model_size_mb: Tamanho estimado do modelo em MB
            last_used_timestamp: Timestamp da última vez que este modelo foi usado
            
        Returns:
            ID da GPU mais adequada ou None se nenhuma GPU tiver memória suficiente
        """
        # Primeiro verifica se alguma GPU tem memória para o modelo
        candidate_gpu = self.get_best_gpu_for_model(model_size_mb, prefer_stronger_gpu=False)
        
        # Se não houver GPU com memória suficiente, retorna None (CPU)
        if candidate_gpu is None:
            logger.info(f"Sem GPU disponível para o modelo {model_name} ({model_size_mb:.2f}MB)")
            return None
        
        # Se o modelo não foi usado recentemente, apenas retorna a GPU com mais memória disponível
        if last_used_timestamp is None or time.time() - last_used_timestamp > 3600:  # Mais de 1 hora
            logger.info(f"Modelo {model_name} não usado recentemente. Usando GPU com mais memória disponível: {candidate_gpu}")
            return candidate_gpu
            
        # Se chegou aqui, implementa uma lógica mais sofisticada para modelos usados frequentemente
        # Essa lógica pode ser expandida com histórico real de uso dos modelos
        metrics = self.get_current_metrics()
        
        # Se houver mais de uma GPU, prefere manter modelos maiores na GPU mais poderosa
        if len(self.gpu_ids) > 1 and model_size_mb > 2000:  # Modelos > 2GB
            # Identificar GPU mais poderosa com base na memória total
            gpus_by_strength = sorted(
                [(gpu_id, metrics[gpu_id].get("total_memory_mb", 0)) for gpu_id in self.gpu_ids if gpu_id in metrics],
                key=lambda x: x[1],
                reverse=True  # Ordem decrescente de memória total
            )
            
            # Verificar se a GPU mais poderosa tem memória suficiente
            for gpu_id, total_mem in gpus_by_strength:
                free_mem = metrics[gpu_id].get("memory_free_mb", 0)
                if free_mem >= model_size_mb * 1.5:  # Fator de segurança
                    logger.info(f"Alocando modelo grande {model_name} para GPU mais potente {gpu_id}")
                    return gpu_id
        
        # Caso padrão: retorna a GPU com melhor equilíbrio de memória
        return candidate_gpu
        
    def unload_models_if_needed(self, gpu_id: int, required_memory_mb: float,
                                release_callback: Optional[callable] = None) -> bool:
        """
        Tenta liberar memória em uma GPU descarregando modelos não utilizados recentemente.
        
        Args:
            gpu_id: ID da GPU onde liberar memória
            required_memory_mb: Memória necessária em MB
            release_callback: Função de callback para liberar um modelo específico
            
        Returns:
            True se conseguiu liberar memória suficiente, False caso contrário
        """
        if not self._has_torch or not self.is_running:
            return False
            
        if gpu_id not in self.gpu_ids:
            logger.warning(f"GPU {gpu_id} não está sendo monitorada")
            return False
            
        # Verificar memória atual
        current_metrics = self.get_current_metrics().get(gpu_id, {})
        current_free = current_metrics.get("memory_free_mb", 0)
        
        # Se já tiver memória suficiente, retorna sucesso
        if current_free >= required_memory_mb:
            return True
            
        # Se não tiver callback para liberar modelos, não pode fazer nada
        if release_callback is None:
            logger.warning("Sem callback para liberar modelos, impossível liberar memória")
            return False
            
        # Tentar liberar memória através do callback
        success = release_callback(gpu_id, required_memory_mb)
        
        # Verificar nova memória disponível
        torch.cuda.empty_cache()  # Liberar caches do PyTorch
        new_metrics = self.get_current_metrics().get(gpu_id, {})
        new_free = new_metrics.get("memory_free_mb", 0)
        
        logger.info(f"Tentativa de liberar {required_memory_mb:.2f}MB na GPU {gpu_id}: " +
                   f"Antes: {current_free:.2f}MB, Após: {new_free:.2f}MB. Sucesso: {success}")
        
        return new_free >= required_memory_mb

    def is_gpu_overloaded(self, gpu_id: int, threshold_percent: float = 85.0) -> bool:
        """
        Verifica se uma GPU específica está sobrecarregada (uso de memória acima do threshold).

        Args:
            gpu_id: ID da GPU.
            threshold_percent: Percentual de uso de memória para considerar sobrecarga.

        Returns:
            True se sobrecarregada, False caso contrário.
        """
        if not self._has_torch or not self.is_running:
            logger.warning("Monitor de GPU não está rodando ou PyTorch indisponível para is_gpu_overloaded.")
            return False

        current_metrics = self.get_current_metrics().get(gpu_id, {})
        mem_utilization = current_metrics.get("mem_utilization", 0)

        is_overloaded = mem_utilization > threshold_percent

        if is_overloaded:
            logger.warning(f"GPU {gpu_id} está sobrecarregada (uso de memória: {mem_utilization:.1f}%)")
        else:
            logger.debug(f"GPU {gpu_id} está OK (uso de memória: {mem_utilization:.1f}%)")

        return is_overloaded

    def check_and_rebalance_models(self, registered_models: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Verifica o estado atual de uso da GPU e recomenda modelos para descarregar
        quando a pressão de memória é alta.
        
        Args:
            registered_models: Dicionário com modelos registrados e seus metadados
                               Formato esperado: {model_id: {
                                   "model": objeto_modelo,
                                   "last_used": timestamp,
                                   "size_mb": tamanho_modelo_mb,
                                   "usage_count": contagem_uso,
                                   "gpu_id": id_gpu_atual
                               }}
                               
        Returns:
            Lista de modelos recomendados para descarregamento/movimento, formato:
            [{"model_id": id, "action": "unload"|"move", "target_gpu": int|None}]
        """
        if not self._has_torch or not self.is_running:
            logger.warning("Monitor de GPU não está rodando ou PyTorch indisponível para rebalanceamento.")
            return []
            
        recommendations = []
        
        # Threshold para considerar GPUs sobrecarregadas
        memory_threshold = 85.0  # 85% de uso como threshold de sobrecarga
        
        # Identifica quais GPUs estão sobrecarregadas
        overloaded_gpus = []
        for gpu_id in self.gpu_ids:
            if self.is_gpu_overloaded(gpu_id, memory_threshold):
                overloaded_gpus.append(gpu_id)
                
        if not overloaded_gpus:
            logger.debug("Nenhuma GPU sobrecarregada, rebalanceamento não é necessário")
            return []
            
        logger.info(f"GPUs sobrecarregadas: {overloaded_gpus}, iniciando análise para rebalanceamento")
        
        # Para cada GPU sobrecarregada, identifica modelos que podem ser descarregados
        for gpu_id in overloaded_gpus:
            # Coleta modelos na GPU sobrecarregada
            gpu_models = []
            for model_id, model_info in registered_models.items():
                if model_info.get("gpu_id") == gpu_id:
                    model_info["model_id"] = model_id  # Adiciona ID do modelo para referência
                    gpu_models.append(model_info)
                    
            if not gpu_models:
                logger.warning(f"GPU {gpu_id} está sobrecarregada, mas nenhum modelo registrado foi encontrado")
                continue
                
            # Ordenar modelos por critérios de prioridade (menos usado e acessado há mais tempo primeiro)
            # Isso prioriza manter modelos frequentemente usados ou usados recentemente
            gpu_models.sort(key=lambda m: (m.get("usage_count", 0), -m.get("last_used", 0)))
            
            current_time = time.time()
            memory_to_free = 0
            
            # Obtem métricas de memória atual
            gpu_metrics = self.get_current_metrics().get(gpu_id, {})
            total_memory = gpu_metrics.get("total_memory_mb", 0)
            used_memory = gpu_metrics.get("mem_allocated_mb", 0)
            free_memory = gpu_metrics.get("memory_free_mb", 0)
            
            # Calcular memória alvo para liberar (para ficar abaixo do threshold)
            target_memory = (memory_threshold / 100.0) * total_memory
            if used_memory > target_memory:
                memory_to_free = used_memory - target_memory
                logger.info(f"GPU {gpu_id}: Necessário liberar {memory_to_free:.2f}MB para ficar abaixo do threshold")
            
            # Encontrar modelos alternativos para GPUs não sobrecarregadas
            alternative_gpus = [g for g in self.gpu_ids if g not in overloaded_gpus]
            
            # Se todas as GPUs estiverem sobrecarregadas, fazer backup para CPU
            if not alternative_gpus:
                logger.warning("Todas as GPUs estão sobrecarregadas, recomendando descarregamento para CPU")
            
            # Calcular quanto de memória precisamos liberar
            memory_freed = 0
            
            # Analisar modelos para descarregar ou mover
            for model in gpu_models:
                # Ignorar modelos recém-utilizados (últimos 5 minutos)
                if current_time - model.get("last_used", 0) < 300:  # 5 minutos
                    logger.debug(f"Modelo {model['model_id']} usado recentemente, mantendo na GPU {gpu_id}")
                    continue
                
                model_size = model.get("size_mb", 0)
                
                # Determinar ação recomendada para este modelo
                if alternative_gpus:
                    # Verificar se podemos mover para GPU alternativa
                    for alt_gpu in alternative_gpus:
                        alt_metrics = self.get_current_metrics().get(alt_gpu, {})
                        alt_free = alt_metrics.get("memory_free_mb", 0)
                        
                        if alt_free >= model_size * 1.2:  # 20% de margem de segurança
                            # Recomenda mover o modelo
                            recommendations.append({
                                "model_id": model["model_id"],
                                "action": "move",
                                "source_gpu": gpu_id,
                                "target_gpu": alt_gpu,
                                "size_mb": model_size,
                                "last_used": model.get("last_used", 0)
                            })
                            memory_freed += model_size
                            logger.info(f"Recomendação: Mover modelo {model['model_id']} da GPU {gpu_id} para GPU {alt_gpu}")
                            break
                    else:
                        # Se não encontrou GPU alternativa, recomenda descarregar
                        recommendations.append({
                            "model_id": model["model_id"],
                            "action": "unload",
                            "source_gpu": gpu_id,
                            "target_gpu": None,
                            "size_mb": model_size,
                            "last_used": model.get("last_used", 0)
                        })
                        memory_freed += model_size
                        logger.info(f"Recomendação: Descarregar modelo {model['model_id']} da GPU {gpu_id}")
                else:
                    # Sem GPUs alternativas, recomenda descarregar
                    recommendations.append({
                        "model_id": model["model_id"],
                        "action": "unload",
                        "source_gpu": gpu_id,
                        "target_gpu": None,
                        "size_mb": model_size,
                        "last_used": model.get("last_used", 0)
                    })
                    memory_freed += model_size
                    logger.info(f"Recomendação: Descarregar modelo {model['model_id']} da GPU {gpu_id}")
                
                # Verificar se já liberamos memória suficiente
                if memory_freed >= memory_to_free:
                    logger.info(f"Memória suficiente recomendada para liberação na GPU {gpu_id}: {memory_freed:.2f}MB")
                    break
        
        return recommendations
        if not self._has_torch or not self.is_running:
            logger.warning("Monitor de GPU não está rodando ou PyTorch indisponível para rebalanceamento.")
            return []
            
        recommendations = []
        
        # Threshold para considerar GPUs sobrecarregadas
        memory_threshold = 85.0  # 85% de uso como threshold de sobrecarga
        
        # Identifica quais GPUs estão sobrecarregadas
        overloaded_gpus = []
        for gpu_id in self.gpu_ids:
            if self.is_gpu_overloaded(gpu_id, memory_threshold):
                overloaded_gpus.append(gpu_id)
                
        if not overloaded_gpus:
            logger.debug("Nenhuma GPU sobrecarregada, rebalanceamento não é necessário")
            return []
            
        logger.info(f"GPUs sobrecarregadas: {overloaded_gpus}, iniciando análise para rebalanceamento")
        
        # Obtém métricas atuais uma vez para evitar chamadas repetidas
        current_metrics = self.get_current_metrics()
        
        # Para cada GPU sobrecarregada, identifica modelos que podem ser descarregados
        for gpu_id in overloaded_gpus:
            # Coleta modelos na GPU sobrecarregada
            gpu_models = []
            for model_id, model_info in registered_models.items():
                if model_info.get("gpu_id") == gpu_id:
                    # Copia o dicionário e adiciona o ID do modelo
                    model_data = model_info.copy()
                    model_data["model_id"] = model_id
                    gpu_models.append(model_data)
                    
            if not gpu_models:
                logger.warning(f"GPU {gpu_id} está sobrecarregada, mas nenhum modelo registrado foi encontrado")
                continue
                
            # Calcular pontuação para cada modelo baseado em:
            # 1. Tempo desde o último uso (mais antigo = maior pontuação)
            # 2. Contagem de uso (menos usado = maior pontuação)
            # 3. Tamanho (maior tamanho = maior pontuação para liberar mais memória)
            current_time = time.time()
            for model in gpu_models:
                time_since_use = current_time - model.get("last_used", 0) if model.get("last_used", 0) > 0 else float('inf')
                usage_count = model.get("usage_count", 0)
                size_mb = model.get("size_mb", 0)
                
                # Normalizar contagem de uso com logaritmo para reduzir o impacto de grandes diferenças
                usage_score = math.log(usage_count + 1) if usage_count > 0 else 0
                
                # Pontuação final (maior = mais candidato a ser descarregado)
                model["unload_score"] = (time_since_use / 3600) - usage_score + (size_mb / 1000)
            
            # Ordenar por pontuação (maior primeiro = melhor candidato para descarregar)
            gpu_models.sort(key=lambda m: -m["unload_score"])
            
            # Obtem métricas de memória atual para esta GPU
            gpu_metrics = current_metrics.get(gpu_id, {})
            total_memory = gpu_metrics.get("total_memory_mb", 0)
            used_memory = gpu_metrics.get("mem_allocated_mb", 0)
            
            # Calcular memória alvo para liberar (para ficar abaixo do threshold)
            target_memory = (memory_threshold / 100.0) * total_memory
            memory_to_free = 0
            if used_memory > target_memory:
                memory_to_free = used_memory - target_memory
                logger.info(f"GPU {gpu_id}: Necessário liberar {memory_to_free:.2f}MB para ficar abaixo do threshold")
            else:
                # Se não precisamos liberar memória específica, liberar pelo menos 20% do total
                memory_to_free = total_memory * 0.2
                logger.info(f"GPU {gpu_id}: Liberando {memory_to_free:.2f}MB preventivamente (20% da memória total)")
            
            # Encontrar modelos alternativos para GPUs não sobrecarregadas
            alternative_gpus = [g for g in self.gpu_ids if g not in overloaded_gpus]
            
            # Se todas as GPUs estiverem sobrecarregadas, fazer backup para CPU
            if not alternative_gpus:
                logger.warning("Todas as GPUs estão sobrecarregadas, recomendando descarregamento para CPU")
            
            # Calcular quanto de memória já liberamos
            memory_freed = 0
            
            # Analisar modelos para descarregar ou mover
            for model in gpu_models:
                # Ignorar modelos recém-utilizados (últimos 5 minutos)
                if current_time - model.get("last_used", 0) < 300:  # 5 minutos
                    logger.debug(f"Modelo {model['model_id']} usado recentemente, mantendo na GPU {gpu_id}")
                    continue
                
                model_size = model.get("size_mb", 0)
                
                # Determinar ação recomendada para este modelo
                if alternative_gpus:
                    # Verificar se podemos mover para GPU alternativa
                    for alt_gpu in alternative_gpus:
                        alt_metrics = current_metrics.get(alt_gpu, {})
                        alt_free = alt_metrics.get("memory_free_mb", 0)
                        
                        if alt_free >= model_size * 1.2:  # 20% de margem de segurança
                            # Recomenda mover o modelo
                            recommendations.append({
                                "model_id": model["model_id"],
                                "action": "move",
                                "source_gpu": gpu_id,
                                "target_gpu": alt_gpu,
                                "size_mb": model_size,
                                "last_used": model.get("last_used", 0),
                                "usage_count": model.get("usage_count", 0)
                            })
                            memory_freed += model_size
                            logger.info(
                                f"Recomendação: Mover modelo {model['model_id']} "
                                f"da GPU {gpu_id} para GPU {alt_gpu}. "
                                f"Tamanho: {model_size:.0f}MB, "
                                f"Último uso: {time.strftime('%H:%M:%S', time.localtime(model.get('last_used', 0)))}"
                            )
                            break
                    else:
                        # Se não encontrou GPU alternativa, recomenda descarregar
                        recommendations.append({
                            "model_id": model["model_id"],
                            "action": "unload",
                            "source_gpu": gpu_id,
                            "target_gpu": None,
                            "size_mb": model_size,
                            "last_used": model.get("last_used", 0),
                            "usage_count": model.get("usage_count", 0)
                        })
                        memory_freed += model_size
                        logger.info(
                            f"Recomendação: Descarregar modelo {model['model_id']} "
                            f"da GPU {gpu_id}. Tamanho: {model_size:.0f}MB, "
                            f"Último uso: {time.strftime('%H:%M:%S', time.localtime(model.get('last_used', 0)))}, "
                            f"Contagem uso: {model.get('usage_count', 0)}"
                        )
                else:
                    # Sem GPUs alternativas, recomenda descarregar
                    recommendations.append({
                        "model_id": model["model_id"],
                        "action": "unload",
                        "source_gpu": gpu_id,
                        "target_gpu": None,
                        "size_mb": model_size,
                        "last_used": model.get("last_used", 0),
                        "usage_count": model.get("usage_count", 0)
                    })
                    memory_freed += model_size
                    logger.info(
                        f"Recomendação: Descarregar modelo {model['model_id']} "
                        f"da GPU {gpu_id}. Tamanho: {model_size:.0f}MB, "
                        f"Último uso: {time.strftime('%H:%M:%S', time.localtime(model.get('last_used', 0)))}, "
                        f"Contagem uso: {model.get('usage_count', 0)}"
                    )
                
                # Verificar se já liberamos memória suficiente
                if memory_freed >= memory_to_free:
                    logger.info(f"Memória suficiente recomendada para liberação na GPU {gpu_id}: {memory_freed:.2f}MB")
                    break
        
        return recommendations

    def get_model_placement_stats(self) -> Dict[str, Any]:
        """
        Obtém estatísticas detalhadas sobre o uso atual das GPUs,
        incluindo recomendações para alocação de modelos.
        
        Returns:
            Dicionário com estatísticas detalhadas de uso das GPUs
        """
        result = {
            "timestamp": time.time(),
            "gpus": {},
            "recommendations": {
                "preferred_gpu_id": None,
                "balancing_needed": False,
                "memory_critical": False
            }
        }
        
        # Coletar métricas de todas as GPUs
        metrics = self.get_current_metrics()
        
        total_memory = 0
        total_used = 0
        max_free_gpu = None
        max_free_memory = 0
        
        for gpu_id, gpu_metrics in metrics.items():
            # Adicionar métricas à resposta
            result["gpus"][gpu_id] = {
                "id": gpu_id,
                "name": gpu_metrics.get("name", f"GPU {gpu_id}"),
                "total_memory_mb": gpu_metrics.get("total_memory_mb", 0),
                "used_memory_mb": gpu_metrics.get("mem_allocated_mb", 0),
                "free_memory_mb": gpu_metrics.get("memory_free_mb", 0),
                "utilization_percent": gpu_metrics.get("mem_utilization", 0),
                "overloaded": gpu_metrics.get("mem_utilization", 0) > 85.0
            }
            
            # Acumular totais
            gpu_total = gpu_metrics.get("total_memory_mb", 0)
            gpu_used = gpu_metrics.get("mem_allocated_mb", 0)
            
            total_memory += gpu_total
            total_used += gpu_used
            
            # Encontrar a GPU com mais memória livre
            free_memory = gpu_metrics.get("memory_free_mb", 0)
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                max_free_gpu = gpu_id
        
        # Verificar se há desequilíbrio significativo entre GPUs
        if len(result["gpus"]) > 1:
            min_util = min(gpu["utilization_percent"] for gpu in result["gpus"].values())
            max_util = max(gpu["utilization_percent"] for gpu in result["gpus"].values())
            
            # Se diferença > 30%, considerar desbalanceado
            if max_util - min_util > 30:
                result["recommendations"]["balancing_needed"] = True
        
        # Verificar estado crítico de memória
        if total_memory > 0:
            overall_util = (total_used / total_memory) * 100
            result["recommendations"]["memory_critical"] = overall_util > 90
        
        # Definir GPU preferencial para próxima alocação
        result["recommendations"]["preferred_gpu_id"] = max_free_gpu
        
        return result

    def detect_rtx2060_gtx1070(self) -> Dict[int, Dict[str, Any]]:
        """
        Detecta especificamente GPUs RTX 2060 e GTX 1070 e suas características.
        Retorna informações detalhadas sobre essas GPUs para otimização avançada.
        
        Returns:
            Dicionário com IDs de GPU como chaves e informações sobre a GPU como valor
        """
        if not self._has_torch or not self.is_running:
            logger.warning("Monitor de GPU não está rodando ou PyTorch não disponível para detecção de GPUs específicas.")
            return {}
            
        result = {}
        
        with self._lock:
            for gpu_id in self.gpu_ids:
                info = self.gpu_info.get(gpu_id, {})
                name = info.get("name", "").lower()
                
                # Verificar se é uma RTX 2060 ou GTX 1070
                if "rtx 2060" in name or "gtx 1070" in name:
                    gpu_type = "rtx_2060" if "rtx 2060" in name else "gtx_1070"
                    
                    # Extrair características específicas
                    has_tensor_cores = gpu_type == "rtx_2060"  # RTX 2060 tem Tensor Cores
                    cuda_capability = info.get("compute_capability", "0.0")
                    supports_fp16 = info.get("supports_fp16", False)
                    
                    # Definir características específicas por modelo
                    specific_info = {
                        "gpu_id": gpu_id,
                        "name": info.get("name", ""),
                        "gpu_type": gpu_type,
                        "has_tensor_cores": has_tensor_cores,
                        "cuda_capability": cuda_capability,
                        "memory_mb": info.get("total_memory", 0),
                        "supports_fp16": supports_fp16,
                        "recommended_models": []
                    }
                    
                    # Recomendações específicas por modelo
                    if gpu_type == "rtx_2060":
                        specific_info["recommended_models"] = [
                            "phi-3-medium-4k-instruct",  # Beneficia-se dos Tensor Cores
                            "phi-3-small-8k-instruct",   # Modelos menores para processamento eficiente
                            "phi-2"                      # Modelos menores para processamento eficiente
                        ]
                        specific_info["model_types"] = ["transformers", "optimized"]
                        specific_info["priority"] = 1  # Prioridade mais alta para modelos que precisam de Tensor Cores
                    else:  # GTX 1070
                        specific_info["recommended_models"] = [
                            "phi-2",                      # Modelos menores para processamento eficiente
                            "phi-1.5",                    # Modelos menores para processamento eficiente
                            "gpt2-medium"                 # Modelos menores para processamento eficiente
                        ]
                        specific_info["model_types"] = ["gguf", "onnx"]  # GGUF/ONNX para GPUs mais antigas
                        specific_info["priority"] = 2  # Prioridade secundária
                    
                    result[gpu_id] = specific_info
                    logger.info(f"GPU {gpu_id} detectada como {specific_info['name']} ({gpu_type})")
        
        if not result:
            logger.warning("Nenhuma GPU RTX 2060 ou GTX 1070 detectada.")
        
        return result

    def get_best_gpu_for_model_by_type(self, model_name: str, model_size_mb: float) -> Optional[int]:
        """
        Seleciona a melhor GPU baseada no tipo de modelo e suas características.
        Otimizado especificamente para as GPUs RTX 2060 e GTX 1070.
        
        Args:
            model_name: Nome do modelo (para identificar características como arquitetura)
            model_size_mb: Tamanho estimado do modelo em MB
            
        Returns:
            ID da GPU mais adequada ou None se nenhuma GPU tiver memória suficiente
        """
        # Primeiro verifica se há GPUs específicas (RTX 2060 / GTX 1070)
        specific_gpus = self.detect_rtx2060_gtx1070()
        
        if not specific_gpus:
            # Se não há GPUs específicas, usa o método padrão
            return self.get_best_gpu_for_model(model_size_mb)
            
        # Analisar nome do modelo para características
        model_name_lower = model_name.lower()
        
        # Verificar tipo/arquitetura do modelo
        is_phi3 = "phi-3" in model_name_lower or "phi3" in model_name_lower
        is_phi2 = "phi-2" in model_name_lower or "phi2" in model_name_lower 
        is_phi = "phi" in model_name_lower and not (is_phi3 or is_phi2)
        is_large = model_size_mb > 3000  # Modelos > 3GB são considerados grandes
        
        # Obter métricas atuais
        current_metrics = self.get_current_metrics()
        candidate_gpus = []
        
        for gpu_id, gpu_info in specific_gpus.items():
            metrics = current_metrics.get(gpu_id, {})
            free_memory = metrics.get("memory_free_mb", 0)
            safety_margin = max(200, gpu_info.get("memory_mb", 0) * 0.05)  # Margem de segurança
            
            # Verificar se há memória suficiente
            if free_memory >= model_size_mb + safety_margin:
                # Calcular pontuação de compatibilidade
                compatibility_score = 0
                
                if gpu_info["gpu_type"] == "rtx_2060":
                    # RTX 2060 é melhor para Phi-3 e modelos que precisam de Tensor Cores
                    if is_phi3:
                        compatibility_score += 100
                    elif is_phi2:
                        compatibility_score += 50
                    elif is_large:
                        compatibility_score += 30  # Geralmente melhor para modelos grandes
                else:  # GTX 1070
                    # GTX 1070 é melhor para modelos menores e Phi/Phi-2
                    if not is_large and (is_phi or is_phi2):
                        compatibility_score += 80
                    elif is_phi3 and not is_large:
                        compatibility_score += 40
                
                # Adicionar score de memória livre (normalizado)
                memory_score = (free_memory / (model_size_mb + safety_margin)) * 50
                
                candidate_gpus.append({
                    "id": gpu_id,
                    "name": gpu_info["name"],
                    "free_memory_mb": free_memory,
                    "compatibility_score": compatibility_score,
                    "memory_score": memory_score,
                    "total_score": compatibility_score + memory_score
                })
        
        if not candidate_gpus:
            logger.info(f"Nenhuma GPU compatível com memória suficiente para o modelo {model_name} ({model_size_mb:.2f}MB)")
            return None
            
        # Ordenar por pontuação total
        candidate_gpus.sort(key=lambda gpu: -gpu["total_score"])
        best_gpu = candidate_gpus[0]
        
        logger.info(
            f"Melhor GPU para {model_name} ({model_size_mb:.2f}MB): "
            f"GPU {best_gpu['id']} ({best_gpu['name']}) - "
            f"Pontuação: {best_gpu['total_score']:.1f} "
            f"(Compatibilidade: {best_gpu['compatibility_score']:.1f}, Memória: {best_gpu['memory_score']:.1f})"
        )
        
        return best_gpu["id"]

    def get_gpu_info(self):
        """
        Retorna informações sobre todas as GPUs monitoradas.
        
        Returns:
            Lista de dicionários com informações sobre cada GPU
        """
        if not self._has_torch:
            logger.warning("PyTorch não disponível para get_gpu_info.")
            return []
            
        result = []
        current_metrics = self.get_current_metrics()
        
        with self._lock:
            for gpu_id in self.gpu_ids:
                gpu_data = {
                    "id": gpu_id,
                    "name": self.gpu_info.get(gpu_id, {}).get("name", f"GPU {gpu_id}"),
                    "memory_total": self.gpu_info.get(gpu_id, {}).get("total_memory", 0),
                    "memory_used": 0,
                    "memory_free": 0,
                    "utilization": 0
                }
                
                # Adicionar métricas atuais se disponíveis
                if gpu_id in current_metrics:
                    metrics = current_metrics[gpu_id]
                    gpu_data.update({
                        "memory_used": metrics.get("memory_used_mb", 0),
                        "memory_free": metrics.get("memory_free_mb", 0),
                        "utilization": metrics.get("gpu_utilization", 0)
                    })
                
                result.append(gpu_data)
                
        return result
# Função para criar instância singleton
_gpu_monitor_instance = None

def get_gpu_monitor(
    gpu_ids: Optional[List[int]] = None,
    poll_interval: float = 1.0,
    history_size: int = 60
) -> GPUMonitor:
    """
    Obtém instância singleton do monitor de GPU.
    
    Args:
        gpu_ids: Lista de IDs de GPUs para monitorar
        poll_interval: Intervalo de atualização em segundos
        history_size: Tamanho do histórico a manter
        
    Returns:
        Instância do monitor de GPU
    """
    global _gpu_monitor_instance
    
    if _gpu_monitor_instance is None:
        _gpu_monitor_instance = GPUMonitor(
            gpu_ids=gpu_ids,
            poll_interval=poll_interval,
            history_size=history_size
        )
        
    return _gpu_monitor_instance


# Ponto de entrada para testes diretos
if __name__ == "__main__":
    import argparse
    
    # Parse argumentos de linha de comando
    parser = argparse.ArgumentParser(description="GPU Monitor")
    parser.add_argument("--gpu-ids", help="IDs das GPUs a monitorar (separados por vírgula)")
    parser.add_argument("--interval", type=float, default=1.0, help="Intervalo de atualização em segundos")
    parser.add_argument("--duration", type=int, default=60, help="Duração do teste em segundos")
    args = parser.parse_args()
    
    # Processa IDs de GPU
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    
    # Inicializa monitor
    monitor = GPUMonitor(
        gpu_ids=gpu_ids,
        poll_interval=args.interval,
        auto_start=True
    )
    
    # Exibe informações iniciais
    print(f"Monitorando GPUs: {monitor.gpu_ids}")
    
    try:
        # Loop de exibição
        end_time = time.time() + args.duration
        while time.time() < end_time:
            metrics = monitor.get_current_metrics()
            
            # Limpa tela
            print("\033c", end="")
            
            # Exibe cabeçalho
            print(f"======= GPU Monitor =======")
            print(f"Intervalo: {args.interval:.1f}s | Restante: {int(end_time - time.time())}s")
            print("===========================")
            
            # Exibe métricas para cada GPU
            for gpu_id, gpu_data in metrics.items():
                print(f"GPU {gpu_id}: {gpu_data['name']}")
                if "mem_utilization" in gpu_data:
                    print(f"  Memória: {gpu_data['mem_allocated_mb']:.1f}MB / {gpu_data['total_memory_mb']:.1f}MB ({gpu_data['mem_utilization']:.1f}%)")
                    print(f"  Livre: {gpu_data['memory_free_mb']:.1f}MB")
                print()
                
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\nMonitoramento interrompido pelo usuário")
    
    # Para o monitor
    monitor.stop()
