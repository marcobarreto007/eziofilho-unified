#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sistema de Demonstração do EzioFilho_LLMGraph com Suporte Multi-GPU
-------------------------------------------------------------------
Script para demonstrar o funcionamento do sistema com GPUs RTX 2060 e GTX 1070,
carregando e gerenciando múltiplos modelos Phi-3, Phi-2 e outros.

Autor: EzioFilho LLMGraph Team
Data: Maio/2025
"""

import os
import sys
import time
import logging
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set

# Adicionar diretório pai ao path para importações relativas
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Importar módulos do projeto
from core.gpu_monitor import get_gpu_monitor
from core.multi_gpu_manager import get_multi_gpu_manager
from core.universal_model_wrapper import UniversalModelWrapper

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("EzioFilho_Demo")

# Modelos de demonstração com tamanhos aproximados
DEMO_MODELS = {
    "phi-3-medium-4k-instruct": 4200,   # 4.2GB
    "phi-3-small-8k-instruct": 1900,    # 1.9GB
    "phi-2": 1100,                      # 1.1GB 
    "phi-1.5": 850,                     # 850MB
    "gpt2-medium": 560,                 # 560MB
    "orca-mini-7b": 7800,               # 7.8GB - Grande demais para caber em uma GPU
    "mistral-7b-instruct-v0.2-q5_K_M": 5200,  # 5.2GB (Quantizado)
}

class DemoModel:
    """Classe de modelo simulado para demonstração."""
    
    def __init__(self, model_id: str, size_mb: float):
        self.model_id = model_id
        self.size_mb = size_mb
        self.is_loaded = False
        self.loaded_on_gpu = None
        self.parameters = {"size_mb": size_mb}
        
    def __repr__(self):
        status = f"Carregado na GPU {self.loaded_on_gpu}" if self.is_loaded else "Não carregado"
        return f"DemoModel({self.model_id}, {self.size_mb:.1f}MB, {status})"

def simulate_loading_model(model_id: str, gpu_id: Optional[int]) -> DemoModel:
    """
    Simula o carregamento de um modelo em uma GPU.
    
    Args:
        model_id: ID do modelo
        gpu_id: ID da GPU ou None para CPU
        
    Returns:
        Objeto de modelo simulado
    """
    if model_id not in DEMO_MODELS:
        raise ValueError(f"Modelo desconhecido: {model_id}")
        
    model_size = DEMO_MODELS[model_id]
    delay = model_size / 1000  # Tempo de carregamento proporcional ao tamanho
    
    device_name = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
    logger.info(f"Carregando modelo {model_id} ({model_size}MB) em {device_name}...")
    
    # Simular atraso de carregamento
    time.sleep(delay / 10)  # Dividido por 10 para não demorar muito em demos
    
    model = DemoModel(model_id, model_size)
    model.is_loaded = True
    model.loaded_on_gpu = gpu_id
    
    logger.info(f"Modelo {model_id} carregado com sucesso em {device_name}")
    return model

def run_demo():
    """Executa demonstração do sistema Multi-GPU."""
    logger.info("Iniciando demonstração do sistema EzioFilho_LLMGraph com Suporte Multi-GPU")
    
    # Inicializar monitor de GPU
    gpu_monitor = get_gpu_monitor()
    gpu_monitor.start()
    
    # Verificar GPUs disponíveis
    specific_gpus = gpu_monitor.detect_rtx2060_gtx1070()
    if not specific_gpus:
        logger.warning("RTX 2060 ou GTX 1070 não detectadas. A demo continuará com as GPUs disponíveis.")
    else:
        gpu_info = [f"{info['name']} (GPU {gpu_id})" for gpu_id, info in specific_gpus.items()]
        logger.info(f"GPUs detectadas: {gpu_info}")
    
    # Inicializar gerenciador Multi-GPU
    gpu_manager = get_multi_gpu_manager()
    
    # Carregar alguns modelos
    loaded_models = {}
    
    # Primeiro, tentar carregar os modelos menores
    for model_id in ["phi-2", "phi-1.5", "gpt2-medium"]:
        model_size = DEMO_MODELS[model_id]
        
        # Registrar o modelo no gerenciador
        model = simulate_loading_model(model_id, None)  # Inicialmente na CPU
        gpu_manager.register_model(model_id, model, model_size)
        loaded_models[model_id] = model
        
        # Alocar para GPU
        best_gpu = gpu_manager.allocate_model_to_gpu(model_id)
        if best_gpu is not None:
            model.loaded_on_gpu = best_gpu
            logger.info(f"Modelo {model_id} alocado para GPU {best_gpu}")
        else:
            logger.warning(f"Não foi possível alocar {model_id} para nenhuma GPU, permanecerá na CPU")
    
    # Mostrar status atual
    logger.info("\n--- Estado das GPUs após carregar modelos pequenos ---")
    status = gpu_manager.get_gpu_status()
    for gpu_id, gpu_info in status["gpus"].items():
        logger.info(
            f"GPU {gpu_id} ({gpu_info['name']}): "
            f"{gpu_info['models_count']} modelos, "
            f"{gpu_info['models_memory_mb']:.1f}MB / {gpu_info['total_memory_mb']:.1f}MB, "
            f"{gpu_info['utilization_percent']:.1f}% utilizada"
        )
    
    # Agora tentar carregar Phi-3
    logger.info("\n--- Tentando carregar modelos Phi-3 ---")
    for model_id in ["phi-3-small-8k-instruct", "phi-3-medium-4k-instruct"]:
        model_size = DEMO_MODELS[model_id]
        
        # Registrar o modelo no gerenciador
        model = simulate_loading_model(model_id, None)  # Inicialmente na CPU
        gpu_manager.register_model(model_id, model, model_size)
        loaded_models[model_id] = model
        
        # Alocar para GPU
        best_gpu = gpu_manager.allocate_model_to_gpu(model_id)
        if best_gpu is not None:
            model.loaded_on_gpu = best_gpu
            logger.info(f"Modelo {model_id} alocado para GPU {best_gpu}")
        else:
            logger.warning(f"Não foi possível alocar {model_id} para nenhuma GPU, permanecerá na CPU")
    
    # Mostrar status atual
    logger.info("\n--- Estado das GPUs após carregar todos os modelos ---")
    status = gpu_manager.get_gpu_status()
    for gpu_id, gpu_info in status["gpus"].items():
        logger.info(
            f"GPU {gpu_id} ({gpu_info['name']}): "
            f"{gpu_info['models_count']} modelos, "
            f"{gpu_info['models_memory_mb']:.1f}MB / {gpu_info['total_memory_mb']:.1f}MB, "
            f"{gpu_info['utilization_percent']:.1f}% utilizada"
        )
    
    # Simular uso de modelos
    logger.info("\n--- Simulando uso de modelos ---")
    for _ in range(3):
        # Usar Phi-2 várias vezes
        gpu_manager.mark_model_used("phi-2")
        time.sleep(0.5)
        
        # Usar Phi-3 Small
        gpu_manager.mark_model_used("phi-3-small-8k-instruct")
        time.sleep(1.0)
    
    # Usar Phi-3 Medium (simulando uso intenso)
    for _ in range(5):
        gpu_manager.mark_model_used("phi-3-medium-4k-instruct")
        time.sleep(0.3)
    
    # Mostrar status atual
    logger.info("\n--- Estatísticas de uso após simulação ---")
    status = gpu_manager.get_gpu_status()
    for model_id, model_info in status["models"].items():
        gpu_txt = f"GPU {model_info['gpu_id']} ({model_info['gpu_name']})" if model_info["gpu_id"] is not None else "CPU"
        logger.info(
            f"Modelo {model_id}: {model_info['usage_count']} usos, "
            f"último uso há {model_info['time_since_last_use']:.1f}s, "
            f"alocado em {gpu_txt}"
        )
    
    # Executar otimização específica para Phi-3
    logger.info("\n--- Otimizando ambiente para modelos Phi-3 ---")
    gpu_manager.optimize_for_phi3()
    
    # Mostrar status final
    logger.info("\n--- Estado final das GPUs após otimização ---")
    status = gpu_manager.get_gpu_status()
    for gpu_id, gpu_info in status["gpus"].items():
        logger.info(
            f"GPU {gpu_id} ({gpu_info['name']}): "
            f"Modelos: {gpu_info['models']}, "
            f"{gpu_info['models_memory_mb']:.1f}MB / {gpu_info['total_memory_mb']:.1f}MB, "
            f"{gpu_info['utilization_percent']:.1f}% utilizada"
        )
    
    # Rebalancear modelos se necessário
    recommendations = gpu_manager.rebalance_models()
    if recommendations:
        logger.info(f"\n--- Recomendações de rebalanceamento: {len(recommendations)} ações ---")
        for rec in recommendations:
            action = rec["action"]
            model_id = rec["model_id"]
            source_gpu = rec["source_gpu"]
            
            if action == "unload":
                logger.info(f"Recomendação: Descarregar {model_id} da GPU {source_gpu}")
                gpu_manager.unload_model(model_id)
                loaded_models[model_id].loaded_on_gpu = None
            elif action == "move":
                target_gpu = rec["target_gpu"]
                logger.info(f"Recomendação: Mover {model_id} da GPU {source_gpu} para GPU {target_gpu}")
                gpu_manager.move_model(model_id, target_gpu)
                loaded_models[model_id].loaded_on_gpu = target_gpu
    
    # Tentar carregar um modelo grande (deve falhar ou descarregar outros)
    logger.info("\n--- Tentando carregar um modelo grande (Orca Mini 7B) ---")
    model_id = "orca-mini-7b"
    model_size = DEMO_MODELS[model_id]
    
    model = simulate_loading_model(model_id, None)  # Inicialmente na CPU
    gpu_manager.register_model(model_id, model, model_size)
    loaded_models[model_id] = model
    
    # Alocar para GPU (deve falhar)
    best_gpu = gpu_manager.allocate_model_to_gpu(model_id)
    if best_gpu is not None:
        logger.info(f"Modelo grande {model_id} alocado para GPU {best_gpu} (inesperado)")
        model.loaded_on_gpu = best_gpu
    else:
        logger.warning(f"Como esperado, não foi possível alocar o modelo grande {model_id} para nenhuma GPU")
    
    # Encerrar demonstração
    logger.info("\n--- Demonstração concluída com sucesso ---")
    gpu_monitor.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstração do EzioFilho_LLMGraph com suporte Multi-GPU")
    args = parser.parse_args()
    
    run_demo()
