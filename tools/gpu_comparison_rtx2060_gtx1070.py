#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste de Desempenho RTX 2060 vs GTX 1070 para EzioFilho_LLMGraph
----------------------------------------------------------------
Este script testa especificamente o desempenho das GPUs RTX 2060 e GTX 1070
para diferentes tipos de modelos e tarefas, fornecendo uma comparação direta
e recomendações para otimização.

Autor: EzioFilho LLMGraph Team
Data: Maio/2025
"""

import os
import sys
import time
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Adicionar diretório pai ao path para importar módulos do projeto
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Importar módulos do projeto
from core.gpu_monitor import GPUMonitor, get_gpu_monitor
from core.universal_model_wrapper import UniversalModelWrapper

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("GPU_RTX2060_GTX1070_Test")

class GPUComparisonTest:
    """
    Testes especializados comparando RTX 2060 e GTX 1070.
    """
    
    def __init__(self):
        """Inicializa o teste de comparação de GPUs."""
        # Verificar CUDA
        if not torch.cuda.is_available():
            logger.error("CUDA não disponível. Não é possível realizar testes de GPU.")
            sys.exit(1)
            
        # Inicializar monitor de GPU
        self.gpu_monitor = get_gpu_monitor()
        self.gpu_ids = self.gpu_monitor.gpu_ids if self.gpu_monitor else []
        
        # Identificar IDs específicos para RTX 2060 e GTX 1070
        self.rtx_2060_id = None
        self.gtx_1070_id = None
        
        # Verificar GPUs disponíveis
        current_metrics = self.gpu_monitor.get_current_metrics()
        for gpu_id, metrics in current_metrics.items():
            name = metrics.get("name", "").lower()
            if "rtx 2060" in name:
                self.rtx_2060_id = gpu_id
                logger.info(f"RTX 2060 detectada como GPU {gpu_id}")
            elif "gtx 1070" in name:
                self.gtx_1070_id = gpu_id
                logger.info(f"GTX 1070 detectada como GPU {gpu_id}")
        
        # Verificar se ambas as GPUs foram encontradas
        if self.rtx_2060_id is None or self.gtx_1070_id is None:
            logger.warning("Não foi possível detectar automaticamente as GPUs RTX 2060 e GTX 1070")
            logger.warning("GPUs detectadas:")
            for gpu_id, metrics in current_metrics.items():
                logger.warning(f"  GPU {gpu_id}: {metrics.get('name', '')}")
            
            # Tentar inferir com base na memória
            # RTX 2060 tem 6GB, GTX 1070 tem 8GB
            self._infer_gpu_ids_by_memory()
        
        # Verificar novamente
        if self.rtx_2060_id is None or self.gtx_1070_id is None:
            logger.error("Não foi possível identificar as GPUs RTX 2060 e GTX 1070 necessárias para o teste")
            sys.exit(1)
        
        # Resultados dos testes
        self.results = {}
        
    def _infer_gpu_ids_by_memory(self):
        """Tenta inferir as GPUs por suas características de memória."""
        current_metrics = self.gpu_monitor.get_current_metrics()
        
        rtx_2060_memory = 6144  # RTX 2060 tem 6GB de memória
        gtx_1070_memory = 8192  # GTX 1070 tem 8GB de memória
        
        # Tolerância de 10% para variações na detecção de memória
        rtx_2060_min = rtx_2060_memory * 0.9
        rtx_2060_max = rtx_2060_memory * 1.1
        gtx_1070_min = gtx_1070_memory * 0.9
        gtx_1070_max = gtx_1070_memory * 1.1
        
        for gpu_id, metrics in current_metrics.items():
            total_memory_mb = metrics.get("total_memory_mb", 0)
            
            # Verificar se é RTX 2060 (6GB)
            if rtx_2060_min <= total_memory_mb <= rtx_2060_max and self.rtx_2060_id is None:
                self.rtx_2060_id = gpu_id
                logger.info(f"RTX 2060 provavelmente detectada como GPU {gpu_id} (memória: {total_memory_mb:.0f}MB)")
            
            # Verificar se é GTX 1070 (8GB)
            elif gtx_1070_min <= total_memory_mb <= gtx_1070_max and self.gtx_1070_id is None:
                self.gtx_1070_id = gpu_id
                logger.info(f"GTX 1070 provavelmente detectada como GPU {gpu_id} (memória: {total_memory_mb:.0f}MB)")
    
    def discover_test_models(self) -> List[Dict[str, Any]]:
        """
        Descobre modelos para testes.
        
        Returns:
            Lista de informações de modelos para teste
        """
        logger.info("Procurando modelos para testes...")
        
        model_dirs = [
            Path(PROJECT_ROOT) / "models",  # Diretório padrão de modelos
            Path.home() / ".cache" / "models",  # Cache padrão de modelos
        ]
        
        # Lista para armazenar modelos encontrados
        models = []
        
        # Lista de tamanhos alvo para testar diferentes categorias
        # Formato: [(nome_categoria, min_mb, max_mb)]
        size_categories = [
            ("small", 500, 1500),
            ("medium", 1500, 4000),
            ("large", 4000, 8000),
            ("xlarge", 8000, 15000)
        ]
        
        # Encontrar pelo menos um modelo para cada categoria
        for category, min_mb, max_mb in size_categories:
            category_models = []
            
            # Procurar em cada diretório
            for model_dir in model_dirs:
                if not model_dir.exists():
                    continue
                
                # Procurar arquivos com extensões comuns de modelos
                for ext in [".gguf", ".bin", ".safetensors"]:
                    for model_file in model_dir.glob(f"**/*{ext}"):
                        try:
                            size_mb = os.path.getsize(model_file) / (1024 * 1024)
                            
                            # Verificar se está na categoria de tamanho
                            if min_mb <= size_mb <= max_mb:
                                model_name = model_file.name
                                
                                category_models.append({
                                    "path": str(model_file),
                                    "name": model_name,
                                    "size_mb": size_mb,
                                    "category": category
                                })
                                
                                logger.info(f"Encontrado modelo {category}: {model_name} ({size_mb:.1f} MB)")
                        except Exception as e:
                            logger.warning(f"Erro ao verificar arquivo {model_file}: {e}")
            
            # Selecionar o primeiro modelo para esta categoria
            if category_models:
                # Ordenar por tamanho (crescente)
                category_models.sort(key=lambda x: x["size_mb"])
                
                # Adicionar o primeiro modelo desta categoria aos modelos de teste
                models.append(category_models[0])
            else:
                logger.warning(f"Nenhum modelo encontrado na categoria {category} ({min_mb}-{max_mb} MB)")
        
        if not models:
            logger.error("Nenhum modelo encontrado para testes. Por favor, forneça modelos manualmente.")
            sys.exit(1)
        
        return models
    
    def run_tensor_cores_test(self):
        """
        Testa o desempenho de Tensor Cores na RTX 2060.
        
        Os Tensor Cores são unidades específicas nas GPUs RTX que aceleram
        operações de matriz para deep learning. A GTX 1070 não possui estas unidades.
        """
        logger.info("Executando teste de Tensor Cores (RTX 2060)...")
        
        # Verificar se CUDA está disponível
        if not torch.cuda.is_available():
            logger.error("CUDA não disponível. Não é possível realizar teste de Tensor Cores.")
            return
        
        # Tensor Cores são usados automaticamente para operações de precisão mista (FP16)
        
        # Criar tensores para teste
        batch_size = 32
        seq_len = 512
        hidden_size = 1024
        
        # Executar teste na RTX 2060
        torch.cuda.set_device(self.rtx_2060_id)
        torch.cuda.empty_cache()
        
        # Criar tensores
        input_rtx = torch.randn(batch_size, seq_len, hidden_size, device=f"cuda:{self.rtx_2060_id}")
        weight_rtx = torch.randn(hidden_size, hidden_size, device=f"cuda:{self.rtx_2060_id}")
        
        # Converter para FP16 para acionar Tensor Cores
        input_rtx_fp16 = input_rtx.half()
        weight_rtx_fp16 = weight_rtx.half()
        
        # Aquecer
        for _ in range(5):
            _ = torch.matmul(input_rtx_fp16, weight_rtx_fp16)
        
        # Medir com FP16 (usando Tensor Cores na RTX 2060)
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            _ = torch.matmul(input_rtx_fp16, weight_rtx_fp16)
        torch.cuda.synchronize()
        fp16_time_rtx = time.time() - start_time
        
        # Medir com FP32 (sem Tensor Cores)
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            _ = torch.matmul(input_rtx, weight_rtx)
        torch.cuda.synchronize()
        fp32_time_rtx = time.time() - start_time
        
        # Executar teste na GTX 1070
        torch.cuda.set_device(self.gtx_1070_id)
        torch.cuda.empty_cache()
        
        # Criar tensores
        input_gtx = torch.randn(batch_size, seq_len, hidden_size, device=f"cuda:{self.gtx_1070_id}")
        weight_gtx = torch.randn(hidden_size, hidden_size, device=f"cuda:{self.gtx_1070_id}")
        
        # Converter para FP16
        input_gtx_fp16 = input_gtx.half()
        weight_gtx_fp16 = weight_gtx.half()
        
        # Aquecer
        for _ in range(5):
            _ = torch.matmul(input_gtx_fp16, weight_gtx_fp16)
        
        # Medir com FP16 (sem Tensor Cores na GTX 1070)
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            _ = torch.matmul(input_gtx_fp16, weight_gtx_fp16)
        torch.cuda.synchronize()
        fp16_time_gtx = time.time() - start_time
        
        # Medir com FP32
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            _ = torch.matmul(input_gtx, weight_gtx)
        torch.cuda.synchronize()
        fp32_time_gtx = time.time() - start_time
        
        # Calcular aceleração
        fp16_speedup_rtx = fp32_time_rtx / fp16_time_rtx
        fp16_speedup_gtx = fp32_time_gtx / fp16_time_gtx
        
        # Comparar RTX vs GTX em FP16
        rtx_vs_gtx_fp16 = fp16_time_gtx / fp16_time_rtx
        
        logger.info(f"RTX 2060 - FP16 vs FP32: {fp16_speedup_rtx:.2f}x mais rápido")
        logger.info(f"GTX 1070 - FP16 vs FP32: {fp16_speedup_gtx:.2f}x mais rápido")
        logger.info(f"RTX 2060 vs GTX 1070 (FP16): {rtx_vs_gtx_fp16:.2f}x mais rápido")
        
        # Armazenar resultados
        self.results["tensor_cores_test"] = {
            "rtx_2060": {
                "fp16_time": fp16_time_rtx,
                "fp32_time": fp32_time_rtx,
                "speedup": fp16_speedup_rtx
            },
            "gtx_1070": {
                "fp16_time": fp16_time_gtx,
                "fp32_time": fp32_time_gtx,
                "speedup": fp16_speedup_gtx
            },
            "rtx_vs_gtx_fp16_speedup": rtx_vs_gtx_fp16
        }
    
    def run_inference_test(self, models: List[Dict[str, Any]]):
        """
        Testa inferência de modelos em ambas as GPUs.
        
        Args:
            models: Lista de modelos para teste
        """
        logger.info("Executando teste de inferência de modelos...")
        
        # Verificar CUDA
        if not torch.cuda.is_available():
            logger.error("CUDA não disponível. Não é possível realizar testes de inferência.")
            return
        
        # Preparar resultados
        self.results["model_inference"] = {}
        
        # Prompt de teste
        prompts = [
            "Explique o conceito de inteligência artificial em termos simples.",
            "Quais são as principais diferenças entre machine learning e deep learning?",
            "O que são redes neurais e como elas funcionam?"
        ]
        
        # Executar teste para cada modelo
        for model_info in models:
            model_name = model_info["name"]
            model_path = model_info["path"]
            model_size = model_info["size_mb"]
            model_category = model_info["category"]
            
            logger.info(f"Testando modelo: {model_name} ({model_category}, {model_size:.1f} MB)")
            
            # Inicializar dicionário de resultados para este modelo
            self.results["model_inference"][model_name] = {
                "rtx_2060": {},
                "gtx_1070": {},
                "size_mb": model_size,
                "category": model_category
            }
            
            # Testar em RTX 2060
            try:
                # Limpar cache CUDA
                torch.cuda.empty_cache()
                
                # Carregar modelo
                start_time = time.time()
                model_rtx = UniversalModelWrapper(
                    model_path=model_path,
                    gpu_id=self.rtx_2060_id,
                    gpu_monitor=self.gpu_monitor,
                    lazy_load=False
                )
                load_time_rtx = time.time() - start_time
                
                # Executar inferência
                inference_times_rtx = []
                tokens_per_second_rtx = []
                
                for prompt in prompts:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    output = model_rtx.generate(
                        prompt=prompt,
                        max_tokens=100,
                        temperature=0.7
                    )
                    
                    torch.cuda.synchronize()
                    inference_time = time.time() - start_time
                    
                    # Medir tokens por segundo
                    output_tokens = len(output.split())
                    tokens_per_s = output_tokens / inference_time
                    
                    inference_times_rtx.append(inference_time)
                    tokens_per_second_rtx.append(tokens_per_s)
                
                # Calcular médias
                avg_inference_time_rtx = np.mean(inference_times_rtx)
                avg_tokens_per_s_rtx = np.mean(tokens_per_second_rtx)
                
                # Descarregar modelo
                del model_rtx
                torch.cuda.empty_cache()
                
                # Armazenar resultados
                self.results["model_inference"][model_name]["rtx_2060"] = {
                    "load_time": load_time_rtx,
                    "inference_time": avg_inference_time_rtx,
                    "tokens_per_second": avg_tokens_per_s_rtx
                }
                
                logger.info(f"RTX 2060: Carregamento: {load_time_rtx:.2f}s, Inferência: {avg_inference_time_rtx:.2f}s, Throughput: {avg_tokens_per_s_rtx:.2f} tokens/s")
                
            except Exception as e:
                logger.error(f"Erro ao testar modelo {model_name} na RTX 2060: {e}")
                self.results["model_inference"][model_name]["rtx_2060"] = {
                    "error": str(e)
                }
            
            # Testar em GTX 1070
            try:
                # Limpar cache CUDA
                torch.cuda.empty_cache()
                
                # Carregar modelo
                start_time = time.time()
                model_gtx = UniversalModelWrapper(
                    model_path=model_path,
                    gpu_id=self.gtx_1070_id,
                    gpu_monitor=self.gpu_monitor,
                    lazy_load=False
                )
                load_time_gtx = time.time() - start_time
                
                # Executar inferência
                inference_times_gtx = []
                tokens_per_second_gtx = []
                
                for prompt in prompts:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    output = model_gtx.generate(
                        prompt=prompt,
                        max_tokens=100,
                        temperature=0.7
                    )
                    
                    torch.cuda.synchronize()
                    inference_time = time.time() - start_time
                    
                    # Medir tokens por segundo
                    output_tokens = len(output.split())
                    tokens_per_s = output_tokens / inference_time
                    
                    inference_times_gtx.append(inference_time)
                    tokens_per_second_gtx.append(tokens_per_s)
                
                # Calcular médias
                avg_inference_time_gtx = np.mean(inference_times_gtx)
                avg_tokens_per_s_gtx = np.mean(tokens_per_second_gtx)
                
                # Descarregar modelo
                del model_gtx
                torch.cuda.empty_cache()
                
                # Armazenar resultados
                self.results["model_inference"][model_name]["gtx_1070"] = {
                    "load_time": load_time_gtx,
                    "inference_time": avg_inference_time_gtx,
                    "tokens_per_second": avg_tokens_per_s_gtx
                }
                
                logger.info(f"GTX 1070: Carregamento: {load_time_gtx:.2f}s, Inferência: {avg_inference_time_gtx:.2f}s, Throughput: {avg_tokens_per_s_gtx:.2f} tokens/s")
                
                # Calcular diferenças
                if "load_time" in self.results["model_inference"][model_name]["rtx_2060"]:
                    load_speedup = load_time_gtx / self.results["model_inference"][model_name]["rtx_2060"]["load_time"]
                    inference_speedup = avg_inference_time_gtx / self.results["model_inference"][model_name]["rtx_2060"]["inference_time"]
                    throughput_ratio = self.results["model_inference"][model_name]["rtx_2060"]["tokens_per_second"] / avg_tokens_per_s_gtx
                    
                    self.results["model_inference"][model_name]["comparison"] = {
                        "load_speedup": load_speedup,
                        "inference_speedup": inference_speedup,
                        "throughput_ratio": throughput_ratio
                    }
                    
                    logger.info(f"Comparação (RTX 2060 vs GTX 1070): Carregamento: {load_speedup:.2f}x, Inferência: {inference_speedup:.2f}x, Throughput: {throughput_ratio:.2f}x")
                
            except Exception as e:
                logger.error(f"Erro ao testar modelo {model_name} na GTX 1070: {e}")
                self.results["model_inference"][model_name]["gtx_1070"] = {
                    "error": str(e)
                }
    
    def run_parallel_test(self, model_info: Dict[str, Any]):
        """
        Testa a execução paralela de modelos em ambas as GPUs.
        
        Args:
            model_info: Informações do modelo para teste
        """
        logger.info("Executando teste de execução paralela...")
        
        # Verificar CUDA
        if not torch.cuda.is_available():
            logger.error("CUDA não disponível. Não é possível realizar testes paralelos.")
            return
        
        # Escolher modelo de tamanho médio para o teste
        model_path = model_info["path"]
        model_name = model_info["name"]
        
        logger.info(f"Usando modelo {model_name} para teste paralelo")
        
        # Limpar cache CUDA
        torch.cuda.empty_cache()
        
        try:
            # Carregar modelos em ambas as GPUs
            model_rtx = UniversalModelWrapper(
                model_path=model_path,
                gpu_id=self.rtx_2060_id,
                gpu_monitor=self.gpu_monitor,
                lazy_load=False
            )
            
            model_gtx = UniversalModelWrapper(
                model_path=model_path,
                gpu_id=self.gtx_1070_id,
                gpu_monitor=self.gpu_monitor,
                lazy_load=False
            )
            
            # Definir prompts diferentes para cada GPU
            prompt_rtx = "O que é uma GPU RTX e quais são suas principais características?"
            prompt_gtx = "Explique as diferenças entre as arquiteturas de GPU Pascal e Turing."
            
            # Executar inferência sequencial (primeiro RTX, depois GTX)
            # RTX 2060
            torch.cuda.synchronize(self.rtx_2060_id)
            start_time = time.time()
            output_rtx_seq = model_rtx.generate(prompt=prompt_rtx, max_tokens=100)
            torch.cuda.synchronize(self.rtx_2060_id)
            rtx_seq_time = time.time() - start_time
            
            # GTX 1070
            torch.cuda.synchronize(self.gtx_1070_id)
            start_time = time.time()
            output_gtx_seq = model_gtx.generate(prompt=prompt_gtx, max_tokens=100)
            torch.cuda.synchronize(self.gtx_1070_id)
            gtx_seq_time = time.time() - start_time
            
            # Tempo total sequencial
            total_seq_time = rtx_seq_time + gtx_seq_time
            
            # Executar inferência paralela usando threads
            import threading
            rtx_output = [None]
            gtx_output = [None]
            rtx_time = [0]
            gtx_time = [0]
            
            def run_rtx():
                torch.cuda.synchronize(self.rtx_2060_id)
                start = time.time()
                rtx_output[0] = model_rtx.generate(prompt=prompt_rtx, max_tokens=100)
                torch.cuda.synchronize(self.rtx_2060_id)
                rtx_time[0] = time.time() - start
            
            def run_gtx():
                torch.cuda.synchronize(self.gtx_1070_id)
                start = time.time()
                gtx_output[0] = model_gtx.generate(prompt=prompt_gtx, max_tokens=100)
                torch.cuda.synchronize(self.gtx_1070_id)
                gtx_time[0] = time.time() - start
            
            # Criar e iniciar threads
            torch.cuda.empty_cache()
            start_time = time.time()
            
            t1 = threading.Thread(target=run_rtx)
            t2 = threading.Thread(target=run_gtx)
            
            t1.start()
            t2.start()
            
            t1.join()
            t2.join()
            
            # Tempo total paralelo
            total_par_time = time.time() - start_time
            
            # Calcular speedup
            speedup = total_seq_time / total_par_time
            
            logger.info(f"Tempo sequencial: {total_seq_time:.2f}s (RTX: {rtx_seq_time:.2f}s, GTX: {gtx_seq_time:.2f}s)")
            logger.info(f"Tempo paralelo: {total_par_time:.2f}s (RTX: {rtx_time[0]:.2f}s, GTX: {gtx_time[0]:.2f}s)")
            logger.info(f"Speedup: {speedup:.2f}x")
            
            # Descarregar modelos
            del model_rtx
            del model_gtx
            torch.cuda.empty_cache()
            
            # Armazenar resultados
            self.results["parallel_test"] = {
                "sequential": {
                    "rtx_time": rtx_seq_time,
                    "gtx_time": gtx_seq_time,
                    "total_time": total_seq_time
                },
                "parallel": {
                    "rtx_time": rtx_time[0],
                    "gtx_time": gtx_time[0],
                    "total_time": total_par_time
                },
                "speedup": speedup
            }
            
        except Exception as e:
            logger.error(f"Erro ao executar teste paralelo: {e}")
            self.results["parallel_test"] = {
                "error": str(e)
            }
    
    def generate_report(self):
        """Gera relatório com os resultados dos testes."""
        logger.info("Gerando relatório de testes...")
        
        # Criar diretório para relatórios
        report_dir = PROJECT_ROOT / "reports" / "gpu_tests"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Nome do arquivo baseado na data e hora
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"rtx2060_vs_gtx1070_report_{timestamp}.html"
        
        # Gerar HTML
        html = """
        <html>
        <head>
            <title>RTX 2060 vs GTX 1070 - Relatório de Testes</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; }
                h3 { color: #16a085; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                tr:hover { background-color: #f5f5f5; }
                .chart-container { width: 100%; height: 400px; margin-bottom: 30px; }
                .rtx { color: #2ecc71; }
                .gtx { color: #e74c3c; }
                .comparison { font-weight: bold; }
                .summary { background-color: #eef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .recommendation { background-color: #efe; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>RTX 2060 vs GTX 1070 - Relatório de Testes</h1>
            <p>Data: """ + time.strftime("%d/%m/%Y %H:%M:%S") + """</p>
            
            <div class="summary">
                <h2>Resumo</h2>
                <p>Este relatório apresenta os resultados de testes comparativos entre as GPUs NVIDIA RTX 2060 e GTX 1070 para uso no sistema EzioFilho_LLMGraph.</p>
                <p>Os testes avaliam o desempenho em inferência de modelos de linguagem, operações de precisão mista (FP16) e execução paralela.</p>
            </div>
        """
        
        # Adicionar resultados do teste de Tensor Cores
        if "tensor_cores_test" in self.results:
            tensor_test = self.results["tensor_cores_test"]
            
            rtx_fp16_time = tensor_test["rtx_2060"]["fp16_time"]
            rtx_fp32_time = tensor_test["rtx_2060"]["fp32_time"]
            rtx_speedup = tensor_test["rtx_2060"]["speedup"]
            
            gtx_fp16_time = tensor_test["gtx_1070"]["fp16_time"]
            gtx_fp32_time = tensor_test["gtx_1070"]["fp32_time"]
            gtx_speedup = tensor_test["gtx_1070"]["speedup"]
            
            rtx_vs_gtx = tensor_test["rtx_vs_gtx_fp16_speedup"]
            
            html += f"""
            <h2>Teste de Operações de Precisão Mista (Tensor Cores)</h2>
            <p>Este teste avalia o desempenho das GPUs em operações de matriz com precisão FP16 vs FP32. 
            A RTX 2060 possui Tensor Cores que aceleram cálculos de precisão mista.</p>
            
            <table>
                <tr>
                    <th>Métrica</th>
                    <th>RTX 2060</th>
                    <th>GTX 1070</th>
                    <th>Comparação</th>
                </tr>
                <tr>
                    <td>Tempo FP16 (s)</td>
                    <td class="rtx">{rtx_fp16_time:.4f}</td>
                    <td class="gtx">{gtx_fp16_time:.4f}</td>
                    <td class="comparison">RTX {rtx_vs_gtx:.2f}x mais rápida</td>
                </tr>
                <tr>
                    <td>Tempo FP32 (s)</td>
                    <td class="rtx">{rtx_fp32_time:.4f}</td>
                    <td class="gtx">{gtx_fp32_time:.4f}</td>
                    <td class="comparison">-</td>
                </tr>
                <tr>
                    <td>Speedup FP16 vs FP32</td>
                    <td class="rtx">{rtx_speedup:.2f}x</td>
                    <td class="gtx">{gtx_speedup:.2f}x</td>
                    <td class="comparison">-</td>
                </tr>
            </table>
            
            <p><strong>Conclusão:</strong> A RTX 2060 mostrou um speedup de {rtx_speedup:.2f}x com FP16, 
            comparado a {gtx_speedup:.2f}x na GTX 1070. Em operações de FP16, a RTX 2060 é {rtx_vs_gtx:.2f}x 
            mais rápida que a GTX 1070, demonstrando o benefício dos Tensor Cores.</p>
            """
        
        # Adicionar resultados do teste de inferência de modelos
        if "model_inference" in self.results:
            html += """
            <h2>Testes de Inferência de Modelos</h2>
            <p>Este teste avalia o desempenho de inferência com diferentes tamanhos de modelos.</p>
            
            <table>
                <tr>
                    <th>Modelo</th>
                    <th>Categoria</th>
                    <th>Tamanho (MB)</th>
                    <th>Métrica</th>
                    <th>RTX 2060</th>
                    <th>GTX 1070</th>
                    <th>Comparação</th>
                </tr>
            """
            
            for model_name, results in self.results["model_inference"].items():
                category = results["category"]
                size_mb = results["size_mb"]
                
                # Verificar se temos resultados completos
                rtx_results = results["rtx_2060"]
                gtx_results = results["gtx_1070"]
                comparison = results.get("comparison", {})
                
                # Tempo de carregamento
                if "load_time" in rtx_results and "load_time" in gtx_results:
                    rtx_load = rtx_results["load_time"]
                    gtx_load = gtx_results["load_time"]
                    load_comp = comparison.get("load_speedup", rtx_load / gtx_load)
                    
                    html += f"""
                    <tr>
                        <td rowspan="3">{model_name}</td>
                        <td rowspan="3">{category}</td>
                        <td rowspan="3">{size_mb:.1f}</td>
                        <td>Tempo de Carregamento (s)</td>
                        <td class="rtx">{rtx_load:.2f}</td>
                        <td class="gtx">{gtx_load:.2f}</td>
                        <td class="comparison">{"RTX" if load_comp > 1 else "GTX"} {max(load_comp, 1/load_comp):.2f}x mais rápida</td>
                    </tr>
                    """
                    
                    # Tempo de inferência
                    rtx_inf = rtx_results.get("inference_time", 0)
                    gtx_inf = gtx_results.get("inference_time", 0)
                    inf_comp = comparison.get("inference_speedup", rtx_inf / gtx_inf if gtx_inf > 0 else 1)
                    
                    html += f"""
                    <tr>
                        <td>Tempo de Inferência (s)</td>
                        <td class="rtx">{rtx_inf:.2f}</td>
                        <td class="gtx">{gtx_inf:.2f}</td>
                        <td class="comparison">{"RTX" if inf_comp > 1 else "GTX"} {max(inf_comp, 1/inf_comp):.2f}x mais rápida</td>
                    </tr>
                    """
                    
                    # Throughput
                    rtx_tps = rtx_results.get("tokens_per_second", 0)
                    gtx_tps = gtx_results.get("tokens_per_second", 0)
                    tps_comp = comparison.get("throughput_ratio", rtx_tps / gtx_tps if gtx_tps > 0 else 1)
                    
                    html += f"""
                    <tr>
                        <td>Throughput (tokens/s)</td>
                        <td class="rtx">{rtx_tps:.2f}</td>
                        <td class="gtx">{gtx_tps:.2f}</td>
                        <td class="comparison">{"RTX" if tps_comp > 1 else "GTX"} {max(tps_comp, 1/tps_comp):.2f}x melhor</td>
                    </tr>
                    """
                else:
                    # Exibir mensagem de erro
                    rtx_error = rtx_results.get("error", "")
                    gtx_error = gtx_results.get("error", "")
                    
                    html += f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{category}</td>
                        <td>{size_mb:.1f}</td>
                        <td colspan="4">Erro no teste: RTX: {rtx_error}, GTX: {gtx_error}</td>
                    </tr>
                    """
            
            html += "</table>"
            
            # Adicionar análise de resultados por categoria
            html += """
            <h3>Análise por Categoria de Modelo</h3>
            <table>
                <tr>
                    <th>Categoria</th>
                    <th>Tempo de Carregamento</th>
                    <th>Tempo de Inferência</th>
                    <th>Throughput</th>
                    <th>Recomendação</th>
                </tr>
            """
            
            # Agrupar resultados por categoria
            categories = {}
            for model_name, results in self.results["model_inference"].items():
                category = results["category"]
                if category not in categories:
                    categories[category] = []
                
                if "comparison" in results:
                    categories[category].append(results["comparison"])
            
            # Gerar linha para cada categoria
            for category, comparisons in categories.items():
                if not comparisons:
                    continue
                
                # Calcular médias
                avg_load = np.mean([comp.get("load_speedup", 1) for comp in comparisons])
                avg_inf = np.mean([comp.get("inference_speedup", 1) for comp in comparisons])
                avg_tps = np.mean([comp.get("throughput_ratio", 1) for comp in comparisons])
                
                # Determinar qual GPU é melhor para esta categoria
                if avg_tps > 1.1:  # RTX é pelo menos 10% melhor
                    recommendation = "RTX 2060"
                    reason = "throughput significativamente melhor"
                elif avg_tps < 0.9:  # GTX é pelo menos 10% melhor
                    recommendation = "GTX 1070"
                    reason = "throughput significativamente melhor"
                else:
                    # Se o throughput é similar, decidir pelo tempo de carregamento
                    if avg_load > 1.1:
                        recommendation = "RTX 2060"
                        reason = "tempo de carregamento melhor com throughput similar"
                    elif avg_load < 0.9:
                        recommendation = "GTX 1070"
                        reason = "tempo de carregamento melhor com throughput similar"
                    else:
                        # Se tudo é similar, preferir RTX pela tecnologia mais recente
                        recommendation = "Ambas GPUs"
                        reason = "desempenho similar"
                
                html += f"""
                <tr>
                    <td><strong>{category.upper()}</strong></td>
                    <td>{"RTX" if avg_load > 1 else "GTX"} {max(avg_load, 1/avg_load):.2f}x mais rápida</td>
                    <td>{"RTX" if avg_inf > 1 else "GTX"} {max(avg_inf, 1/avg_inf):.2f}x mais rápida</td>
                    <td>{"RTX" if avg_tps > 1 else "GTX"} {max(avg_tps, 1/avg_tps):.2f}x melhor</td>
                    <td><strong>{recommendation}</strong><br>({reason})</td>
                </tr>
                """
            
            html += "</table>"
        
        # Adicionar resultados do teste paralelo
        if "parallel_test" in self.results and "error" not in self.results["parallel_test"]:
            parallel = self.results["parallel_test"]
            
            seq_rtx = parallel["sequential"]["rtx_time"]
            seq_gtx = parallel["sequential"]["gtx_time"]
            seq_total = parallel["sequential"]["total_time"]
            
            par_rtx = parallel["parallel"]["rtx_time"]
            par_gtx = parallel["parallel"]["gtx_time"]
            par_total = parallel["parallel"]["total_time"]
            
            speedup = parallel["speedup"]
            
            html += f"""
            <h2>Teste de Execução Paralela</h2>
            <p>Este teste avalia o desempenho do sistema quando modelos são executados simultaneamente em ambas as GPUs.</p>
            
            <table>
                <tr>
                    <th>Execução</th>
                    <th>RTX 2060 (s)</th>
                    <th>GTX 1070 (s)</th>
                    <th>Tempo Total (s)</th>
                </tr>
                <tr>
                    <td>Sequencial</td>
                    <td>{seq_rtx:.2f}</td>
                    <td>{seq_gtx:.2f}</td>
                    <td>{seq_total:.2f}</td>
                </tr>
                <tr>
                    <td>Paralela</td>
                    <td>{par_rtx:.2f}</td>
                    <td>{par_gtx:.2f}</td>
                    <td>{par_total:.2f}</td>
                </tr>
            </table>
            
            <p><strong>Speedup com execução paralela: {speedup:.2f}x</strong></p>
            
            <p>A execução simultânea de modelos em ambas as GPUs resultou em um speedup de {speedup:.2f}x 
            comparado à execução sequencial. Isso demonstra o benefício de utilizar múltiplas GPUs para 
            processar requisições diferentes em paralelo.</p>
            """
        
        # Adicionar recomendações finais
        html += """
        <div class="recommendation">
            <h2>Recomendações</h2>
            
            <h3>Alocação de Modelos</h3>
            <ul>
        """
        
        # Analisar resultados para recomendações
        small_models_rec = "RTX 2060" if "small" in categories and np.mean([comp.get("throughput_ratio", 1) for comp in categories["small"]]) > 1 else "GTX 1070"
        medium_models_rec = "RTX 2060" if "medium" in categories and np.mean([comp.get("throughput_ratio", 1) for comp in categories["medium"]]) > 1 else "GTX 1070"
        large_models_rec = "RTX 2060" if "large" in categories and np.mean([comp.get("throughput_ratio", 1) for comp in categories["large"]]) > 1 else "GTX 1070"
        xlarge_models_rec = "RTX 2060" if "xlarge" in categories and np.mean([comp.get("throughput_ratio", 1) for comp in categories["xlarge"]]) > 1 else "GTX 1070"
        
        # Se RTX for melhor para modelos grandes, recomendar RTX para modelos grandes
        # e GTX para modelos pequenos/médios para balancear
        if large_models_rec == "RTX 2060" and "tensor_cores_test" in self.results and self.results["tensor_cores_test"]["rtx_vs_gtx_fp16_speedup"] > 1.2:
            html += f"""
                <li><strong>Modelos grandes/XL:</strong> Priorize a <strong class="rtx">RTX 2060</strong> para modelos grandes e XL, aproveitando seus Tensor Cores e melhor desempenho em FP16.</li>
                <li><strong>Modelos médios:</strong> Distribua entre as duas GPUs conforme disponibilidade, com leve preferência para a <strong class="rtx">RTX 2060</strong>.</li>
                <li><strong>Modelos pequenos:</strong> Priorize a <strong class="gtx">GTX 1070</strong> para modelos pequenos, deixando a RTX livre para modelos maiores.</li>
            """
        else:
            html += f"""
                <li><strong>Modelos pequenos:</strong> Priorize a <strong class="{small_models_rec.lower()}">{small_models_rec}</strong> para modelos pequenos.</li>
                <li><strong>Modelos médios:</strong> Priorize a <strong class="{medium_models_rec.lower()}">{medium_models_rec}</strong> para modelos médios.</li>
                <li><strong>Modelos grandes:</strong> Priorize a <strong class="{large_models_rec.lower()}">{large_models_rec}</strong> para modelos grandes.</li>
                <li><strong>Modelos XL:</strong> Priorize a <strong class="{xlarge_models_rec.lower()}">{xlarge_models_rec}</strong> para modelos extra grandes.</li>
            """
        
        html += """
            </ul>
            
            <h3>Otimizações</h3>
            <ul>
                <li><strong>Precision:</strong> Utilize FP16 (half precision) quando possível na RTX 2060 para aproveitar os Tensor Cores.</li>
                <li><strong>Execução paralela:</strong> Configure o sistema para processar requisições simultaneamente em ambas as GPUs.</li>
                <li><strong>Balanceamento de carga:</strong> Implemente um mecanismo dinâmico que aloque requisições para a GPU menos ocupada.</li>
                <li><strong>Monitoramento:</strong> Monitore constantemente o uso de memória e atividade em ambas as GPUs.</li>
                <li><strong>Descarregamento adaptativo:</strong> Implemente um sistema que descarregue modelos menos utilizados quando a pressão de memória for alta.</li>
            </ul>
            
            <h3>Configuração de Sistema</h3>
            <ul>
                <li><strong>RTX 2060 (GPU 0):</strong> Configure como GPU primária para modelos que se beneficiam de Tensor Cores.</li>
                <li><strong>GTX 1070 (GPU 1):</strong> Configure como GPU secundária, ideal para modelos menores e processamento paralelo.</li>
                <li><strong>Fallbacks:</strong> Configure fallbacks automáticos para CPU em caso de sobrecarga de ambas as GPUs.</li>
            </ul>
        </div>
        </body>
        </html>
        """
        
        # Salvar relatório
        with open(report_file, "w") as f:
            f.write(html)
        
        logger.info(f"Relatório salvo em {report_file}")
        
        # Salvar resultados brutos em JSON
        import json
        json_file = report_dir / f"rtx2060_vs_gtx1070_results_{timestamp}.json"
        
        with open(json_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Resultados brutos salvos em {json_file}")
    
    def run_all_tests(self):
        """Executa todos os testes."""
        logger.info("Iniciando testes de comparação RTX 2060 vs GTX 1070...")
        
        # Descobrir modelos para teste
        models = self.discover_test_models()
        
        if not models:
            logger.error("Sem modelos para testar. Abortando.")
            return
        
        # Teste de Tensor Cores
        self.run_tensor_cores_test()
        
        # Teste de inferência de modelos
        self.run_inference_test(models)
        
        # Teste de execução paralela (usando modelo médio se disponível)
        medium_models = [m for m in models if m["category"] == "medium"]
        if medium_models:
            self.run_parallel_test(medium_models[0])
        elif models:
            self.run_parallel_test(models[0])
        
        # Gerar relatório
        self.generate_report()
        
        logger.info("Testes concluídos.")

def main():
    """Função principal."""
    logger.info("Iniciando teste de comparação RTX 2060 vs GTX 1070")
    
    # Criar e executar teste
    test = GPUComparisonTest()
    test.run_all_tests()

if __name__ == "__main__":
    main()
