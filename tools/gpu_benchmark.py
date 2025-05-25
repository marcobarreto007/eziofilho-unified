#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU Benchmark - Ferramenta para testar desempenho entre diferentes GPUs
---------------------------------------------------------------------
Esta ferramenta permite:
- Comparar desempenho entre RTX 2060 e GTX 1070
- Medir tempo de carregamento e inferência em cada GPU
- Analisar características específicas de cada modelo
- Fornecer recomendações de alocação de modelos

Autor: EzioFilho LLMGraph Team
Data: Maio/2025
"""

import os
import sys
import time
import argparse
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
logger = logging.getLogger("GPUBenchmark")

class GPUBenchmark:
    """
    Ferramenta para benchmark de GPUs em sistemas com RTX 2060 e GTX 1070.
    """
    
    def __init__(self, gpu_ids: Optional[List[int]] = None, model_paths: Optional[List[str]] = None):
        """
        Inicializa o benchmark.
        
        Args:
            gpu_ids: Lista de IDs das GPUs para testar (None para todas)
            model_paths: Lista de caminhos para os modelos a testar (None para usar modelos padrão)
        """
        # Verificar CUDA
        if not torch.cuda.is_available():
            logger.error("CUDA não disponível. Não é possível realizar benchmark de GPU.")
            sys.exit(1)
            
        # Inicializar monitor de GPU
        self.gpu_monitor = get_gpu_monitor(gpu_ids=gpu_ids)
        self.gpu_ids = self.gpu_monitor.gpu_ids if self.gpu_monitor else []
        
        if not self.gpu_ids:
            logger.error("Nenhuma GPU detectada para benchmark.")
            sys.exit(1)
            
        logger.info(f"GPUs disponíveis para benchmark: {self.gpu_ids}")
        
        # Lista de modelos para testar
        # Se não for fornecida, usar modelos padrão
        self.model_paths = model_paths or []
        
        # Se nenhum modelo fornecido, procurar modelos disponíveis
        if not self.model_paths:
            self._discover_available_models()
            
        # Resultados do benchmark
        self.results = {
            "load_time": {},
            "inference_time": {},
            "memory_usage": {},
            "throughput": {}
        }
        
    def _discover_available_models(self):
        """Descobre modelos disponíveis no sistema."""
        model_dirs = [
            Path(PROJECT_ROOT) / "models",  # Diretório padrão de modelos
            Path.home() / ".cache" / "models",  # Cache padrão de modelos
        ]
        
        # Procurar por modelos GGUF e outros formatos comuns
        extensions = [".gguf", ".bin", ".safetensors", ".onnx"]
        models_found = []
        
        for model_dir in model_dirs:
            if not model_dir.exists():
                continue
                
            for ext in extensions:
                for model_file in model_dir.glob(f"**/*{ext}"):
                    models_found.append(str(model_file))
        
        # Filtrar modelos por tamanho e outros critérios
        final_models = []
        for model_path in models_found:
            try:
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                # Adicionar apenas modelos com tamanho entre 500MB e 10GB
                if 500 <= size_mb <= 10000:
                    final_models.append(model_path)
                    logger.info(f"Modelo encontrado: {Path(model_path).name} ({size_mb:.1f} MB)")
            except Exception as e:
                logger.warning(f"Erro ao verificar tamanho do modelo {model_path}: {e}")
        
        # Ordenar por tamanho e selecionar até 5 modelos
        if final_models:
            final_models.sort(key=lambda x: os.path.getsize(x))
            self.model_paths = final_models[:5]
        else:
            logger.warning("Nenhum modelo encontrado automaticamente. Por favor forneça caminhos explícitos.")
            
    def run_load_time_benchmark(self, gpu_id: int, model_path: str) -> Tuple[float, float]:
        """
        Testa o tempo de carregamento de um modelo em uma GPU.
        
        Args:
            gpu_id: ID da GPU para teste
            model_path: Caminho para o modelo
            
        Returns:
            Tupla (tempo_carregamento, uso_memoria)
        """
        logger.info(f"Testando carregamento de {Path(model_path).name} na GPU {gpu_id}")
        
        # Limpar cache CUDA
        torch.cuda.empty_cache()
        
        try:
            # Obter uso de memória antes
            metrics_before = self.gpu_monitor.get_current_metrics().get(gpu_id, {})
            mem_before = metrics_before.get("mem_allocated_mb", 0)
            
            # Medir tempo de carregamento
            start_time = time.time()
            
            # Criar wrapper universal
            wrapper = UniversalModelWrapper(
                model_path=model_path,
                gpu_id=gpu_id,
                gpu_monitor=self.gpu_monitor,
                lazy_load=False  # Carregar imediatamente
            )
            
            # Registrar tempo
            load_time = time.time() - start_time
            
            # Obter uso de memória após
            metrics_after = self.gpu_monitor.get_current_metrics().get(gpu_id, {})
            mem_after = metrics_after.get("mem_allocated_mb", 0)
            
            # Calcular uso de memória
            memory_usage = mem_after - mem_before
            
            logger.info(f"Modelo carregado em {load_time:.2f}s, uso de memória: {memory_usage:.2f} MB")
            
            # Descarregar modelo para liberar memória
            del wrapper
            torch.cuda.empty_cache()
            
            return load_time, memory_usage
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {model_path}: {e}")
            return -1, -1
            
    def run_inference_benchmark(self, gpu_id: int, model_path: str, num_runs: int = 5) -> Tuple[float, float]:
        """
        Testa o tempo de inferência de um modelo em uma GPU.
        
        Args:
            gpu_id: ID da GPU para teste
            model_path: Caminho para o modelo
            num_runs: Número de execuções para média
            
        Returns:
            Tupla (tempo_medio_inferencia, throughput)
        """
        logger.info(f"Testando inferência de {Path(model_path).name} na GPU {gpu_id}")
        
        # Limpar cache CUDA
        torch.cuda.empty_cache()
        
        try:
            # Criar wrapper universal
            wrapper = UniversalModelWrapper(
                model_path=model_path,
                gpu_id=gpu_id,
                gpu_monitor=self.gpu_monitor,
                lazy_load=False  # Carregar imediatamente
            )
            
            # Prompt de teste
            test_prompt = "Explique o conceito de aprendizado de máquina em termos simples."
            
            # Lista para armazenar tempos
            inference_times = []
            throughput_list = []
            
            # Executar inferência múltiplas vezes
            for i in range(num_runs):
                # Medir tempo de inferência
                start_time = time.time()
                
                response = wrapper.generate(
                    test_prompt,
                    max_tokens=100,  # Resposta curta para teste
                    temperature=0.7
                )
                
                # Registrar tempo
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Calcular throughput (tokens/segundo)
                response_length = len(response.split())
                throughput = response_length / inference_time
                throughput_list.append(throughput)
                
                logger.info(f"Execução {i+1}/{num_runs}: {inference_time:.2f}s, throughput: {throughput:.2f} tokens/s")
                
                # Aguardar um pouco entre execuções
                time.sleep(1)
            
            # Calcular médias
            avg_inference_time = np.mean(inference_times)
            avg_throughput = np.mean(throughput_list)
            
            logger.info(f"Média: {avg_inference_time:.2f}s, throughput médio: {avg_throughput:.2f} tokens/s")
            
            # Descarregar modelo para liberar memória
            del wrapper
            torch.cuda.empty_cache()
            
            return avg_inference_time, avg_throughput
            
        except Exception as e:
            logger.error(f"Erro ao executar inferência para {model_path}: {e}")
            return -1, -1
            
    def run_benchmark(self):
        """Executa o benchmark completo para todas as GPUs e modelos."""
        logger.info("Iniciando benchmark de GPUs")
        
        if not self.model_paths:
            logger.error("Nenhum modelo disponível para benchmark.")
            return
            
        # Obter informações sobre as GPUs
        gpu_info = {}
        for gpu_id in self.gpu_ids:
            metrics = self.gpu_monitor.get_current_metrics().get(gpu_id, {})
            gpu_info[gpu_id] = {
                "name": metrics.get("name", f"GPU {gpu_id}"),
                "total_memory_mb": metrics.get("total_memory_mb", 0)
            }
            logger.info(f"GPU {gpu_id}: {gpu_info[gpu_id]['name']} ({gpu_info[gpu_id]['total_memory_mb']:.0f} MB)")
        
        # Executar benchmark para cada combinação de GPU e modelo
        for model_path in self.model_paths:
            model_name = Path(model_path).name
            
            for gpu_id in self.gpu_ids:
                # Benchmark de tempo de carregamento
                load_time, memory_usage = self.run_load_time_benchmark(gpu_id, model_path)
                
                # Armazenar resultados
                if model_name not in self.results["load_time"]:
                    self.results["load_time"][model_name] = {}
                    self.results["memory_usage"][model_name] = {}
                
                self.results["load_time"][model_name][gpu_id] = load_time
                self.results["memory_usage"][model_name][gpu_id] = memory_usage
                
                # Benchmark de inferência
                inference_time, throughput = self.run_inference_benchmark(gpu_id, model_path)
                
                # Armazenar resultados
                if model_name not in self.results["inference_time"]:
                    self.results["inference_time"][model_name] = {}
                    self.results["throughput"][model_name] = {}
                
                self.results["inference_time"][model_name][gpu_id] = inference_time
                self.results["throughput"][model_name][gpu_id] = throughput
                
                # Limpar cache CUDA entre testes
                torch.cuda.empty_cache()
                time.sleep(2)  # Aguardar para estabilizar
        
        # Gerar relatório com os resultados
        self.generate_report(gpu_info)
    
    def generate_report(self, gpu_info: Dict[int, Dict[str, Any]]):
        """
        Gera relatório com os resultados do benchmark.
        
        Args:
            gpu_info: Informações sobre as GPUs testadas
        """
        logger.info("Gerando relatório de benchmark")
        
        # Criar diretório para relatórios
        report_dir = PROJECT_ROOT / "reports" / "benchmark"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Nome do arquivo baseado na data e hora
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"gpu_benchmark_{timestamp}.html"
        csv_file = report_dir / f"gpu_benchmark_{timestamp}.csv"
        
        # Preparar dados para o relatório
        report_data = []
        
        for model_name in self.results["load_time"]:
            for gpu_id in self.gpu_ids:
                if gpu_id in self.results["load_time"][model_name]:
                    row = {
                        "model_name": model_name,
                        "gpu_id": gpu_id,
                        "gpu_name": gpu_info[gpu_id]["name"],
                        "load_time": self.results["load_time"][model_name][gpu_id],
                        "memory_usage": self.results["memory_usage"][model_name][gpu_id],
                        "inference_time": self.results["inference_time"][model_name][gpu_id],
                        "throughput": self.results["throughput"][model_name][gpu_id]
                    }
                    report_data.append(row)
        
        # Converter para DataFrame
        df = pd.DataFrame(report_data)
        
        # Salvar CSV
        df.to_csv(csv_file, index=False)
        logger.info(f"Dados de benchmark salvos em {csv_file}")
        
        # Gerar relatório HTML
        html = """
        <html>
        <head>
            <title>GPU Benchmark Report - EzioFilho_LLMGraph</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                tr:hover { background-color: #f5f5f5; }
                .chart-container { width: 100%; height: 400px; margin-bottom: 30px; }
                .summary { background-color: #eef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .recommendation { background-color: #efe; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>GPU Benchmark Report - EzioFilho_LLMGraph</h1>
            <p>Data: """ + time.strftime("%d/%m/%Y %H:%M:%S") + """</p>
            
            <h2>Informações das GPUs</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Nome</th>
                    <th>Memória Total (MB)</th>
                </tr>
        """
        
        # Adicionar informações das GPUs
        for gpu_id, info in gpu_info.items():
            html += f"""
                <tr>
                    <td>{gpu_id}</td>
                    <td>{info['name']}</td>
                    <td>{info['total_memory_mb']:.0f}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Resultados do Benchmark</h2>
            <table>
                <tr>
                    <th>Modelo</th>
                    <th>GPU</th>
                    <th>Tempo de Carregamento (s)</th>
                    <th>Uso de Memória (MB)</th>
                    <th>Tempo de Inferência (s)</th>
                    <th>Throughput (tokens/s)</th>
                </tr>
        """
        
        # Adicionar resultados
        for row in report_data:
            html += f"""
                <tr>
                    <td>{row['model_name']}</td>
                    <td>{row['gpu_name']} (ID: {row['gpu_id']})</td>
                    <td>{row['load_time']:.2f}</td>
                    <td>{row['memory_usage']:.2f}</td>
                    <td>{row['inference_time']:.2f}</td>
                    <td>{row['throughput']:.2f}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Análise de Desempenho por GPU</h2>
            <div class="summary">
        """
        
        # Análise de desempenho
        for gpu_id in self.gpu_ids:
            gpu_name = gpu_info[gpu_id]["name"]
            
            # Calcular médias
            load_times = [row["load_time"] for row in report_data if row["gpu_id"] == gpu_id and row["load_time"] > 0]
            inference_times = [row["inference_time"] for row in report_data if row["gpu_id"] == gpu_id and row["inference_time"] > 0]
            throughputs = [row["throughput"] for row in report_data if row["gpu_id"] == gpu_id and row["throughput"] > 0]
            
            if load_times and inference_times and throughputs:
                avg_load = np.mean(load_times)
                avg_inference = np.mean(inference_times)
                avg_throughput = np.mean(throughputs)
                
                html += f"""
                    <h3>GPU {gpu_id}: {gpu_name}</h3>
                    <ul>
                        <li>Tempo médio de carregamento: {avg_load:.2f}s</li>
                        <li>Tempo médio de inferência: {avg_inference:.2f}s</li>
                        <li>Throughput médio: {avg_throughput:.2f} tokens/s</li>
                    </ul>
                """
        
        html += """
            </div>
            
            <h2>Recomendações para Alocação de Modelos</h2>
            <div class="recommendation">
        """
        
        # Gerar recomendações para alocação de modelos
        if len(self.gpu_ids) > 1:
            # Comparar GPUs por throughput médio
            gpu_throughputs = {}
            for gpu_id in self.gpu_ids:
                throughputs = [row["throughput"] for row in report_data if row["gpu_id"] == gpu_id and row["throughput"] > 0]
                if throughputs:
                    gpu_throughputs[gpu_id] = np.mean(throughputs)
            
            # Ordenar GPUs por throughput (maior primeiro)
            sorted_gpus = sorted(gpu_throughputs.items(), key=lambda x: -x[1])
            
            if sorted_gpus:
                fastest_gpu = sorted_gpus[0][0]
                fastest_gpu_name = gpu_info[fastest_gpu]["name"]
                
                # Analisar modelo mais rápido em cada GPU
                model_perf_by_gpu = {}
                for gpu_id in self.gpu_ids:
                    model_throughputs = {row["model_name"]: row["throughput"] for row in report_data if row["gpu_id"] == gpu_id and row["throughput"] > 0}
                    if model_throughputs:
                        best_model = max(model_throughputs.items(), key=lambda x: x[1])
                        model_perf_by_gpu[gpu_id] = best_model
                
                html += f"""
                    <h3>Recomendações Baseadas em Desempenho</h3>
                    <p>GPU mais rápida em média: <strong>{fastest_gpu_name} (ID: {fastest_gpu})</strong></p>
                    <p>Recomendações de alocação de modelos:</p>
                    <ul>
                """
                
                # Recomendar modelos maiores para GPU mais rápida
                large_models = []
                small_models = []
                
                for model_name in set(row["model_name"] for row in report_data):
                    # Calcular tamanho aproximado do modelo pelo uso de memória
                    memory_usages = [row["memory_usage"] for row in report_data if row["model_name"] == model_name and row["memory_usage"] > 0]
                    if memory_usages:
                        avg_memory = np.mean(memory_usages)
                        if avg_memory > 2000:  # Mais de 2GB
                            large_models.append(model_name)
                        else:
                            small_models.append(model_name)
                
                if large_models:
                    html += f"""
                        <li>Modelos grandes (>2GB) como <strong>{', '.join(large_models)}</strong> devem ser alocados na GPU mais rápida: <strong>{fastest_gpu_name} (ID: {fastest_gpu})</strong></li>
                    """
                
                # Recomendar outros modelos para GPUs alternativas
                if len(self.gpu_ids) > 1 and small_models:
                    other_gpus = [gpu_id for gpu_id in self.gpu_ids if gpu_id != fastest_gpu]
                    other_gpu_names = [gpu_info[gpu_id]["name"] for gpu_id in other_gpus]
                    
                    html += f"""
                        <li>Modelos menores como <strong>{', '.join(small_models)}</strong> podem ser alocados nas GPUs alternativas: <strong>{', '.join(f"{name} (ID: {gpu_id})" for name, gpu_id in zip(other_gpu_names, other_gpus))}</strong></li>
                    """
                
                html += """
                    </ul>
                    <h3>Estratégia de Balanceamento Recomendada</h3>
                    <ol>
                        <li>Priorize a alocação de modelos grandes e frequentemente utilizados na GPU mais rápida.</li>
                        <li>Use a GPU mais lenta para modelos menores e tarefas de pré/pós-processamento.</li>
                        <li>Configure o sistema para monitorar uso de memória e throughput em tempo real.</li>
                        <li>Implemente realocação dinâmica baseada em padrões de uso.</li>
                    </ol>
                """
        else:
            html += """
                <p>Apenas uma GPU detectada. Para recomendações de balanceamento entre GPUs, execute o benchmark com múltiplas GPUs.</p>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        # Salvar relatório HTML
        with open(report_file, "w") as f:
            f.write(html)
        
        logger.info(f"Relatório de benchmark salvo em {report_file}")
        
        # Gerar gráficos
        self._generate_charts(report_dir, timestamp, report_data)
    
    def _generate_charts(self, report_dir: Path, timestamp: str, report_data: List[Dict[str, Any]]):
        """
        Gera gráficos comparativos para o benchmark.
        
        Args:
            report_dir: Diretório para salvar os gráficos
            timestamp: Timestamp para nomes de arquivos
            report_data: Dados do relatório
        """
        try:
            # Criar DataFrame para facilitar a geração de gráficos
            df = pd.DataFrame(report_data)
            
            # Ordenar por modelo e GPU
            df = df.sort_values(["model_name", "gpu_id"])
            
            # Criar diretório para gráficos
            charts_dir = report_dir / "charts"
            charts_dir.mkdir(exist_ok=True)
            
            # Cores para cada GPU
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
            
            # 1. Gráfico de tempo de carregamento por modelo e GPU
            plt.figure(figsize=(12, 8))
            
            # Agrupar por modelo e GPU
            grouped = df.groupby(['model_name', 'gpu_name'])['load_time'].mean().unstack()
            
            ax = grouped.plot(kind='bar', color=colors[:len(self.gpu_ids)])
            plt.title('Tempo de Carregamento por Modelo e GPU')
            plt.xlabel('Modelo')
            plt.ylabel('Tempo (segundos)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.savefig(charts_dir / f"load_time_{timestamp}.png", dpi=300)
            
            # 2. Gráfico de throughput por modelo e GPU
            plt.figure(figsize=(12, 8))
            
            # Agrupar por modelo e GPU
            grouped = df.groupby(['model_name', 'gpu_name'])['throughput'].mean().unstack()
            
            ax = grouped.plot(kind='bar', color=colors[:len(self.gpu_ids)])
            plt.title('Throughput por Modelo e GPU')
            plt.xlabel('Modelo')
            plt.ylabel('Throughput (tokens/segundo)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.savefig(charts_dir / f"throughput_{timestamp}.png", dpi=300)
            
            # 3. Gráfico de uso de memória por modelo e GPU
            plt.figure(figsize=(12, 8))
            
            # Agrupar por modelo e GPU
            grouped = df.groupby(['model_name', 'gpu_name'])['memory_usage'].mean().unstack()
            
            ax = grouped.plot(kind='bar', color=colors[:len(self.gpu_ids)])
            plt.title('Uso de Memória por Modelo e GPU')
            plt.xlabel('Modelo')
            plt.ylabel('Memória (MB)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.savefig(charts_dir / f"memory_usage_{timestamp}.png", dpi=300)
            
            logger.info(f"Gráficos salvos no diretório {charts_dir}")
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráficos: {e}")

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Benchmark de GPUs para EzioFilho_LLMGraph")
    parser.add_argument("--gpu-ids", help="IDs das GPUs para testar (separados por vírgula, default: todas)")
    parser.add_argument("--models", help="Caminhos para modelos a testar (separados por vírgula)")
    
    args = parser.parse_args()
    
    # Processar IDs de GPU
    gpu_ids = None
    if args.gpu_ids:
        try:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
            logger.info(f"Usando GPUs especificadas: {gpu_ids}")
        except ValueError:
            logger.error(f"Formato inválido para IDs de GPU: {args.gpu_ids}")
            logger.info("Formato esperado: números separados por vírgula (ex: 0,1)")
            sys.exit(1)
    
    # Processar caminhos de modelos
    model_paths = None
    if args.models:
        model_paths = [x.strip() for x in args.models.split(",")]
        logger.info(f"Usando modelos especificados: {model_paths}")
    
    # Inicializar e executar benchmark
    benchmark = GPUBenchmark(gpu_ids=gpu_ids, model_paths=model_paths)
    benchmark.run_benchmark()

if __name__ == "__main__":
    main()
