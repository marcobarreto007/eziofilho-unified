#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EzioFilho_LLMGraph - Painel Avançado de Monitoramento Multi-GPU
---------------------------------------------------------------
Interface web para monitoramento em tempo real de GPUs RTX 2060 e GTX 1070,
exibindo métricas, modelos carregados e recomendações de otimização.

Autor: EzioFilho LLMGraph Team
Data: Maio/2025
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

# Adicionar diretório pai ao path para importações relativas
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Importar módulos do projeto
from core.gpu_monitor import get_gpu_monitor
from core.multi_gpu_manager import get_multi_gpu_manager

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("GPUDashboard")

class DashboardManager:
    """Gerenciador da interface de monitoramento de GPUs."""
    
    def __init__(self):
        # Inicializar o monitor de GPU
        self.gpu_monitor = get_gpu_monitor()
        self.gpu_monitor.start()
        
        # Tentar obter o gerenciador Multi-GPU (pode não estar disponível)
        try:
            self.gpu_manager = get_multi_gpu_manager()
            self.has_gpu_manager = True
        except Exception as e:
            logger.warning(f"Gerenciador Multi-GPU não disponível: {e}")
            self.gpu_manager = None
            self.has_gpu_manager = False
            
        # Histórico para gráficos
        self.memory_history = {gpu_id: [] for gpu_id in self.gpu_monitor.gpu_ids}
        self.time_labels = []
        self.history_max_size = 120  # 2 minutos com atualização a cada segundo
        
        # Informações sobre GPUs específicas
        self.specific_gpus = self.gpu_monitor.detect_rtx2060_gtx1070()
        
        # Flag para controle de atualização
        self.is_running = True
        self.update_interval = 1.0  # segundos
        
        # Iniciar thread de atualização
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def _update_loop(self):
        """Loop de atualização para coletar dados históricos."""
        while self.is_running:
            try:
                # Coletar métricas atuais
                current_metrics = self.gpu_monitor.get_current_metrics()
                current_time = time.strftime("%H:%M:%S")
                
                # Atualizar históricos
                self.time_labels.append(current_time)
                for gpu_id in self.memory_history.keys():
                    if gpu_id in current_metrics:
                        mem_util = current_metrics[gpu_id].get("mem_utilization", 0)
                        self.memory_history[gpu_id].append(mem_util)
                    else:
                        # Preencher com zero se a GPU não estiver disponível
                        self.memory_history[gpu_id].append(0)
                
                # Limitar tamanho do histórico
                if len(self.time_labels) > self.history_max_size:
                    self.time_labels = self.time_labels[-self.history_max_size:]
                    for gpu_id in self.memory_history:
                        if len(self.memory_history[gpu_id]) > self.history_max_size:
                            self.memory_history[gpu_id] = self.memory_history[gpu_id][-self.history_max_size:]
                
                # Aguardar próxima atualização
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Erro no loop de atualização: {e}")
                time.sleep(self.update_interval)
    
    def stop(self):
        """Para o loop de atualização."""
        self.is_running = False
        if self.update_thread.is_alive():
            self.update_thread.join(timeout=3.0)
    
    def get_gpu_info_html(self) -> str:
        """
        Gera HTML com informações detalhadas das GPUs.
        
        Returns:
            String HTML formatada
        """
        html = "<h2>Informações Detalhadas das GPUs</h2>"
        
        # Verificar se há GPUs específicas (RTX 2060, GTX 1070)
        if self.specific_gpus:
            html += "<h3>GPUs Específicas Detectadas</h3>"
            html += "<table style='width:100%; border-collapse: collapse;'>"
            html += "<tr style='background-color:#f2f2f2;'><th>GPU ID</th><th>Nome</th><th>Tipo</th><th>Tensor Cores</th><th>CUDA</th><th>Memória Total</th></tr>"
            
            for gpu_id, info in self.specific_gpus.items():
                tensor_cores = "Sim" if info.get("has_tensor_cores", False) else "Não"
                memory_mb = info.get("memory_mb", 0)
                memory_gb = memory_mb / 1024
                
                html += f"<tr><td>{gpu_id}</td><td>{info['name']}</td><td>{info['gpu_type']}</td>"
                html += f"<td>{tensor_cores}</td><td>{info.get('cuda_capability', 'N/A')}</td>"
                html += f"<td>{memory_gb:.2f} GB</td></tr>"
                
            html += "</table>"
            
            # Adicionar recomendações
            html += "<h3>Modelos Recomendados por GPU</h3>"
            html += "<table style='width:100%; border-collapse: collapse;'>"
            html += "<tr style='background-color:#f2f2f2;'><th>GPU</th><th>Modelos Recomendados</th><th>Tipos de Modelo</th></tr>"
            
            for gpu_id, info in self.specific_gpus.items():
                recommended = info.get("recommended_models", [])
                model_types = info.get("model_types", [])
                
                html += f"<tr><td>{info['name']}</td>"
                html += f"<td>{', '.join(recommended)}</td>"
                html += f"<td>{', '.join(model_types)}</td></tr>"
                
            html += "</table>"
        else:
            html += "<p>Nenhuma GPU RTX 2060 ou GTX 1070 detectada. Usando configuração genérica.</p>"
        
        # Métricas atuais
        metrics = self.gpu_monitor.get_current_metrics()
        
        if metrics:
            html += "<h3>Uso Atual de Memória</h3>"
            html += "<table style='width:100%; border-collapse: collapse;'>"
            html += "<tr style='background-color:#f2f2f2;'><th>GPU ID</th><th>Nome</th><th>Memória Usada</th><th>Memória Livre</th><th>Utilização</th></tr>"
            
            for gpu_id, gpu_metrics in metrics.items():
                name = gpu_metrics.get("name", f"GPU {gpu_id}")
                used_mb = gpu_metrics.get("mem_allocated_mb", 0)
                free_mb = gpu_metrics.get("memory_free_mb", 0)
                total_mb = gpu_metrics.get("total_memory_mb", 0)
                util_pct = gpu_metrics.get("mem_utilization", 0)
                
                # Determinar cor baseada no uso (verde < 70%, amarelo < 85%, vermelho >= 85%)
                if util_pct >= 85:
                    color = "#ffcccc"  # Vermelho claro
                elif util_pct >= 70:
                    color = "#ffffcc"  # Amarelo claro
                else:
                    color = "#ccffcc"  # Verde claro
                
                html += f"<tr style='background-color:{color};'>"
                html += f"<td>{gpu_id}</td><td>{name}</td>"
                html += f"<td>{used_mb:.1f} MB / {total_mb:.1f} MB</td>"
                html += f"<td>{free_mb:.1f} MB</td>"
                html += f"<td>{util_pct:.1f}%</td></tr>"
                
            html += "</table>"
        else:
            html += "<p>Nenhuma métrica de GPU disponível.</p>"
        
        return html
    
    def get_gpu_usage_chart(self) -> plt.Figure:
        """
        Gera um gráfico de uso de memória das GPUs.
        
        Returns:
            Figura matplotlib
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plotar histórico para cada GPU
        for gpu_id, history in self.memory_history.items():
            if not history:
                continue
                
            # Obter nome da GPU
            gpu_name = f"GPU {gpu_id}"
            metrics = self.gpu_monitor.get_current_metrics().get(gpu_id, {})
            if "name" in metrics:
                gpu_name = metrics["name"]
            elif gpu_id in self.specific_gpus:
                gpu_name = self.specific_gpus[gpu_id]["name"]
            
            # Determinar cor baseada no tipo de GPU
            color = "blue"
            if gpu_id in self.specific_gpus:
                gpu_type = self.specific_gpus[gpu_id]["gpu_type"]
                color = "green" if gpu_type == "rtx_2060" else "orange"
            
            ax.plot(history, label=gpu_name, color=color, linewidth=2)
        
        # Configurar gráfico
        ax.set_title("Histórico de Utilização de Memória das GPUs")
        ax.set_xlabel("Tempo (últimos 2 minutos)")
        ax.set_ylabel("Utilização de Memória (%)")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_ylim(0, 100)
        
        # Se tivermos labels de tempo suficientes, mostrar alguns
        if len(self.time_labels) > 0:
            indices = np.linspace(0, len(self.time_labels) - 1, min(10, len(self.time_labels))).astype(int)
            ax.set_xticks(indices)
            ax.set_xticklabels([self.time_labels[i] for i in indices], rotation=45)
        
        ax.legend(loc="upper left")
        plt.tight_layout()
        
        return fig
    
    def get_model_info_html(self) -> str:
        """
        Gera HTML com informações dos modelos carregados.
        
        Returns:
            String HTML formatada
        """
        html = "<h2>Modelos Carregados</h2>"
        
        if not self.has_gpu_manager:
            html += "<p>Gerenciador Multi-GPU não disponível. Informações de modelo não disponíveis.</p>"
            return html
        
        # Obter status atual
        status = self.gpu_manager.get_gpu_status()
        
        if "models" in status and status["models"]:
            html += "<table style='width:100%; border-collapse: collapse;'>"
            html += "<tr style='background-color:#f2f2f2;'><th>Modelo</th><th>Tamanho</th><th>GPU</th><th>Último Uso</th><th>Contagem de Uso</th></tr>"
            
            # Ordenar modelos por recência de uso
            models_sorted = sorted(
                status["models"].items(),
                key=lambda x: x[1]["time_since_last_use"]
            )
            
            for model_id, info in models_sorted:
                # Formatações
                size_mb = info["size_mb"]
                if size_mb >= 1000:
                    size_str = f"{size_mb/1000:.2f} GB"
                else:
                    size_str = f"{size_mb:.1f} MB"
                
                gpu_str = info["gpu_name"] if info["gpu_id"] is not None else "CPU"
                
                # Formatação do tempo desde último uso
                time_since = info["time_since_last_use"]
                if time_since < 60:
                    time_str = f"{time_since:.1f} seg atrás"
                elif time_since < 3600:
                    time_str = f"{time_since/60:.1f} min atrás"
                else:
                    time_str = f"{time_since/3600:.1f} horas atrás"
                
                # Estilo baseado na recência de uso
                if time_since < 300:  # 5 minutos
                    style = "background-color:#e6ffe6;"  # Verde claro (uso recente)
                elif time_since < 3600:  # 1 hora
                    style = "background-color:#ffffe6;"  # Amarelo claro
                else:
                    style = ""  # Padrão
                
                html += f"<tr style='{style}'>"
                html += f"<td>{model_id}</td><td>{size_str}</td><td>{gpu_str}</td>"
                html += f"<td>{time_str}</td><td>{info['usage_count']}</td></tr>"
                
            html += "</table>"
        else:
            html += "<p>Nenhum modelo carregado.</p>"
        
        return html
    
    def get_recommendations_html(self) -> str:
        """
        Gera HTML com recomendações de otimização.
        
        Returns:
            String HTML formatada
        """
        html = "<h2>Recomendações de Otimização</h2>"
        
        if not self.has_gpu_manager:
            html += "<p>Gerenciador Multi-GPU não disponível. Recomendações não disponíveis.</p>"
            return html
        
        # Obter recomendações de balanceamento
        try:
            recommendations = self.gpu_manager.rebalance_models()
            
            if recommendations:
                html += "<h3>Recomendações de Rebalanceamento</h3>"
                html += "<ul>"
                
                for rec in recommendations:
                    action = rec["action"]
                    model_id = rec["model_id"]
                    source_gpu = rec["source_gpu"]
                    
                    if action == "unload":
                        html += f"<li>Descarregar <strong>{model_id}</strong> da GPU {source_gpu}</li>"
                    elif action == "move":
                        target_gpu = rec["target_gpu"]
                        html += f"<li>Mover <strong>{model_id}</strong> da GPU {source_gpu} para GPU {target_gpu}</li>"
                
                html += "</ul>"
                
                html += "<p><button id='apply-recommendations' onclick='applyRecommendations()'>Aplicar Recomendações</button></p>"
                html += "<script>function applyRecommendations() { alert('Recomendações aplicadas! (Simulação)'); }</script>"
            else:
                html += "<p>Nenhuma recomendação de rebalanceamento necessária no momento.</p>"
        
        except Exception as e:
            html += f"<p>Erro ao gerar recomendações: {e}</p>"
        
        # Adicionar recomendações específicas para RTX 2060 e GTX 1070
        if self.specific_gpus:
            has_rtx2060 = any(info["gpu_type"] == "rtx_2060" for info in self.specific_gpus.values())
            has_gtx1070 = any(info["gpu_type"] == "gtx_1070" for info in self.specific_gpus.values())
            
            if has_rtx2060 and has_gtx1070:
                html += "<h3>Recomendações para RTX 2060 + GTX 1070</h3>"
                html += "<ul>"
                html += "<li>Coloque modelos Phi-3 na <strong>RTX 2060</strong> para aproveitar os Tensor Cores</li>"
                html += "<li>Utilize a <strong>GTX 1070</strong> para modelos menores (Phi-2, Phi-1.5, GPT-2)</li>"
                html += "<li>Para modelos com otimização FP16, prefira a <strong>RTX 2060</strong></li>"
                html += "<li>Para modelos quantizados (GGUF/ONNX), ambas GPUs têm desempenho similar</li>"
                html += "</ul>"
                
                html += "<p><button id='optimize-phi3' onclick='optimizePhi3()'>Otimizar para Phi-3</button></p>"
                html += "<script>function optimizePhi3() { alert('Otimização para Phi-3 aplicada! (Simulação)'); }</script>"
            elif has_rtx2060:
                html += "<h3>Recomendações para RTX 2060</h3>"
                html += "<ul>"
                html += "<li>Priorize modelos Phi-3 e outros que se beneficiam de Tensor Cores</li>"
                html += "<li>Habilite otimizações FP16 quando possível</li>"
                html += "</ul>"
            elif has_gtx1070:
                html += "<h3>Recomendações para GTX 1070</h3>"
                html += "<ul>"
                html += "<li>Priorize modelos menores (Phi-2, Phi-1.5, GPT-2)</li>"
                html += "<li>Considere usar modelos quantizados (GGUF/ONNX) para melhor desempenho</li>"
                html += "</ul>"
        
        return html

def create_dashboard():
    """Cria e inicia a dashboard de monitoramento."""
    # Inicializar gerenciador
    manager = DashboardManager()
    
    # Criar interface com Gradio
    with gr.Blocks(title="EzioFilho_LLMGraph - Painel Multi-GPU") as dashboard:
        with gr.Row():
            gr.Markdown("# EzioFilho_LLMGraph - Sistema de Monitoramento Multi-GPU")
        
        with gr.Row():
            with gr.Column(scale=2):
                gpu_info = gr.HTML(manager.get_gpu_info_html())
                gpu_chart = gr.Plot(manager.get_gpu_usage_chart())
            
            with gr.Column(scale=1):
                model_info = gr.HTML(manager.get_model_info_html())
        
        with gr.Row():
            recommendations = gr.HTML(manager.get_recommendations_html())
        
        # Função de atualização
        def update_dashboard():
            return (
                manager.get_gpu_info_html(),
                manager.get_gpu_usage_chart(),
                manager.get_model_info_html(),
                manager.get_recommendations_html()
            )
        
        # Atualizar a cada 3 segundos
        dashboard.load(update_dashboard, inputs=None, outputs=[gpu_info, gpu_chart, model_info, recommendations], every=3)
    
    # Iniciar interface
    try:
        dashboard.launch(share=False)
    finally:
        manager.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EzioFilho_LLMGraph - Painel Multi-GPU")
    args = parser.parse_args()
    
    create_dashboard()
