#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU Monitor Dashboard - Monitor de GPUs em tempo real para EzioFilho_LLMGraph
----------------------------------------------------------------------------
Esta ferramenta oferece:
- Visualização em tempo real do uso de memória das GPUs
- Gráficos históricos de utilização
- Detecção de possíveis problemas de memória
- Interface interativa para monitoramento

Autor: EzioFilho LLMGraph Team
Data: Maio/2025
"""

import os
import sys
import time
import logging
import gradio as gr
import numpy as np
import pandas as pd
import threading
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Adicionar diretório pai ao path para importar módulos do projeto
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Importar módulos do projeto
from core.gpu_monitor import GPUMonitor, get_gpu_monitor

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("GPUMonitorDashboard")

class GPUMonitorDashboard:
    """
    Dashboard para monitoramento de GPUs em tempo real.
    """
    
    def __init__(self, gpu_ids: Optional[List[int]] = None, poll_interval: float = 1.0):
        """
        Inicializa o dashboard.
        
        Args:
            gpu_ids: Lista de IDs das GPUs para monitorar (None para todas)
            poll_interval: Intervalo de atualização em segundos
        """
        # Inicializar monitor de GPU
        self.gpu_monitor = get_gpu_monitor(gpu_ids=gpu_ids, poll_interval=poll_interval)
        self.gpu_ids = self.gpu_monitor.gpu_ids if self.gpu_monitor else []
        
        # Verificar se há GPUs disponíveis
        if not self.gpu_ids:
            logger.error("Nenhuma GPU detectada para monitoramento.")
            sys.exit(1)
            
        logger.info(f"GPUs disponíveis para monitoramento: {self.gpu_ids}")
        
        # Histórico de métricas para gráficos
        self.max_history = 60  # Mostrar últimos 60 pontos nos gráficos
        self.metrics_history = {gpu_id: {
            "timestamps": [],
            "mem_utilization": [],
            "mem_allocated_mb": [],
            "memory_free_mb": []
        } for gpu_id in self.gpu_ids}
        
        # Thread para atualização periódica das métricas
        self._stop_event = threading.Event()
        self._update_thread = None
        
        # Interface Gradio
        self.interface = None
        
    def _update_metrics_loop(self):
        """Loop para atualização periódica das métricas."""
        while not self._stop_event.is_set():
            try:
                self._update_metrics()
                time.sleep(self.gpu_monitor.poll_interval)
            except Exception as e:
                logger.error(f"Erro ao atualizar métricas: {e}")
                time.sleep(1)
    
    def _update_metrics(self):
        """Atualiza métricas para todas as GPUs."""
        current_time = time.time()
        current_metrics = self.gpu_monitor.get_current_metrics()
        
        for gpu_id in self.gpu_ids:
            if gpu_id in current_metrics:
                metrics = current_metrics[gpu_id]
                
                # Adicionar timestamp
                self.metrics_history[gpu_id]["timestamps"].append(current_time)
                
                # Adicionar métricas
                self.metrics_history[gpu_id]["mem_utilization"].append(metrics.get("mem_utilization", 0))
                self.metrics_history[gpu_id]["mem_allocated_mb"].append(metrics.get("mem_allocated_mb", 0))
                self.metrics_history[gpu_id]["memory_free_mb"].append(metrics.get("memory_free_mb", 0))
                
                # Limitar tamanho do histórico
                if len(self.metrics_history[gpu_id]["timestamps"]) > self.max_history:
                    for key in self.metrics_history[gpu_id]:
                        self.metrics_history[gpu_id][key] = self.metrics_history[gpu_id][key][-self.max_history:]
    
    def _generate_metrics_html(self) -> str:
        """
        Gera HTML com métricas atuais das GPUs.
        
        Returns:
            HTML formatado com métricas
        """
        current_metrics = self.gpu_monitor.get_current_metrics()
        
        html = """
        <style>
            .gpu-card {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                background-color: #f8f9fa;
            }
            .gpu-header {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #2c3e50;
            }
            .gpu-metric {
                margin: 5px 0;
            }
            .gpu-metric-label {
                font-weight: bold;
                display: inline-block;
                width: 180px;
            }
            .gpu-metric-value {
                display: inline-block;
            }
            .memory-bar {
                height: 20px;
                border-radius: 3px;
                background-color: #ecf0f1;
                margin-top: 10px;
                position: relative;
                overflow: hidden;
            }
            .memory-bar-fill {
                height: 100%;
                background-color: #3498db;
                position: absolute;
                top: 0;
                left: 0;
                transition: width 0.5s ease;
            }
            .memory-bar-text {
                position: absolute;
                width: 100%;
                text-align: center;
                top: 0;
                font-size: 12px;
                line-height: 20px;
                color: #333;
                font-weight: bold;
                text-shadow: 0 0 2px #fff;
            }
            .warning {
                color: #e74c3c;
                font-weight: bold;
            }
        </style>
        """
        
        # Adicionar cards para cada GPU
        for gpu_id in self.gpu_ids:
            if gpu_id not in current_metrics:
                html += f"""
                <div class="gpu-card">
                    <div class="gpu-header">GPU {gpu_id}: <span class="warning">INDISPONÍVEL</span></div>
                    <div class="gpu-metric">
                        <span class="warning">Esta GPU não está respondendo ao monitoramento</span>
                    </div>
                </div>
                """
                continue
                
            metrics = current_metrics[gpu_id]
            
            # Definir classe de alerta
            warning_class = ""
            if metrics.get("mem_utilization", 0) > 85:
                warning_class = "warning"
            
            name = metrics.get("name", f"GPU {gpu_id}")
            total_memory = metrics.get("total_memory_mb", 0)
            used_memory = metrics.get("mem_allocated_mb", 0)
            free_memory = metrics.get("memory_free_mb", 0)
            utilization = metrics.get("mem_utilization", 0)
            
            # Calcular largura da barra de memória
            bar_width = min(100, utilization)
            
            html += f"""
            <div class="gpu-card">
                <div class="gpu-header">GPU {gpu_id}: {name}</div>
                <div class="gpu-metric">
                    <span class="gpu-metric-label">Memória Total:</span>
                    <span class="gpu-metric-value">{total_memory:.0f} MB</span>
                </div>
                <div class="gpu-metric">
                    <span class="gpu-metric-label">Memória Utilizada:</span>
                    <span class="gpu-metric-value {warning_class}">{used_memory:.0f} MB ({utilization:.1f}%)</span>
                </div>
                <div class="gpu-metric">
                    <span class="gpu-metric-label">Memória Livre:</span>
                    <span class="gpu-metric-value">{free_memory:.0f} MB</span>
                </div>
                
                <div class="memory-bar">
                    <div class="memory-bar-fill" style="width: {bar_width}%;"></div>
                    <div class="memory-bar-text">{utilization:.1f}% Utilização</div>
                </div>
            </div>
            """
        
        # Adicionar recomendações se houver GPUs sobrecarregadas
        any_overloaded = any(
            current_metrics.get(gpu_id, {}).get("mem_utilization", 0) > 85 
            for gpu_id in self.gpu_ids
        )
        
        if any_overloaded:
            html += """
            <div style="border-left: 4px solid #e74c3c; padding: 10px; margin-top: 20px; background-color: #ffebee;">
                <h3 style="color: #e74c3c; margin-top: 0;">⚠️ Alerta de Sobrecarga de GPU</h3>
                <p>Pelo menos uma GPU está com alto uso de memória. Recomendações:</p>
                <ul>
                    <li>Considere descarregar modelos não utilizados</li>
                    <li>Verifique por vazamentos de memória em operações sequenciais</li>
                    <li>Mova modelos para GPUs menos utilizadas ou para CPU temporariamente</li>
                </ul>
            </div>
            """
        
        return html
    
    def _generate_history_plot(self, gpu_id: int) -> Tuple[plt.Figure, plt.Axes]:
        """
        Gera gráfico com histórico de utilização para uma GPU.
        
        Args:
            gpu_id: ID da GPU
            
        Returns:
            Figura do Matplotlib
        """
        if gpu_id not in self.metrics_history:
            # GPU não encontrada, retornar figura vazia
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"GPU {gpu_id} não encontrada", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig
            
        history = self.metrics_history[gpu_id]
        
        # Converter timestamps para formato legível
        if history["timestamps"]:
            time_labels = [time.strftime("%H:%M:%S", time.localtime(ts)) for ts in history["timestamps"]]
            
            # Usar um subconjunto dos labels para evitar sobreposição
            n_labels = len(time_labels)
            step = max(1, n_labels // 10)
            
            # Criar figura e eixos
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plotar utilização de memória
            ax.plot(history["mem_utilization"], 'r-', label='Utilização (%)', linewidth=2)
            ax.set_ylabel('Utilização de Memória (%)', color='r')
            ax.tick_params(axis='y', labelcolor='r')
            ax.set_ylim(0, 105)  # Limitar eixo Y a pouco mais de 100%
            
            # Adicionar eixo Y secundário para memória
            ax2 = ax.twinx()
            ax2.plot(history["mem_allocated_mb"], 'b-', label='Uso (MB)', linewidth=2)
            ax2.plot(history["memory_free_mb"], 'g-', label='Livre (MB)', linewidth=2)
            ax2.set_ylabel('Memória (MB)', color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            
            # Configurar eixo X com labels de tempo
            plt.xticks(np.arange(0, n_labels, step), [time_labels[i] for i in range(0, n_labels, step)], rotation=45)
            
            # Adicionar legenda
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Adicionar título
            name = self.gpu_monitor.get_current_metrics().get(gpu_id, {}).get("name", f"GPU {gpu_id}")
            plt.title(f'Histórico de Utilização - {name} (ID: {gpu_id})')
            
            plt.tight_layout()
            
            return fig
        else:
            # Sem dados históricos, retornar figura vazia
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Aguardando dados históricos...", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig
    
    def update_dashboard(self) -> Tuple[str, List[plt.Figure]]:
        """
        Atualiza o dashboard com métricas atuais.
        
        Returns:
            Tuple contendo (HTML com métricas, Lista de gráficos para cada GPU)
        """
        # Gerar HTML com métricas
        metrics_html = self._generate_metrics_html()
        
        # Gerar gráficos para cada GPU
        plots = [self._generate_history_plot(gpu_id) for gpu_id in self.gpu_ids]
        
        return metrics_html, plots
    
    def start_monitoring(self):
        """Inicia o monitoramento em thread separada."""
        if self._update_thread is None or not self._update_thread.is_alive():
            self._stop_event.clear()
            self._update_thread = threading.Thread(
                target=self._update_metrics_loop, 
                name="GPUMonitorThread", 
                daemon=True
            )
            self._update_thread.start()
            logger.info("Thread de monitoramento iniciada")
    
    def stop_monitoring(self):
        """Para o monitoramento."""
        self._stop_event.set()
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=3.0)
        logger.info("Thread de monitoramento parada")
    
    def launch_dashboard(self, share: bool = False):
        """
        Lança o dashboard usando Gradio.
        
        Args:
            share: Se deve compartilhar publicamente o dashboard
        """
        self.start_monitoring()
        
        with gr.Blocks(title="EzioFilho GPU Monitor", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# EzioFilho GPU Monitor Dashboard")
            
            with gr.Row():
                with gr.Column():
                    # Informações do sistema
                    system_info = gr.HTML()
                    
                    with gr.Row():
                        auto_refresh = gr.Checkbox(label="Atualização Automática", value=True)
                        refresh_interval = gr.Slider(
                            label="Intervalo de Atualização (segundos)", 
                            minimum=1, maximum=10, value=2, step=1
                        )
                    
                    refresh_button = gr.Button("Atualizar Agora")
            
            # Gráficos para cada GPU
            with gr.Row():
                plot_outputs = [gr.Plot() for _ in self.gpu_ids]
            
            # Função para atualizar o dashboard
            def update_dashboard_wrapper():
                html, figs = self.update_dashboard()
                return [html] + figs
            
            # Atualizar quando solicitado
            refresh_button.click(
                fn=update_dashboard_wrapper,
                outputs=[system_info] + plot_outputs
            )
            
            # Atualização automática
            auto_interval = gr.Number(value=2, visible=False)
            
            def set_interval(auto, interval):
                return interval if auto else 0
            
            # Atualizar intervalo quando checkbox ou slider mudar
            refresh_deps = [auto_refresh, refresh_interval]
            for dep in refresh_deps:
                dep.change(
                    fn=set_interval,
                    inputs=[auto_refresh, refresh_interval],
                    outputs=auto_interval
                )
            
            # Configurar atualização automática com intervalo variável
            gr.Markdown("Sistema iniciando. Por favor aguarde...")
            
            # Chamada inicial para preencher o dashboard na inicialização
            interface.load(
                fn=update_dashboard_wrapper,
                outputs=[system_info] + plot_outputs
            )
            
            # Atualização periódica
            interface.load(
                fn=update_dashboard_wrapper,
                inputs=None,
                outputs=[system_info] + plot_outputs,
                every=auto_interval
            )
        
        # Lançar interface
        self.interface = interface
        interface.queue().launch(share=share)
    
    def cleanup(self):
        """Limpa recursos ao finalizar."""
        self.stop_monitoring()
        logger.info("Dashboard finalizado")

def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dashboard de Monitoramento de GPUs para EzioFilho_LLMGraph")
    parser.add_argument("--gpu-ids", help="IDs das GPUs para monitorar (separados por vírgula, default: todas)")
    parser.add_argument("--interval", type=float, default=1.0, help="Intervalo de atualização em segundos")
    parser.add_argument("--share", action="store_true", help="Compartilhar dashboard publicamente")
    
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
    
    # Inicializar e lançar dashboard
    try:
        dashboard = GPUMonitorDashboard(gpu_ids=gpu_ids, poll_interval=args.interval)
        dashboard.launch_dashboard(share=args.share)
    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuário")
    finally:
        if 'dashboard' in locals():
            dashboard.cleanup()

if __name__ == "__main__":
    main()
