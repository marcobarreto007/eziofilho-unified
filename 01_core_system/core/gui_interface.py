"""
GUI Interface - Interface gráfica para o sistema unificado EzioFilho
Implementa uma interface web usando Gradio
"""
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Importar componentes unificados
from core.unified_orchestrator import UnifiedOrchestrator
from model_auto_discovery import ModelAutoDiscovery
from core.gpu_monitor import get_gpu_monitor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("GradioInterface")

class GradioInterface:
    """
    Interface gráfica do sistema EzioFilho usando Gradio
    """
    
    # Versão da interface
    VERSION = "1.0.0"
    
    def __init__(
        self,
        orchestrator: Optional[UnifiedOrchestrator] = None,
        config_path: Optional[Union[str, Path]] = None,
        title: str = "EzioFilho Unified",
        description: str = "Sistema unificado de análise de texto",
        theme: str = "soft",
        port: int = 7860,
        auto_discover_models: bool = True,
        gpu_ids: Optional[List[int]] = None
    ):
        """
        Inicializa a interface Gradio
        
        Args:
            orchestrator: Instância do UnifiedOrchestrator (opcional)
            config_path: Caminho para arquivo de configuração
            title: Título da interface
            description: Descrição da interface
            theme: Tema da interface
            port: Porta para iniciar o servidor
            auto_discover_models: Se deve usar o sistema de descoberta automática de modelos
            gpu_ids: Lista de IDs das GPUs disponíveis
        """
        self.orchestrator = orchestrator
        self.config_path = config_path
        self.title = title
        self.description = description
        self.theme = theme
        self.port = port
        self.auto_discover_models = auto_discover_models
        self.gpu_ids = gpu_ids or []
        
        # Estado interno
        self.app = None
        self.initialized = False
        self.initialization_error = None
        self.model_discovery = None
        
        # Importar gradio
        try:
            import gradio as gr
            self.gr = gr
            logger.info(f"Gradio importado com sucesso")
        except ImportError as e:
            self.initialization_error = f"Erro ao importar Gradio: {str(e)}"
            logger.error(self.initialization_error)
            logger.error("Instale com: pip install gradio")
            self.gr = None
    
    def initialize(self) -> bool:
        """
        Inicializa a interface Gradio
        
        Returns:
            bool: True se inicialização bem-sucedida
        """
        if self.initialized:
            return True
        
        start_time = time.time()
        
        try:
            if self.gr is None:
                return False
            
            # Inicializar descoberta automática de modelos se habilitado
            if self.auto_discover_models:
                try:
                    logger.info("Inicializando sistema de descoberta automática de modelos...")
                    self.model_discovery = ModelAutoDiscovery()
                    self.model_discovery.initialize()
                    logger.info(f"Modelos detectados: {len(self.model_discovery.get_discovered_models())}")
                except Exception as e:
                    logger.error(f"Erro ao inicializar descoberta de modelos: {e}")
                    self.model_discovery = None
                
            # Inicializar orquestrador se não foi fornecido
            if self.orchestrator is None:
                try:
                    self.orchestrator = UnifiedOrchestrator(config_path=self.config_path)
                    self.orchestrator.initialize()
                except Exception as e:
                    logger.warning(f"Erro ao inicializar orquestrador: {e}")
                    # Criar orquestrador mesmo com erro
                    self.orchestrator = UnifiedOrchestrator(config_path=self.config_path, 
                                                           expert_types=[])
                
                # Integrar modelos descobertos ao orquestrador se disponível
                if self.model_discovery and hasattr(self.orchestrator, 'register_models'):
                    discovered_models = self.model_discovery.get_discovered_models()
                    if discovered_models:
                        logger.info(f"Registrando {len(discovered_models)} modelos no orquestrador")
                        self.orchestrator.register_models(discovered_models)
            
            self.initialized = True
            logger.info(f"GradioInterface inicializado em {time.time() - start_time:.2f}s")
            return True
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"Erro ao inicializar GradioInterface: {e}")
            return False
    
    def _analyze_text(self, text: str, expert_types: List[str]) -> Dict[str, Any]:
        """
        Função para analisar texto usando o orquestrador
        
        Args:
            text: Texto para análise
            expert_types: Lista de tipos de especialistas
            
        Returns:
            Dict com resultados da análise
        """
        try:
            if not self.orchestrator:
                return {"error": "Orquestrador não inicializado"}
                
            if not text.strip():
                return {"error": "Texto vazio"}
                
            start_time = time.time()
            results = self.orchestrator.analyze_text(text, expert_types=expert_types)
            elapsed = time.time() - start_time
            
            # Adicionar metadados
            results["text"] = text
            results["timestamp"] = time.time()
            results["elapsed_time"] = elapsed
            
            return results
        except Exception as e:
            logger.error(f"Erro ao analisar texto: {e}")
            return {"error": str(e), "text": text}
    
    def _format_results(self, results: Dict[str, Any]) -> str:
        """
        Formata resultados para exibição
        
        Args:
            results: Dicionário de resultados
        
        Returns:
            Texto formatado para exibição
        """
        if "error" in results:
            return f"Erro: {results['error']}"
            
        try:
            # Cabeçalho
            text = results.get("text", "")
            text_length = len(text)
            elapsed = results.get("elapsed_time", 0)
            
            output = f"### Resultados da Análise\n\n"
            output += f"Texto analisado: {text_length} caracteres\n"
            output += f"Tempo de análise: {elapsed:.2f} segundos\n\n"
            
            # Resultados por especialista
            for expert_name, expert_data in results.items():
                if expert_name in ["text", "timestamp", "elapsed_time"]:
                    continue
                    
                output += f"#### {expert_name.capitalize()}\n\n"
                
                # Verificar formato dos resultados
                if "result" in expert_data:
                    result_data = expert_data["result"]
                    
                    if isinstance(result_data, dict):
                        # Formatar dicionário de resultados
                        for key, value in result_data.items():
                            output += f"- **{key}**: {value}\n"
                    elif isinstance(result_data, str):
                        # Resultado em texto
                        output += f"{result_data}\n"
                    else:
                        # Outro formato
                        output += f"{result_data}\n"
                        
                # Adicionar métricas se disponíveis
                if "metrics" in expert_data:
                    metrics = expert_data["metrics"]
                    output += f"\n**Métricas**:\n"
                    for key, value in metrics.items():
                        if key == "elapsed_time":
                            output += f"- Tempo: {value:.2f}s\n"
                        else:
                            output += f"- {key}: {value}\n"
                            
                output += "\n"
                
            return output
            
        except Exception as e:
            logger.error(f"Erro ao formatar resultados: {e}")
            return f"Erro ao formatar resultados: {e}"
    
    def create_basic_interface(self):
        """
        Cria uma interface básica para análise de texto
        
        Returns:
            Interface Gradio
        """
        if not self.initialized:
            self.initialize()
            
        if not self.gr:
            logger.error("Gradio não está disponível")
            return None
            
        try:
            # Obter tipos de especialistas disponíveis
            available_experts = self.orchestrator.get_expert_types()
            
            with self.gr.Blocks(title=self.title, theme=self.theme) as app:
                # Cabeçalho
                self.gr.Markdown(f"# {self.title}")
                self.gr.Markdown(f"{self.description}")
                
                # Área de texto
                with self.gr.Row():
                    with self.gr.Column(scale=3):
                        text_input = self.gr.TextArea(
                            label="Texto para análise",
                            placeholder="Digite ou cole o texto que deseja analisar...",
                            lines=10
                        )
                        
                        # Seleção de especialistas
                        experts_checkboxes = self.gr.CheckboxGroup(
                            choices=available_experts,
                            label="Especialistas",
                            value=["sentiment", "factcheck"] if "sentiment" in available_experts else available_experts[:2],
                            interactive=True
                        )
                        
                        # Botão de análise
                        analyze_button = self.gr.Button("Analisar", variant="primary")
                        
                    with self.gr.Column(scale=4):
                        results_output = self.gr.Markdown(
                            label="Resultados",
                            value="Os resultados da análise aparecerão aqui."
                        )
                        
                        # JSON output para debug
                        with self.gr.Accordion("Dados JSON", open=False):
                            json_output = self.gr.JSON(label="Resultados brutos")
                
                # Callback de análise
                analyze_button.click(
                    fn=self._run_analysis,
                    inputs=[text_input, experts_checkboxes],
                    outputs=[results_output, json_output]
                )
                
                # Exemplos
                examples = [
                    ["O novo produto da empresa é excelente e superou todas as expectativas do mercado.", ["sentiment", "factcheck"]],
                    ["Estou muito insatisfeito com o péssimo atendimento recebido na loja.", ["sentiment"]],
                    ["O Brasil é o quinto maior país do mundo em território e população.", ["factcheck"]],
                ]
                
                self.gr.Examples(
                    examples=examples,
                    inputs=[text_input, experts_checkboxes],
                    outputs=[results_output, json_output],
                    fn=self._run_analysis
                )
                
            self.app = app
            return app
            
        except Exception as e:
            logger.error(f"Erro ao criar interface: {e}")
            return None
    
    def create_advanced_interface(self):
        """
        Cria uma interface avançada para análise de texto
        com múltiplas funcionalidades
        
        Returns:
            Interface Gradio
        """
        if not self.initialized:
            self.initialize()
            
        if not self.gr:
            logger.error("Gradio não está disponível")
            return None
            
        try:
            # Obter tipos de especialistas disponíveis
            available_experts = self.orchestrator.get_expert_types()
            
            with self.gr.Blocks(title=self.title, theme=self.theme) as app:
                # Cabeçalho
                self.gr.Markdown(f"# {self.title}")
                self.gr.Markdown(f"{self.description}")
                
                # Abas para diferentes funcionalidades
                with self.gr.Tabs() as tabs:
                    # Aba de Análise de Texto
                    with self.gr.TabItem("Análise de Texto"):
                        with self.gr.Row():
                            with self.gr.Column(scale=3):
                                text_input = self.gr.TextArea(
                                    label="Texto para análise",
                                    placeholder="Digite ou cole o texto que deseja analisar...",
                                    lines=10
                                )
                                
                                # Seleção de especialistas
                                experts_checkboxes = self.gr.CheckboxGroup(
                                    choices=available_experts,
                                    label="Especialistas",
                                    value=["sentiment", "factcheck"] if "sentiment" in available_experts else available_experts[:2],
                                    interactive=True
                                )
                                
                                # Botão de análise
                                analyze_button = self.gr.Button("Analisar", variant="primary")
                                
                            with self.gr.Column(scale=4):
                                results_output = self.gr.Markdown(
                                    label="Resultados",
                                    value="Os resultados da análise aparecerão aqui."
                                )
                                
                                # JSON output para debug
                                with self.gr.Accordion("Dados JSON", open=False):
                                    json_output = self.gr.JSON(label="Resultados brutos")
                    
                    # Aba de Chat assistido por especialistas
                    with self.gr.TabItem("Chat com Especialistas"):
                        chatbot = self.gr.Chatbot(label="Conversa")
                        
                        with self.gr.Row():
                            chat_input = self.gr.Textbox(
                                label="Mensagem",
                                placeholder="Digite sua mensagem...",
                                lines=2
                            )
                            chat_button = self.gr.Button("Enviar", variant="primary")
                            
                        with self.gr.Row():
                            chat_experts_checkboxes = self.gr.CheckboxGroup(
                                choices=available_experts,
                                label="Especialistas para assistir o chat",
                                value=["sentiment", "qa"] if "qa" in available_experts else [available_experts[0]],
                                interactive=True
                            )
                            
                        with self.gr.Accordion("Configurações do Chat", open=False):
                            chat_memory = self.gr.Checkbox(
                                label="Manter histórico",
                                value=True
                            )
                            
                            analysis_detail = self.gr.Radio(
                                choices=["Resumido", "Detalhado", "Nenhum"],
                                label="Nível de detalhes da análise",
                                value="Resumido"
                            )
                    
                    # Aba de Status do Sistema
                    with self.gr.TabItem("Status do Sistema"):
                        status_button = self.gr.Button("Atualizar Status")
                        
                        with self.gr.Row():
                            with self.gr.Column():
                                system_status = self.gr.Markdown("Carregando status do sistema...")
                                
                            with self.gr.Column():
                                expert_status = self.gr.DataFrame(
                                    headers=["Especialista", "Status", "Memória", "Tokens/s"],
                                    label="Status dos Especialistas"
                                )
                                
                    # Aba de Gestão de Modelos
                    with self.gr.TabItem("Gestão de Modelos"):
                        with self.gr.Row():
                            model_discovery_button = self.gr.Button("Detectar Modelos", variant="primary")
                            
                        with self.gr.Row():
                            with self.gr.Column(scale=3):
                                model_status = self.gr.DataFrame(
                                    headers=["Modelo", "Tipo", "Caminho", "Expert Recomendado", "Status"],
                                    label="Modelos Detectados"
                                )
                            
                            with self.gr.Column(scale=2):
                                model_details = self.gr.JSON(
                                    label="Detalhes do Modelo",
                                    value={}
                                )
                                
                        with self.gr.Accordion("Opções de Descoberta", open=False):
                            with self.gr.Row():
                                with self.gr.Column():
                                    search_paths = self.gr.Textbox(
                                        label="Caminhos de Busca (separados por vírgula)",
                                        placeholder="/caminho/para/modelos, /outro/caminho",
                                        lines=1
                                    )
                                    
                                with self.gr.Column():
                                    model_patterns = self.gr.Textbox(
                                        label="Padrões de Busca (separados por vírgula)",
                                        placeholder="*.gguf, *.onnx, *.bin",
                                        value="*.gguf, *.onnx, *.bin, *.safetensors",
                                        lines=1
                                    )
                                    
                            with self.gr.Row():
                                recursive_search = self.gr.Checkbox(
                                    label="Busca recursiva",
                                    value=True
                                )
                                
                                auto_config = self.gr.Checkbox(
                                    label="Configuração automática",
                                    value=True
                                )
                
                # Callbacks
                analyze_button.click(
                    fn=self._run_analysis,
                    inputs=[text_input, experts_checkboxes],
                    outputs=[results_output, json_output]
                )
                
                chat_history = []
                
                def chat(message, history, experts, memory, detail_level):
                    history.append([message, ""])
                    
                    try:
                        # Analisar a mensagem do usuário usando especialistas selecionados
                        analysis_results = {}
                        if detail_level != "Nenhum":
                            analysis_results = self._analyze_text(message, experts)
                        
                        # Preparar resposta do sistema
                        response = ""
                        
                        # Se o especialista QA estiver selecionado, tenta responder diretamente
                        if "qa" in experts and self.orchestrator.has_expert("qa"):
                            qa_response = self.orchestrator.ask_question(message)
                            response = qa_response.get("answer", "")
                        
                        # Adicionar análise ao final se solicitado
                        if detail_level == "Resumido":
                            # Adicionar apenas um resumo das análises
                            if "sentiment" in experts and "sentiment" in analysis_results:
                                sentiment = analysis_results["sentiment"].get("result", {}).get("sentiment", "neutro")
                                response += f"\n\n*Análise de sentimento: {sentiment}*"
                                
                            if "factcheck" in experts and "factcheck" in analysis_results:
                                factuality = analysis_results["factcheck"].get("result", {}).get("factuality", "desconhecido")
                                response += f"\n*Verificação de fatos: {factuality}*"
                                
                        elif detail_level == "Detalhado":
                            # Adicionar análise detalhada
                            if analysis_results and "error" not in analysis_results:
                                response += "\n\n**Análise Detalhada**:\n"
                                response += self._format_results(analysis_results)
                        
                        # Se não tiver resposta nem análise, usar mensagem padrão
                        if not response:
                            response = "Não foi possível processar sua mensagem com os especialistas selecionados."
                        
                        history[-1][1] = response
                        
                        # Manter histórico se solicitado
                        if not memory:
                            history = [history[-1]]
                            
                        return "", history
                        
                    except Exception as e:
                        logger.error(f"Erro no chat: {e}")
                        history[-1][1] = f"Erro ao processar mensagem: {str(e)}"
                        return "", history
                
                chat_button.click(
                    fn=chat,
                    inputs=[chat_input, chatbot, chat_experts_checkboxes, chat_memory, analysis_detail],
                    outputs=[chat_input, chatbot]
                )
                
                def get_system_status():
                    if not self.orchestrator:
                        return "Orquestrador não inicializado", [], "Monitor de GPU não disponível"
                    
                    # Preparar informações de status
                    system_info = f"### Status do Sistema EzioFilho Unified\n\n"
                    system_info += f"**Versão:** {self.orchestrator.VERSION}\n"
                    system_info += f"**Especialistas Ativos:** {len(self.orchestrator.get_expert_types())}\n"
                    
                    # Status de GPU com GPUMonitor
                    gpu_monitor = get_gpu_monitor()
                    gpu_status_md = "### Status das GPUs\\n\\n"
                    if gpu_monitor and gpu_monitor.is_running:
                        gpu_metrics = gpu_monitor.get_gpu_info_summary()
                        if gpu_metrics:
                            for gpu_data in gpu_metrics:
                                gpu_status_md += f"**GPU {gpu_data['id']}: {gpu_data['name']}**\\n"
                                gpu_status_md += f"  - Memória Total: {gpu_data['total_memory_mb']:.2f} MB\\n"
                                if 'mem_utilization' in gpu_data:
                                    gpu_status_md += f"  - Utilização: {gpu_data['mem_utilization']:.2f}%\\n"
                                    gpu_status_md += f"  - Memória Livre: {gpu_data['memory_free_mb']:.2f} MB\\n"
                                else:
                                    gpu_status_md += f"  - Utilização: N/A\\n"
                                gpu_status_md += f"  - Capacidade de Cômputo: {gpu_data['compute_capability']}\\n"
                                gpu_status_md += f"  - Suporta Tensor Cores: {'Sim' if gpu_data['supports_tensor_cores'] else 'Não'}\\n\\n"
                        else:
                            gpu_status_md += "Nenhuma GPU monitorada ou dados indisponíveis.\\n"
                    else:
                        gpu_status_md += "Monitor de GPU não está ativo ou não foi inicializado corretamente.\\n"
                        # Fallback para torch.cuda se o monitor não estiver ativo
                        try:
                            import torch
                            if torch.cuda.is_available():
                                num_gpus = torch.cuda.device_count()
                                if num_gpus > 0:
                                    gpu_status_md += "\\n**Informações básicas (via PyTorch):**\\n"
                                    for i in range(num_gpus):
                                        device_name = torch.cuda.get_device_name(i)
                                        memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**2) # MB
                                        memory_allocated = torch.cuda.memory_allocated(i) / (1024**2) # MB
                                        gpu_status_md += f"- GPU {i}: {device_name}, Total: {memory_total:.2f}MB, Alocada: {memory_allocated:.2f}MB\\n"
                                else:
                                    gpu_status_md += "- Nenhuma GPU CUDA detectada via PyTorch.\\n"
                            else:
                                gpu_status_md += "- CUDA não disponível (via PyTorch).\\n"
                        except Exception as e:
                            gpu_status_md += f"- Erro ao obter informações básicas de GPU via PyTorch: {str(e)}\\n"

                    system_info += gpu_status_md
                    
                    # Status dos especialistas
                    status_data = []
                    for expert_type, expert in self.orchestrator.experts.items():
                        status = "Ativo" if expert.is_initialized else "Erro"
                        memory = "-"
                        tokens = "-"
                        
                        if hasattr(expert, "metrics") and expert.metrics:
                            memory_usage = expert.metrics.get("memory_usage", 0)
                            if memory_usage > 0:
                                memory = f"{memory_usage:.2f}MB"
                            
                            tokens_per_sec = expert.metrics.get("tokens_per_second", 0)
                            if tokens_per_sec > 0:
                                tokens = f"{tokens_per_sec:.1f}"
                        
                        status_data.append([expert_type, status, memory, tokens])
                    
                    return system_info, status_data
                
                status_button.click(
                    fn=get_system_status,
                    inputs=[],
                    outputs=[system_status, expert_status] # Removido gpu_status_output, pois foi concatenado em system_status
                )
                
                # Funções de callback para gestão de modelos
                def discover_models(search_paths, model_patterns, recursive, auto_config):
                    if not self.model_discovery:
                        try:
                            self.model_discovery = ModelAutoDiscovery()
                            self.model_discovery.initialize()
                        except Exception as e:
                            return [], {"error": str(e)}
                    
                    # Configurar parâmetros de busca
                    if search_paths and search_paths.strip():
                        paths = [p.strip() for p in search_paths.split(",")]
                        self.model_discovery.set_search_paths(paths)
                    
                    if model_patterns and model_patterns.strip():
                        patterns = [p.strip() for p in model_patterns.split(",")]
                        self.model_discovery.set_model_patterns(patterns)
                    
                    # Executar descoberta
                    self.model_discovery.set_recursive_search(recursive)
                    self.model_discovery.discover_models(force=True)
                    
                    if auto_config:
                        self.model_discovery.configure_models()
                    
                    # Obter resultados
                    models = self.model_discovery.get_discovered_models()
                    model_data = []
                    
                    for model_id, model_info in models.items():
                        status = "Configurado" if model_info.get("configured", False) else "Detectado"
                        model_data.append([
                            model_info.get("name", model_id),
                            model_info.get("type", "Desconhecido"),
                            model_info.get("path", ""),
                            model_info.get("expert_role", ""),
                            status
                        ])
                    
                    # Atualizar orquestrador se disponível
                    if self.orchestrator and hasattr(self.orchestrator, 'register_models') and auto_config:
                        self.orchestrator.register_models(models)
                    
                    return model_data, {}
                
                def select_model(evt: gr.SelectData, models_data):
                    if not self.model_discovery:
                        return {}
                    
                    try:
                        # Obter o nome do modelo selecionado
                        row_idx = evt.index[0]
                        model_name = models_data[row_idx][0]
                        
                        # Encontrar o modelo nos modelos descobertos
                        models = self.model_discovery.get_discovered_models()
                        selected_model = None
                        
                        for model_id, model_info in models.items():
                            if model_info.get("name") == model_name:
                                selected_model = model_info
                                break
                        
                        if selected_model:
                            return selected_model
                        return {"error": "Modelo não encontrado"}
                        
                    except Exception as e:
                        return {"error": str(e)}
                
                # Conectar callbacks para gestão de modelos
                model_discovery_button.click(
                    fn=discover_models,
                    inputs=[search_paths, model_patterns, recursive_search, auto_config],
                    outputs=[model_status, model_details]
                )
                
                model_status.select(
                    fn=select_model,
                    inputs=[model_status],
                    outputs=[model_details]
                )
                
                # Exemplos
                examples = [
                    ["O novo produto da empresa é excelente e superou todas as expectativas do mercado.", ["sentiment", "factcheck"]],
                    ["Estou muito insatisfeito com o péssimo atendimento recebido na loja.", ["sentiment"]],
                    ["O Brasil é o quinto maior país do mundo em território e população.", ["factcheck"]],
                ]
                
                self.gr.Examples(
                    examples=examples,
                    inputs=[text_input, experts_checkboxes],
                    outputs=[results_output, json_output],
                    fn=self._run_analysis
                )
                
            self.app = app
            return app
            
        except Exception as e:
            logger.error(f"Erro ao criar interface avançada: {e}")
            return None
    
    def _run_analysis(self, text, expert_types):
        """
        Executa a análise e retorna resultados formatados
        
        Args:
            text: Texto para análise
            expert_types: Lista de tipos de especialistas
            
        Returns:
            (str, dict): Resultados formatados e dados JSON
        """
        results = self._analyze_text(text, expert_types)
        formatted_results = self._format_results(results)
        return formatted_results, results
    
    def launch(
        self, 
        interface_type: str = "advanced",
        share: bool = False,
        debug: bool = False
    ):
        """
        Inicia o servidor Gradio
        
        Args:
            interface_type: Tipo de interface ('basic' ou 'advanced')
            share: Se deve compartilhar publicamente
            debug: Se deve mostrar debug
            
        Returns:
            Instância do servidor Gradio
        """
        if not self.initialized:
            self.initialize()
            
        if not self.gr:
            logger.error("Gradio não está disponível")
            return None
        
        try:
            # Criar interface se necessário
            if self.app is None:
                if interface_type.lower() == "basic":
                    self.create_basic_interface()
                else:
                    self.create_advanced_interface()
            
            if not self.app:
                logger.error("Falha ao criar interface")
                return None
            
            # Iniciar servidor
            logger.info(f"Iniciando servidor Gradio na porta {self.port}...")
            return self.app.launch(
                server_name="0.0.0.0",  # Acessível externamente
                server_port=self.port,
                share=share,
                debug=debug
            )
            
        except Exception as e:
            logger.error(f"Erro ao iniciar servidor: {e}")
            return None


# Função auxiliar para criar e iniciar interface rapidamente
def create_ezio_gui(
    config_path: Optional[Union[str, Path]] = None,
    port: int = 7860,
    interface_type: str = "advanced",
    share: bool = False,
    debug: bool = False,
    title: str = "EzioFilho Unified",
    auto_discover_models: bool = True,
    gpu_ids: Optional[List[int]] = None
) -> GradioInterface:
    """
    Cria e inicia uma interface Gradio para o sistema EzioFilho
    
    Args:
        config_path: Caminho para arquivo de configuração
        port: Porta para o servidor
        interface_type: Tipo de interface ('basic' ou 'advanced')
        share: Se deve compartilhar publicamente
        debug: Se deve mostrar debug
        title: Título da interface
        auto_discover_models: Se deve usar descoberta automática de modelos
        gpu_ids: Lista de IDs das GPUs disponíveis
        
    Returns:
        Instância do GradioInterface ou None se falhar
    """
    try:
        # Inicializar orquestrador
        from core.unified_orchestrator import UnifiedOrchestrator
        orchestrator = UnifiedOrchestrator(
            config_path=config_path,
            gpu_id=gpu_ids[0] if gpu_ids and len(gpu_ids) > 0 else None
        )
        if not orchestrator.initialize():
            logger.error("Falha ao inicializar orquestrador")
            return None
            
        # Criar interface
        interface = GradioInterface(
            orchestrator=orchestrator,
            config_path=config_path,
            title=title,
            port=port,
            auto_discover_models=auto_discover_models,
            gpu_ids=gpu_ids
        )
        
        if not interface.initialize():
            logger.error("Falha ao inicializar interface")
            return None
            
        # Iniciar servidor
        interface.launch(interface_type=interface_type, share=share, debug=debug)
        return interface
        
    except Exception as e:
        logger.error(f"Erro ao criar GUI: {e}")
        return None
