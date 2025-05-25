"""
LangGraph Integration - Integração do LangGraph com o sistema unificado EzioFilho
Implementa grafos direcionados para pipelines de processamento de texto
"""
import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable

# Importar componentes unificados
from core.unified_base_expert import EzioBaseExpert
from core.unified_orchestrator import UnifiedOrchestrator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("LangGraphIntegration")

class LangGraphIntegration:
    """
    Integração do sistema EzioFilho com LangGraph
    """
    
    # Versão da integração
    VERSION = "1.0.0"
    
    def __init__(
        self,
        orchestrator: Optional[UnifiedOrchestrator] = None,
        config_path: Optional[Union[str, Path]] = None,
        langgraph_config: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa a integração LangGraph
        
        Args:
            orchestrator: Instância do UnifiedOrchestrator (opcional)
            config_path: Caminho para arquivo de configuração
            langgraph_config: Configuração personalizada para LangGraph
        """
        self.orchestrator = orchestrator
        self.config_path = config_path
        self.langgraph_config = langgraph_config or {}
        
        # Estado interno
        self.graphs = {}
        self.initialized = False
        self.initialization_error = None
        
        # Importar langgraph
        try:
            import langgraph
            self.langgraph = langgraph
            logger.info(f"LangGraph {langgraph.__version__} importado com sucesso")
            
            from langgraph.graph import StateGraph, END
            self.StateGraph = StateGraph
            self.END = END
        except ImportError as e:
            self.initialization_error = f"Erro ao importar LangGraph: {str(e)}"
            logger.error(self.initialization_error)
            logger.error("Instale com: pip install langgraph")
            self.langgraph = None
            self.StateGraph = None
            self.END = None
    
    def initialize(self) -> bool:
        """
        Inicializa a integração LangGraph
        
        Returns:
            bool: True se inicialização bem-sucedida
        """
        if self.initialized:
            return True
        
        start_time = time.time()
        
        try:
            if self.langgraph is None:
                return False
                
            # Inicializar orquestrador se não foi fornecido
            if self.orchestrator is None:
                self.orchestrator = UnifiedOrchestrator(config_path=self.config_path)
                self.orchestrator.initialize()
            
            # Carregar configuração LangGraph
            if self.config_path:
                self._load_langgraph_config()
            
            self.initialized = True
            logger.info(f"LangGraphIntegration inicializado em {time.time() - start_time:.2f}s")
            return True
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"Erro ao inicializar LangGraphIntegration: {e}")
            return False
    
    def _load_langgraph_config(self) -> None:
        """Carrega configuração do LangGraph de arquivo"""
        try:
            if not self.config_path:
                return
                
            config_path = Path(self.config_path)
            if not config_path.exists():
                logger.warning(f"Arquivo de configuração não encontrado: {config_path}")
                return
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if "langgraph" in config:
                self.langgraph_config = config["langgraph"]
                logger.info(f"Configuração LangGraph carregada: {len(self.langgraph_config)} entradas")
        except Exception as e:
            logger.error(f"Erro ao carregar configuração LangGraph: {e}")
    
    def create_sequential_graph(
        self,
        name: str,
        nodes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Cria um grafo sequencial simples
        
        Args:
            name: Nome do grafo
            nodes: Lista de dicionários de nós com funções e metadados
        
        Returns:
            Dict com informações do grafo criado
        """
        if not self.initialized:
            self.initialize()
        
        if not self.StateGraph:
            logger.error("LangGraph não está inicializado corretamente")
            return {"error": "LangGraph não inicializado"}
        
        try:
            # Criar grafo de estado
            builder = self.StateGraph(nodes=[node["name"] for node in nodes])
            
            # Adicionar nós
            for i, node in enumerate(nodes):
                if i < len(nodes) - 1:
                    # Conectar ao próximo nó
                    builder.add_edge(node["name"], nodes[i+1]["name"])
                else:
                    # Último nó se conecta ao END
                    builder.add_edge(node["name"], self.END)
            
            # Construir e compilar grafo
            graph = builder.compile()
            
            # Criar funções de nós
            node_functions = {}
            for node in nodes:
                if "function" in node:
                    node_functions[node["name"]] = node["function"]
                    
            # Armazenar informações
            graph_info = {
                "name": name,
                "graph": graph,
                "nodes": nodes,
                "node_functions": node_functions,
                "type": "sequential",
                "created_at": time.time()
            }
            
            self.graphs[name] = graph_info
            logger.info(f"Grafo sequencial '{name}' criado com {len(nodes)} nós")
            
            return graph_info
            
        except Exception as e:
            error = f"Erro ao criar grafo sequencial '{name}': {str(e)}"
            logger.error(error)
            return {"error": error}
    
    def create_complex_graph(
        self,
        name: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Cria um grafo complexo com múltiplas ramificações e condicionais
        
        Args:
            name: Nome do grafo
            nodes: Lista de dicionários de nós com funções e metadados
            edges: Lista de dicionários definindo as conexões entre nós
        
        Returns:
            Dict com informações do grafo criado
        """
        if not self.initialized:
            self.initialize()
        
        if not self.StateGraph:
            logger.error("LangGraph não está inicializado corretamente")
            return {"error": "LangGraph não inicializado"}
        
        try:
            # Criar grafo de estado
            builder = self.StateGraph(nodes=[node["name"] for node in nodes])
            
            # Adicionar arestas conforme definido
            for edge in edges:
                from_node = edge.get("from")
                to_node = edge.get("to")
                condition = edge.get("condition")
                
                if from_node and to_node:
                    if to_node == "END":
                        to_node = self.END
                        
                    if condition:
                        # Adicionar aresta condicional
                        builder.add_conditional_edges(
                            from_node,
                            condition,
                            {
                                True: to_node,
                                False: edge.get("else", self.END)
                            }
                        )
                    else:
                        # Adicionar aresta normal
                        builder.add_edge(from_node, to_node)
            
            # Construir e compilar grafo
            graph = builder.compile()
            
            # Criar funções de nós
            node_functions = {}
            for node in nodes:
                if "function" in node:
                    node_functions[node["name"]] = node["function"]
                    
            # Armazenar informações
            graph_info = {
                "name": name,
                "graph": graph,
                "nodes": nodes,
                "edges": edges,
                "node_functions": node_functions,
                "type": "complex",
                "created_at": time.time()
            }
            
            self.graphs[name] = graph_info
            logger.info(f"Grafo complexo '{name}' criado com {len(nodes)} nós e {len(edges)} arestas")
            
            return graph_info
            
        except Exception as e:
            error = f"Erro ao criar grafo complexo '{name}': {str(e)}"
            logger.error(error)
            return {"error": error}
    
    def run_graph(
        self,
        name: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executa um grafo com dados de entrada específicos
        
        Args:
            name: Nome do grafo registrado
            input_data: Dicionário com dados de entrada
        
        Returns:
            Dict com resultados da execução
        """
        if not self.initialized:
            self.initialize()
            
        results = {
            "success": False,
            "outputs": {},
            "errors": []
        }
        
        if name not in self.graphs:
            error = f"Grafo '{name}' não encontrado"
            logger.error(error)
            results["errors"].append(error)
            return results
        
        graph_info = self.graphs[name]
        
        try:
            # Preparar dados de entrada
            state = {"inputs": input_data, "results": {}}
            
            # Obter funções de nós
            node_functions = graph_info.get("node_functions", {})
            
            # Executar grafo usando as funções de nós
            graph_instance = graph_info["graph"]
            
            # Se temos funções para os nós, usar como config de invocação
            if node_functions:
                config = {}
                for node_name, node_func in node_functions.items():
                    config[node_name] = lambda state, node_name=node_name: node_functions[node_name](state)
                    
                output = graph_instance.invoke(state, config=config)
            else:
                # Executar sem config
                output = graph_instance.invoke(state)
            
            # Processar resultados
            results["success"] = True
            results["outputs"] = output
            return results
            
        except Exception as e:
            error = f"Erro ao executar grafo '{name}': {str(e)}"
            logger.error(error)
            results["errors"].append(error)
            return results
    
    def create_analysis_graph(
        self,
        name: Optional[str] = None,
        expert_types: List[str] = ["sentiment", "factcheck"]
    ) -> Dict[str, Any]:
        """
        Cria um grafo especializado para análise de texto usando especialistas
        
        Args:
            name: Nome do grafo (opcional)
            expert_types: Lista de tipos de especialistas a usar
        
        Returns:
            Dict com informações do grafo criado
        """
        graph_name = name or f"analysis_graph_{int(time.time())}"
        
        if not self.initialized:
            self.initialize()
            
        if not self.orchestrator:
            error = "Orquestrador não inicializado"
            logger.error(error)
            return {"error": error}
            
        try:
            # Definir funções dos nós
            def parse_input(state):
                input_text = state["inputs"].get("text", "")
                return {"text": input_text, "processed": False}
                
            def run_analysis(state):
                text = state.get("text", "")
                if not text:
                    return {"error": "Texto vazio"}
                
                # Executar análise usando orquestrador
                try:
                    results = self.orchestrator.analyze_text(text, expert_types=expert_types)
                    return {"text": text, "analysis": results, "processed": True}
                except Exception as e:
                    return {"text": text, "error": str(e), "processed": False}
                    
            def summarize_analysis(state):
                if "error" in state:
                    return state  # Repassar erro
                
                analysis = state.get("analysis", {})
                text = state.get("text", "")
                
                # Resumir análises em um formato estruturado
                summary = {
                    "text": text,
                    "summary": {},
                    "processed": True
                }
                
                # Processar resultados de sentiment
                if "sentiment" in analysis:
                    sent_result = analysis["sentiment"].get("result", {})
                    summary["summary"]["sentiment"] = sent_result.get("sentiment", "neutro")
                    summary["summary"]["confidence"] = sent_result.get("confidence", 0.0)
                
                # Processar resultados de factcheck
                if "factcheck" in analysis:
                    fact_result = analysis["factcheck"].get("result", {})
                    summary["summary"]["factuality"] = fact_result.get("factuality", "unknown")
                
                summary["details"] = analysis
                return summary
            
            # Criar nós
            nodes = [
                {"name": "parse_input", "function": parse_input},
                {"name": "run_analysis", "function": run_analysis},
                {"name": "summarize_analysis", "function": summarize_analysis}
            ]
            
            # Criar grafo sequencial
            return self.create_sequential_graph(
                name=graph_name,
                nodes=nodes
            )
            
        except Exception as e:
            error = f"Erro ao criar grafo de análise: {str(e)}"
            logger.error(error)
            return {"error": error}
    
    def create_document_processing_graph(
        self,
        name: Optional[str] = None,
        experts: List[str] = ["sentiment", "factcheck", "qa"],
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Cria um grafo mais complexo para processamento de documentos
        que inclui classificação, análises e ramificações condicionais
        
        Args:
            name: Nome do grafo (opcional)
            experts: Lista de especialistas a usar
            include_summary: Se deve incluir etapa de sumarização
            
        Returns:
            Dict com informações do grafo criado
        """
        graph_name = name or f"document_graph_{int(time.time())}"
        
        if not self.initialized:
            self.initialize()
            
        if not self.orchestrator:
            error = "Orquestrador não inicializado"
            logger.error(error)
            return {"error": error}
            
        try:
            # Definir funções dos nós
            def parse_input(state):
                input_data = state["inputs"]
                text = input_data.get("text", "")
                metadata = input_data.get("metadata", {})
                return {"text": text, "metadata": metadata, "stage": "input"}
                
            def classify_document(state):
                text = state.get("text", "")
                # Classificar em categorias: article, question, opinion
                has_question_marks = "?" in text
                length = len(text)
                
                if has_question_marks and length < 200:
                    doc_type = "question"
                elif length > 500:
                    doc_type = "article"
                else:
                    doc_type = "opinion"
                    
                state["doc_type"] = doc_type
                state["stage"] = "classified"
                return state
                
            def route_document(state):
                # Função de roteamento usada para condicional
                doc_type = state.get("doc_type", "")
                return doc_type == "question"
                
            def process_question(state):
                text = state.get("text", "")
                if not "qa" in self.orchestrator.get_expert_types():
                    state["results"] = {"error": "Especialista de QA não disponível"}
                    return state
                
                qa_result = self.orchestrator.ask_question(text)
                state["results"] = {"qa_result": qa_result}
                state["stage"] = "qa_processed"
                return state
                
            def analyze_content(state):
                text = state.get("text", "")
                analysis = self.orchestrator.analyze_text(text, expert_types=experts)
                state["results"] = analysis
                state["stage"] = "analyzed"
                return state
                
            def generate_summary(state):
                if not include_summary:
                    return state
                    
                text = state.get("text", "")
                results = state.get("results", {})
                
                # Gerar resumo usando o especialista principal
                # Esta é uma implementação simplificada
                summary = f"Documento de {len(text)} caracteres analisado."
                
                # Adicionar detalhes baseado nos resultados
                if "sentiment" in results:
                    sentiment = results["sentiment"].get("result", {}).get("sentiment", "neutro")
                    summary += f" Sentimento identificado: {sentiment}."
                    
                state["summary"] = summary
                state["stage"] = "summarized"
                return state
            
            # Definir nós
            nodes = [
                {"name": "parse_input", "function": parse_input},
                {"name": "classify_document", "function": classify_document},
                {"name": "process_question", "function": process_question},
                {"name": "analyze_content", "function": analyze_content},
                {"name": "generate_summary", "function": generate_summary}
            ]
            
            # Definir arestas
            edges = [
                {"from": "parse_input", "to": "classify_document"},
                {
                    "from": "classify_document", 
                    "to": "process_question", 
                    "condition": route_document,
                    "else": "analyze_content"
                },
                {"from": "process_question", "to": "generate_summary"},
                {"from": "analyze_content", "to": "generate_summary"},
                {"from": "generate_summary", "to": "END"}
            ]
            
            # Criar grafo complexo
            return self.create_complex_graph(
                name=graph_name,
                nodes=nodes,
                edges=edges
            )
            
        except Exception as e:
            error = f"Erro ao criar grafo de processamento de documento: {str(e)}"
            logger.error(error)
            return {"error": error}
            
    def analyze_text_with_graph(
        self,
        text: str,
        graph_name: Optional[str] = None,
        expert_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analisa um texto usando um grafo LangGraph
        
        Args:
            text: Texto a ser analisado
            graph_name: Nome do grafo a usar (opcional)
            expert_types: Lista de tipos de especialistas (opcional)
            
        Returns:
            Dict com resultados da análise
        """
        if not self.initialized:
            self.initialize()
            
        # Configurar especialistas se fornecidos
        experts = expert_types or ["sentiment", "factcheck"]
        
        # Usar grafo existente ou criar novo
        if graph_name and graph_name in self.graphs:
            graph_info = self.graphs[graph_name]
        else:
            graph_info = self.create_analysis_graph(expert_types=experts)
            graph_name = graph_info.get("name")
        
        # Verificar se criação do grafo foi bem-sucedida
        if "error" in graph_info:
            return graph_info
        
        # Executar grafo
        results = self.run_graph(
            name=graph_name,
            input_data={"text": text}
        )
        
        return results
