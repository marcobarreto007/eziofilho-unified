#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testes de integração para o sistema EzioFilho Unified
Testa novos componentes: AutoGen, LangGraph e GUI
"""

import os
import sys
import json
import time
import unittest
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("TestIntegrations")

# Adicionar diretório raiz ao path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Importar componentes unificados
try:
    from core.unified_orchestrator import UnifiedOrchestrator
    from core.autogen_integration import AutogenIntegration
    from core.langgraph_integration import LangGraphIntegration
    from core.gui_interface import GradioInterface
except ImportError as e:
    logger.error(f"Erro ao importar componentes: {e}")
    sys.exit(1)

class TestAutogenIntegration(unittest.TestCase):
    """Testes para integração com PyAutogen"""
    
    @classmethod
    def setUpClass(cls):
        """Inicializa recursos compartilhados"""
        try:
            logger.info("Inicializando orquestrador para testes de AutoGen...")
            cls.orchestrator = UnifiedOrchestrator()
            cls.orchestrator.initialize(expert_types=["sentiment"])
            
            logger.info("Inicializando AutogenIntegration...")
            cls.autogen_integration = AutogenIntegration(orchestrator=cls.orchestrator)
            cls.autogen_integration.initialize()
            
        except Exception as e:
            logger.error(f"Falha ao inicializar para testes: {e}")
            cls.orchestrator = None
            cls.autogen_integration = None
    
    def setUp(self):
        """Prepara cada teste"""
        if self.orchestrator is None or self.autogen_integration is None:
            self.skipTest("Orquestrador ou AutogenIntegration não inicializados corretamente")
    
    def test_autogen_initialization(self):
        """Testa inicialização do AutogenIntegration"""
        self.assertTrue(self.autogen_integration.initialized)
        self.assertIsNone(self.autogen_integration.initialization_error)
    
    def test_autogen_agents(self):
        """Testa criação de agentes"""
        # Verificar agentes padrão
        self.assertIn("assistant", self.autogen_integration.agents)
        self.assertIn("user_proxy", self.autogen_integration.agents)
        
        # Criar novo agente
        test_agent = self.autogen_integration.create_agent(
            name="test_agent",
            agent_type="assistant",
            system_message="Você é um agente de teste"
        )
        
        self.assertIsNotNone(test_agent)
        self.assertIn("test_agent", self.autogen_integration.agents)
    
    def test_workflow_creation(self):
        """Testa criação de workflow"""
        workflow = self.autogen_integration.create_workflow(
            workflow_name="test_workflow",
            agents=["assistant", "user_proxy"]
        )
        
        self.assertEqual(workflow["name"], "test_workflow")
        self.assertTrue(workflow["initialized"])
        self.assertEqual(len(workflow["agents"]), 2)
        self.assertIn("test_workflow", self.autogen_integration.workflows)
    
    @unittest.skip("Testes com execução de fluxo podem ser lentos")
    def test_analysis_flow(self):
        """Testa fluxo de análise completo"""
        text = "Este produto é excelente! Recomendo muito."
        
        results = self.autogen_integration.run_analysis_flow(
            text=text,
            analysis_types=["sentiment"],
            return_agent_conversation=False
        )
        
        self.assertIn("analyses", results)
        self.assertIn("insights", results)
        self.assertIn("text", results)
        self.assertEqual(results["text"], text)
        self.assertIn("sentiment", results["analyses"])


class TestLangGraphIntegration(unittest.TestCase):
    """Testes para integração com LangGraph"""
    
    @classmethod
    def setUpClass(cls):
        """Inicializa recursos compartilhados"""
        try:
            logger.info("Inicializando orquestrador para testes de LangGraph...")
            cls.orchestrator = UnifiedOrchestrator()
            cls.orchestrator.initialize(expert_types=["sentiment"])
            
            # Verificar se LangGraph está disponível
            try:
                import langgraph
                logger.info("LangGraph disponível para testes")
                
                logger.info("Inicializando LangGraphIntegration...")
                cls.langgraph_integration = LangGraphIntegration(orchestrator=cls.orchestrator)
                cls.langgraph_integration.initialize()
                
            except ImportError:
                logger.warning("LangGraph não instalado, pulando testes")
                cls.langgraph_integration = None
                
        except Exception as e:
            logger.error(f"Falha ao inicializar para testes: {e}")
            cls.orchestrator = None
            cls.langgraph_integration = None
    
    def setUp(self):
        """Prepara cada teste"""
        if self.orchestrator is None or self.langgraph_integration is None:
            self.skipTest("Orquestrador ou LangGraphIntegration não inicializados corretamente")
            
        if not self.langgraph_integration.langgraph:
            self.skipTest("LangGraph não está disponível")
    
    def test_langgraph_initialization(self):
        """Testa inicialização do LangGraphIntegration"""
        self.assertTrue(self.langgraph_integration.initialized)
        self.assertIsNone(self.langgraph_integration.initialization_error)
        self.assertIsNotNone(self.langgraph_integration.langgraph)
        self.assertIsNotNone(self.langgraph_integration.StateGraph)
    
    def test_create_sequential_graph(self):
        """Testa criação de grafo sequencial"""
        def node_func1(state):
            return {"processed": True, "node": "node1"}
            
        def node_func2(state):
            return {"processed": True, "node": "node2"}
            
        nodes = [
            {"name": "node1", "function": node_func1},
            {"name": "node2", "function": node_func2}
        ]
        
        graph_info = self.langgraph_integration.create_sequential_graph(
            name="test_graph",
            nodes=nodes
        )
        
        self.assertEqual(graph_info["name"], "test_graph")
        self.assertEqual(len(graph_info["nodes"]), 2)
        self.assertIn("graph", graph_info)
        self.assertIn("test_graph", self.langgraph_integration.graphs)
    
    def test_create_analysis_graph(self):
        """Testa criação de grafo de análise"""
        graph_info = self.langgraph_integration.create_analysis_graph(
            name="test_analysis",
            expert_types=["sentiment"]
        )
        
        self.assertEqual(graph_info["name"], "test_analysis")
        self.assertEqual(len(graph_info["nodes"]), 3)  # parse, analyze, summarize
        self.assertIn("graph", graph_info)
        self.assertIn("test_analysis", self.langgraph_integration.graphs)
    
    @unittest.skip("Testes com execução de grafo podem ser lentos")
    def test_run_graph(self):
        """Testa execução de um grafo"""
        # Criar grafo simples para teste
        def echo_func(state):
            return {"output": state["inputs"]["text"]}
            
        nodes = [
            {"name": "echo", "function": echo_func}
        ]
        
        self.langgraph_integration.create_sequential_graph(
            name="echo_graph",
            nodes=nodes
        )
        
        # Executar grafo
        text = "Test message"
        results = self.langgraph_integration.run_graph(
            name="echo_graph",
            input_data={"text": text}
        )
        
        self.assertTrue(results["success"])
        self.assertEqual(results["outputs"]["output"], text)


class TestGradioInterface(unittest.TestCase):
    """Testes para interface Gradio"""
    
    @classmethod
    def setUpClass(cls):
        """Inicializa recursos compartilhados"""
        try:
            logger.info("Inicializando orquestrador para testes de GUI...")
            cls.orchestrator = UnifiedOrchestrator()
            cls.orchestrator.initialize(expert_types=["sentiment"])
            
            # Verificar se Gradio está disponível
            try:
                import gradio as gr
                logger.info("Gradio disponível para testes")
                
                logger.info("Inicializando GradioInterface...")
                cls.gui = GradioInterface(orchestrator=cls.orchestrator)
                cls.gui.initialize()
                
            except ImportError:
                logger.warning("Gradio não instalado, pulando testes")
                cls.gui = None
                
        except Exception as e:
            logger.error(f"Falha ao inicializar para testes: {e}")
            cls.orchestrator = None
            cls.gui = None
    
    def setUp(self):
        """Prepara cada teste"""
        if self.orchestrator is None or self.gui is None:
            self.skipTest("Orquestrador ou GradioInterface não inicializados corretamente")
            
        if not self.gui.gr:
            self.skipTest("Gradio não está disponível")
    
    def test_gui_initialization(self):
        """Testa inicialização do GradioInterface"""
        self.assertTrue(self.gui.initialized)
        self.assertIsNone(self.gui.initialization_error)
        self.assertIsNotNone(self.gui.gr)
    
    def test_basic_interface_creation(self):
        """Testa criação da interface básica"""
        app = self.gui.create_basic_interface()
        self.assertIsNotNone(app)
        self.assertIsNotNone(self.gui.app)
    
    def test_advanced_interface_creation(self):
        """Testa criação da interface avançada"""
        app = self.gui.create_advanced_interface()
        self.assertIsNotNone(app)
        self.assertIsNotNone(self.gui.app)
    
    def test_analyze_and_format(self):
        """Testa funções de análise e formatação"""
        text = "Este produto é excelente! Recomendo muito."
        results = self.gui._analyze_text(text, ["sentiment"])
        
        self.assertIn("sentiment", results)
        self.assertIn("text", results)
        self.assertEqual(results["text"], text)
        
        # Testar formatação
        formatted = self.gui._format_results(results)
        self.assertIsInstance(formatted, str)
        self.assertIn("Resultados da Análise", formatted)
        self.assertIn("Sentiment", formatted)


if __name__ == "__main__":
    unittest.main()
