#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sistema EzioFilho Unified - Interface principal
Integra todos os componentes do sistema unificado em uma aplicação completa
"""
import os
import json
import time
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("EzioUnified")

def import_modules():
    """Importa todos os módulos necessários"""
    try:
        # Adicionar diretório atual ao path do Python
        current_dir = Path(__file__).parent.absolute()
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        # Verificar core
        core_dir = current_dir / "core"
        if not core_dir.exists():
            logger.error(f"Diretório core não encontrado: {core_dir}")
            return False
        
        # Importar componentes unificados
        from core.unified_orchestrator import UnifiedOrchestrator
        from core.unified_base_expert import EzioBaseExpert
        from core.autogen_integration import AutogenIntegration
        from core.langgraph_integration import LangGraphIntegration
        from core.gui_interface import GradioInterface
        
        logger.info("Importação bem-sucedida dos módulos principais")
        return True
        
    except ImportError as e:
        logger.error(f"Falha ao importar módulos: {e}")
        return False

def check_dependencies():
    """Verifica dependências críticas"""
    deps = {
        "torch": "Processamento de modelos",
        "transformers": "Modelos de linguagem",
        "gradio": "Interface gráfica",
        "autogen": "Agentes autônomos",
        "langgraph": "Grafos de processamento"
    }
    
    missing = []
    optional_missing = []
    
    for dep, desc in deps.items():
        try:
            if dep == "autogen":
                # PyAutogen usa outro nome de importação
                __import__("pyautogen")
            else:
                __import__(dep)
            logger.info(f"✓ {dep}: OK")
        except ImportError:
            if dep in ["torch", "transformers"]:
                missing.append(f"{dep} ({desc})")
            else:
                optional_missing.append(f"{dep} ({desc})")
    
    if missing:
        logger.error(f"Dependências críticas faltando: {', '.join(missing)}")
        logger.error("Instale com: pip install -r requirements-unified.txt")
        return False
        
    if optional_missing:
        logger.warning(f"Dependências opcionais faltando: {', '.join(optional_missing)}")
        logger.warning("Algumas funcionalidades podem estar indisponíveis")
        
    return len(missing) == 0

def initialize_system(config_path: Optional[str] = None, expert_types: Optional[List[str]] = None):
    """
    Inicializa o sistema com todos os componentes
    
    Args:
        config_path: Caminho para configuração
        expert_types: Lista de tipos de especialistas para inicializar
        
    Returns:
        Dict com componentes inicializados
    """
    system = {
        "initialized": False,
        "orchestrator": None,
        "autogen": None,
        "langgraph": None,
        "gui": None,
        "initialization_time": time.time()
    }
    
    try:
        # Importar módulos se necessário
        if "unified_orchestrator" not in sys.modules:
            if not import_modules():
                return system
        
        from core.unified_orchestrator import UnifiedOrchestrator
        from core.autogen_integration import AutogenIntegration
        from core.langgraph_integration import LangGraphIntegration
        from core.gui_interface import GradioInterface
        
        # Inicializar orquestrador
        logger.info("Inicializando orquestrador...")
        orchestrator = UnifiedOrchestrator(config_path=config_path)
        if not orchestrator.initialize(expert_types=expert_types):
            logger.error("Falha ao inicializar orquestrador")
            system["error"] = "Falha ao inicializar orquestrador"
            return system
            
        system["orchestrator"] = orchestrator
        
        # Inicializar integração AutoGen
        try:
            logger.info("Inicializando integração com AutoGen...")
            autogen_integration = AutogenIntegration(orchestrator=orchestrator)
            if autogen_integration.initialize():
                system["autogen"] = autogen_integration
            else:
                logger.warning("Inicialização parcial: AutoGen indisponível")
        except Exception as e:
            logger.warning(f"Erro ao inicializar AutoGen: {e}")
        
        # Inicializar integração LangGraph
        try:
            logger.info("Inicializando integração com LangGraph...")
            langgraph_integration = LangGraphIntegration(orchestrator=orchestrator)
            if langgraph_integration.initialize():
                system["langgraph"] = langgraph_integration
            else:
                logger.warning("Inicialização parcial: LangGraph indisponível")
        except Exception as e:
            logger.warning(f"Erro ao inicializar LangGraph: {e}")
        
        # Inicializar interface gráfica
        try:
            logger.info("Inicializando interface gráfica...")
            gui = GradioInterface(
                orchestrator=orchestrator,
                title="EzioFilho Unified",
                description="Sistema unificado de análise e processamento de texto"
            )
            if gui.initialize():
                system["gui"] = gui
            else:
                logger.warning("Inicialização parcial: GUI indisponível")
        except Exception as e:
            logger.warning(f"Erro ao inicializar GUI: {e}")
        
        # Concluído
        system["initialized"] = True
        system["initialization_time"] = time.time() - system["initialization_time"]
        
        logger.info(f"Sistema inicializado em {system['initialization_time']:.2f}s")
        
        return system
        
    except Exception as e:
        logger.error(f"Erro ao inicializar sistema: {e}")
        system["error"] = str(e)
        return system

def run_cli(system: Dict[str, Any], args: argparse.Namespace):
    """
    Executa o sistema em modo CLI
    
    Args:
        system: Sistema inicializado
        args: Argumentos da linha de comando
    """
    orchestrator = system.get("orchestrator")
    if not orchestrator:
        logger.error("Orquestrador não disponível")
        return
    
    # Processar comando específico
    if args.command == "analyze":
        # Analisar texto
        if args.file:
            # Analisar de arquivo
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    text = f.read()
                logger.info(f"Arquivo carregado: {args.file} ({len(text)} caracteres)")
            except Exception as e:
                logger.error(f"Erro ao ler arquivo: {e}")
                return
        else:
            # Usar texto fornecido
            text = args.text or input("Digite o texto para análise: ")
        
        # Definir especialistas
        expert_types = args.experts.split(",") if args.experts else None
        
        # Executar análise
        logger.info(f"Analisando texto com especialistas: {expert_types or 'todos'}")
        results = orchestrator.analyze_text(text, expert_types=expert_types)
        
        # Mostrar resultados
        print("\n===== RESULTADOS DA ANÁLISE =====\n")
        for expert_type, expert_results in results.items():
            print(f"--- {expert_type.upper()} ---")
            if "result" in expert_results:
                if isinstance(expert_results["result"], dict):
                    for key, value in expert_results["result"].items():
                        print(f"{key}: {value}")
                else:
                    print(expert_results["result"])
            print()
            
        # Salvar resultados se solicitado
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Resultados salvos em: {args.output}")
            except Exception as e:
                logger.error(f"Erro ao salvar resultados: {e}")
                
    elif args.command == "autogen":
        # Executar análise com AutoGen
        autogen_integration = system.get("autogen")
        if not autogen_integration:
            logger.error("Integração com AutoGen não disponível")
            return
            
        # Texto para análise
        if args.file:
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    text = f.read()
                logger.info(f"Arquivo carregado: {args.file} ({len(text)} caracteres)")
            except Exception as e:
                logger.error(f"Erro ao ler arquivo: {e}")
                return
        else:
            text = args.text or input("Digite o texto para análise com AutoGen: ")
            
        # Definir especialistas
        expert_types = args.experts.split(",") if args.experts else ["sentiment", "factcheck"]
            
        # Executar análise com AutoGen
        logger.info(f"Analisando texto com AutoGen e especialistas: {expert_types}")
        results = autogen_integration.run_analysis_flow(
            text=text,
            analysis_types=expert_types,
            return_agent_conversation=True
        )
        
        # Mostrar insights
        print("\n===== INSIGHTS DO AUTOGEN =====\n")
        for insight in results.get("insights", []):
            print(insight)
            print("\n" + "-" * 50 + "\n")
            
        # Salvar resultados se solicitado
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Resultados salvos em: {args.output}")
            except Exception as e:
                logger.error(f"Erro ao salvar resultados: {e}")
                
    elif args.command == "graph":
        # Executar análise com LangGraph
        langgraph_integration = system.get("langgraph")
        if not langgraph_integration:
            logger.error("Integração com LangGraph não disponível")
            return
            
        # Texto para análise
        if args.file:
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    text = f.read()
                logger.info(f"Arquivo carregado: {args.file} ({len(text)} caracteres)")
            except Exception as e:
                logger.error(f"Erro ao ler arquivo: {e}")
                return
        else:
            text = args.text or input("Digite o texto para análise com LangGraph: ")
            
        # Definir especialistas
        expert_types = args.experts.split(",") if args.experts else ["sentiment", "factcheck"]
            
        # Criar grafo se não existir
        graph_name = "analysis_graph"
        if graph_name not in langgraph_integration.graphs:
            logger.info(f"Criando grafo de análise: {graph_name}")
            langgraph_integration.create_analysis_graph(
                name=graph_name,
                expert_types=expert_types
            )
            
        # Executar análise com LangGraph
        logger.info(f"Analisando texto com grafo LangGraph: {graph_name}")
        results = langgraph_integration.run_graph(
            name=graph_name,
            input_data={"text": text}
        )
        
        # Mostrar resultados
        print("\n===== RESULTADOS DO LANGGRAPH =====\n")
        output = results.get("outputs", {})
        if "summary" in output:
            for key, value in output["summary"].items():
                print(f"{key}: {value}")
        else:
            print(json.dumps(output, indent=2, ensure_ascii=False))
            
        # Salvar resultados se solicitado
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Resultados salvos em: {args.output}")
            except Exception as e:
                logger.error(f"Erro ao salvar resultados: {e}")
                
    elif args.command == "list-experts":
        # Listar especialistas disponíveis
        experts = orchestrator.get_expert_types()
        print("\nEspecialistas disponíveis:")
        for i, expert in enumerate(experts, 1):
            print(f"{i}. {expert}")
        print()
        
    else:
        logger.error(f"Comando desconhecido: {args.command}")

def main():
    """Função principal de inicialização e execução"""
    parser = argparse.ArgumentParser(description="Sistema EzioFilho Unified")
    
    parser.add_argument("--config", help="Caminho para arquivo de configuração")
    parser.add_argument("--experts", help="Lista de especialistas separados por vírgula")
    parser.add_argument("--debug", action="store_true", help="Ativar modo debug")
    parser.add_argument("--log-file", help="Arquivo de log")
    
    # Subcomandos
    subparsers = parser.add_subparsers(dest="command", help="Comando a executar")
    
    # Comando de GUI
    gui_parser = subparsers.add_parser("gui", help="Iniciar interface gráfica")
    gui_parser.add_argument("--port", type=int, default=7860, help="Porta para servidor web")
    gui_parser.add_argument("--share", action="store_true", help="Compartilhar publicamente")
    gui_parser.add_argument("--basic", action="store_true", help="Usar interface básica")
    
    # Comando de análise
    analyze_parser = subparsers.add_parser("analyze", help="Analisar texto")
    analyze_parser.add_argument("--text", help="Texto para análise")
    analyze_parser.add_argument("--file", help="Arquivo com texto para análise")
    analyze_parser.add_argument("--output", help="Arquivo para salvar resultados")
    
    # Comando de autogen
    autogen_parser = subparsers.add_parser("autogen", help="Analisar com AutoGen")
    autogen_parser.add_argument("--text", help="Texto para análise")
    autogen_parser.add_argument("--file", help="Arquivo com texto para análise")
    autogen_parser.add_argument("--output", help="Arquivo para salvar resultados")
    
    # Comando de langgraph
    graph_parser = subparsers.add_parser("graph", help="Analisar com LangGraph")
    graph_parser.add_argument("--text", help="Texto para análise")
    graph_parser.add_argument("--file", help="Arquivo com texto para análise")
    graph_parser.add_argument("--output", help="Arquivo para salvar resultados")
    
    # Comando para listar especialistas
    subparsers.add_parser("list-experts", help="Listar especialistas disponíveis")
    
    # Processar argumentos
    args = parser.parse_args()
    
    # Configurar logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.root.setLevel(log_level)
    
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | [%(name)s] | %(message)s"))
        logging.root.addHandler(file_handler)
    
    # Verificar dependências
    if not check_dependencies():
        logger.warning("Algumas dependências estão faltando. O sistema pode não funcionar corretamente.")
    
    # Inicializar sistema
    expert_types = args.experts.split(",") if args.experts else None
    system = initialize_system(config_path=args.config, expert_types=expert_types)
    
    if not system["initialized"]:
        logger.error("Falha ao inicializar sistema")
        if "error" in system:
            logger.error(f"Erro: {system['error']}")
        return 1
    
    # Processar comando ou iniciar GUI
    if args.command == "gui":
        # Iniciar interface gráfica
        gui = system.get("gui")
        if not gui:
            logger.error("Interface gráfica não disponível")
            return 1
            
        interface_type = "basic" if args.basic else "advanced"
        logger.info(f"Iniciando interface {interface_type} na porta {args.port}")
        
        gui.launch(
            interface_type=interface_type,
            share=args.share,
            debug=args.debug
        )
    elif args.command:
        # Executar comando CLI
        run_cli(system, args)
    else:
        # Sem comando, mostrar ajuda
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
