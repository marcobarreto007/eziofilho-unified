"""
Autogen Integration - Integração do PyAutogen com o sistema unificado EzioFilho
Implementa fluxos de agentes usando o framework PyAutogen da Microsoft
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

logger = logging.getLogger("AutogenIntegration")

class AutogenIntegration:
    """
    Integração do sistema EzioFilho com PyAutogen
    """
    
    # Versão da integração
    VERSION = "1.0.0"
    
    def __init__(
        self,
        orchestrator: UnifiedOrchestrator = None,
        config_path: Optional[Union[str, Path]] = None,
        autogen_config: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa a integração AutoGen
        
        Args:
            orchestrator: Instância do UnifiedOrchestrator (opcional)
            config_path: Caminho para arquivo de configuração
            autogen_config: Configuração personalizada para AutoGen
        """
        self.orchestrator = orchestrator
        self.config_path = config_path
        self.autogen_config = autogen_config or {}
        
        # Estado interno
        self.agents = {}
        self.workflows = {}
        self.initialized = False
        self.initialization_error = None
        
        # Importar autogen
        try:
            import autogen
            self.autogen = autogen
            logger.info(f"PyAutogen {autogen.__version__} importado com sucesso")
        except ImportError as e:
            self.initialization_error = f"Erro ao importar PyAutogen: {str(e)}"
            logger.error(self.initialization_error)
            logger.error("Instale com: pip install pyautogen")
            self.autogen = None
    
    def initialize(self) -> bool:
        """
        Inicializa a integração Autogen
        
        Returns:
            bool: True se inicialização bem-sucedida
        """
        if self.initialized:
            return True
        
        start_time = time.time()
        
        try:
            if self.autogen is None:
                return False
                
            # Inicializar orquestrador se não foi fornecido
            if self.orchestrator is None:
                self.orchestrator = UnifiedOrchestrator(config_path=self.config_path)
                self.orchestrator.initialize()
            
            # Carregar configuração Autogen
            if self.config_path:
                self._load_autogen_config()
            
            # Inicializar agentes padrões
            self._initialize_default_agents()
            
            self.initialized = True
            logger.info(f"AutogenIntegration inicializado em {time.time() - start_time:.2f}s")
            return True
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"Erro ao inicializar AutogenIntegration: {e}")
            return False
    
    def _load_autogen_config(self) -> None:
        """Carrega configuração do AutoGen de arquivo"""
        try:
            if not self.config_path:
                return
                
            config_path = Path(self.config_path)
            if not config_path.exists():
                logger.warning(f"Arquivo de configuração não encontrado: {config_path}")
                return
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if "autogen" in config:
                self.autogen_config = config["autogen"]
                logger.info(f"Configuração AutoGen carregada: {len(self.autogen_config)} entradas")
        except Exception as e:
            logger.error(f"Erro ao carregar configuração AutoGen: {e}")
    
    def _initialize_default_agents(self) -> None:
        """Inicializa agentes padrão do AutoGen"""
        if not self.autogen:
            return
            
        try:
            # Configurações do LLM
            config_list = self.autogen_config.get("config_list", [])
            
            # Se não houver configurações, criar uma padrão usando modelos locais
            if not config_list and self.orchestrator:
                # Usar o primeiro especialista disponível como modelo
                experts = self.orchestrator.get_experts()
                if experts:
                    expert = next(iter(experts.values()))
                    # Criar configuração para o autogen usando o modelo do especialista
                    config_list = [
                        {
                            "model": "EzioUnified",
                            "api_key": "not-needed",
                            "ezio_expert": expert
                        }
                    ]
            
            # Agente assistente
            self.agents["assistant"] = self.autogen.AssistantAgent(
                name="assistant",
                system_message="Você é um assistente avançado que ajuda a resolver problemas.",
                llm_config={"config_list": config_list}
            )
            
            # Agente usuário
            self.agents["user_proxy"] = self.autogen.UserProxyAgent(
                name="user_proxy",
                human_input_mode="TERMINATE",
                max_consecutive_auto_reply=10,
                is_termination_msg=lambda msg: "TAREFA CONCLUÍDA" in msg.get("content", "").upper(),
                code_execution_config={"work_dir": "workspace"},
                llm_config={"config_list": config_list}
            )
            
            logger.info("Agentes padrão inicializados com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar agentes padrão: {e}")
    
    def create_agent(
        self,
        name: str,
        agent_type: str,
        system_message: Optional[str] = None,
        **kwargs
    ):
        """
        Cria um novo agente AutoGen
        
        Args:
            name: Nome do agente
            agent_type: Tipo do agente ('assistant' ou 'user_proxy')
            system_message: Mensagem do sistema
            **kwargs: Argumentos adicionais para o construtor do agente
        
        Returns:
            Objeto agente do AutoGen
        """
        if not self.initialized:
            self.initialize()
        
        if not self.autogen:
            logger.error("AutoGen não está disponível")
            return None
        
        try:
            # Configurações base do LLM
            config_list = self.autogen_config.get("config_list", [])
            llm_config = {"config_list": config_list}
            
            # Criar instância do agente
            if agent_type.lower() == "assistant":
                agent = self.autogen.AssistantAgent(
                    name=name,
                    system_message=system_message or "Você é um assistente útil e experiente.",
                    llm_config=llm_config,
                    **kwargs
                )
            elif agent_type.lower() == "user_proxy":
                agent = self.autogen.UserProxyAgent(
                    name=name,
                    human_input_mode="TERMINATE",
                    max_consecutive_auto_reply=10,
                    is_termination_msg=lambda msg: "TAREFA CONCLUÍDA" in msg.get("content", "").upper(),
                    system_message=system_message,
                    llm_config=llm_config,
                    **kwargs
                )
            else:
                logger.error(f"Tipo de agente desconhecido: {agent_type}")
                return None
            
            # Armazenar o agente
            self.agents[name] = agent
            logger.info(f"Agente '{name}' do tipo '{agent_type}' criado com sucesso")
            return agent
            
        except Exception as e:
            logger.error(f"Erro ao criar agente '{name}': {e}")
            return None
    
    def create_workflow(
        self,
        workflow_name: str,
        agents: List[str]
    ) -> Dict[str, Any]:
        """
        Cria um fluxo de trabalho com agentes específicos
        
        Args:
            workflow_name: Nome do fluxo de trabalho
            agents: Lista de nomes de agentes
        
        Returns:
            Dict com informações do workflow
        """
        workflow = {
            "name": workflow_name,
            "agents": [],
            "initialized": False,
            "timestamp": time.time()
        }
        
        # Adicionar agentes ao workflow
        for agent_name in agents:
            if agent_name in self.agents:
                workflow["agents"].append(self.agents[agent_name])
            else:
                logger.warning(f"Agente '{agent_name}' não encontrado")
        
        if not workflow["agents"]:
            logger.error(f"Workflow '{workflow_name}' não tem agentes válidos")
            return workflow
        
        workflow["initialized"] = True
        self.workflows[workflow_name] = workflow
        logger.info(f"Workflow '{workflow_name}' criado com {len(workflow['agents'])} agentes")
        
        return workflow
    
    def run_workflow(
        self,
        workflow_name: str,
        initial_message: str,
        sender_name: Optional[str] = None,
        receiver_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executa um fluxo de trabalho com uma mensagem inicial
        
        Args:
            workflow_name: Nome do fluxo de trabalho
            initial_message: Mensagem inicial para o workflow
            sender_name: Nome do agente emissor (opcional)
            receiver_name: Nome do agente receptor (opcional)
        
        Returns:
            Dict com os resultados da execução
        """
        if not self.initialized:
            self.initialize()
        
        results = {
            "success": False,
            "messages": [],
            "timestamp": time.time(),
            "errors": []
        }
        
        # Verificar se o workflow existe
        if workflow_name not in self.workflows:
            error = f"Workflow '{workflow_name}' não encontrado"
            logger.error(error)
            results["errors"].append(error)
            return results
        
        workflow = self.workflows[workflow_name]
        if not workflow["initialized"]:
            error = f"Workflow '{workflow_name}' não está inicializado"
            logger.error(error)
            results["errors"].append(error)
            return results
        
        try:
            # Definir emissor e receptor
            sender = None
            receiver = None
            
            if sender_name:
                for agent in workflow["agents"]:
                    if agent.name == sender_name:
                        sender = agent
                        break
            else:
                # Usar primeiro agente como emissor padrão
                sender = workflow["agents"][0]
            
            if receiver_name:
                for agent in workflow["agents"]:
                    if agent.name == receiver_name:
                        receiver = agent
                        break
            else:
                # Usar segundo agente como receptor padrão
                if len(workflow["agents"]) > 1:
                    receiver = workflow["agents"][1]
                else:
                    receiver = workflow["agents"][0]  # Mesmo agente se só houver um
            
            # Iniciar chat
            logger.info(f"Iniciando conversa: {sender.name} -> {receiver.name}")
            chat_response = sender.initiate_chat(
                recipient=receiver,
                message=initial_message
            )
            
            # Registrar resultados
            results["success"] = True
            results["messages"] = chat_response.chat_history
            
            return results
            
        except Exception as e:
            error = f"Erro ao executar workflow '{workflow_name}': {str(e)}"
            logger.error(error)
            results["errors"].append(error)
            return results
    
    def run_analysis_flow(
        self,
        text: str,
        analysis_types: List[str] = ["sentiment", "factcheck"],
        return_agent_conversation: bool = False
    ) -> Dict[str, Any]:
        """
        Executa um fluxo de trabalho especializado para análises usando especialistas
        
        Args:
            text: Texto a ser analisado
            analysis_types: Tipos de análise a serem realizados
            return_agent_conversation: Se True, retorna a conversa completa dos agentes
        
        Returns:
            Dict com os resultados das análises
        """
        # Inicializar se necessário
        if not self.initialized:
            self.initialize()
        
        # Verificar se orquestrador está disponível
        if not self.orchestrator:
            return {"error": "Orquestrador não inicializado"}
        
        # Criar workflow especializado para análises
        workflow_name = "analysis_flow"
        
        # Criar agentes especializados se não existirem
        if "analyst" not in self.agents:
            self.create_agent(
                name="analyst",
                agent_type="assistant",
                system_message="Você é um analista especializado que integra múltiplas análises para extrair insights."
            )
        
        if "user_analyst" not in self.agents:
            self.create_agent(
                name="user_analyst",
                agent_type="user_proxy",
                human_input_mode="NEVER",
                code_execution_config={"work_dir": "workspace"}
            )
        
        # Criar workflow se não existir
        if workflow_name not in self.workflows:
            self.create_workflow(
                workflow_name=workflow_name,
                agents=["analyst", "user_analyst"]
            )
        
        # Preparar mensagem com análises dos especialistas
        analyses = self.orchestrator.analyze_text(text, expert_types=analysis_types)
        
        # Formatar análises para a mensagem
        formatted_analyses = json.dumps(analyses, indent=2, ensure_ascii=False)
        
        prompt = f"""
        Analise o seguinte texto e os resultados das análises especializadas:
        
        TEXTO:
        {text}
        
        ANÁLISES DOS ESPECIALISTAS:
        {formatted_analyses}
        
        Por favor, integre essas análises e forneça insights sobre o texto.
        """
        
        # Executar fluxo
        results = self.run_workflow(
            workflow_name=workflow_name,
            initial_message=prompt,
            sender_name="user_analyst",
            receiver_name="analyst"
        )
        
        # Processar e retornar resultados
        if results["success"]:
            output = {
                "text": text,
                "analyses": analyses,
                "insights": [],
                "timestamp": time.time()
            }
            
            # Extrair insights da conversa
            for msg in results["messages"]:
                if msg.get("role") == "assistant" and msg.get("name") == "analyst":
                    output["insights"].append(msg.get("content", ""))
            
            # Adicionar conversa completa se solicitado
            if return_agent_conversation:
                output["conversation"] = results["messages"]
                
            return output
        else:
            return {
                "error": "Falha ao executar análise",
                "details": results.get("errors", [])
            }


# Exemplo de função auxiliar para criar assistentes
def create_ezio_assistant(
    orchestrator: Union[UnifiedOrchestrator, str, Path],
    name: str = "ezio_assistant",
    experts: List[str] = ["sentiment", "factcheck", "qa"],
    system_message: Optional[str] = None
) -> Optional[Any]:
    """
    Cria um assistente AutoGen integrado com especialistas EzioFilho
    
    Args:
        orchestrator: Instância do UnifiedOrchestrator ou caminho para config
        name: Nome do assistente
        experts: Lista de tipos de especialistas a usar
        system_message: Mensagem do sistema personalizada
    
    Returns:
        Instância do assistente ou None se falhar
    """
    try:
        import autogen
        
        # Inicializar orquestrador se for caminho
        if isinstance(orchestrator, (str, Path)):
            orch_instance = UnifiedOrchestrator(config_path=orchestrator)
            orch_instance.initialize(expert_types=experts)
        else:
            orch_instance = orchestrator
        
        # Criar mensagem do sistema
        if system_message is None:
            system_message = """Você é um assistente inteligente com acesso a especialistas em:
            - Análise de Sentimento
            - Verificação de Fatos
            - Resposta a Perguntas
            
            Use essas capacidades para fornecer respostas precisas e confiáveis.
            """
        
        # Criar integração
        integration = AutogenIntegration(orchestrator=orch_instance)
        if not integration.initialize():
            logger.error("Falha ao inicializar integração AutoGen")
            return None
        
        # Criar e retornar o assistente
        return integration.create_agent(
            name=name,
            agent_type="assistant",
            system_message=system_message
        )
        
    except ImportError:
        logger.error("AutoGen não está instalado. Execute: pip install pyautogen")
        return None
    except Exception as e:
        logger.error(f"Erro ao criar assistente EzioFilho: {e}")
        return None
