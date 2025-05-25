"""
UnifiedOrchestrator - Orquestrador corrigido para o sistema EzioFilho
"""
import logging
import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import torch
from transformers import BitsAndBytesConfig

logger = logging.getLogger(__name__)

class UnifiedOrchestrator:
    """Orquestrador unificado para especialistas do EzioFilho"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa o orquestrador
        
        Args:
            config_path: Caminho para arquivo de configuração
        """
        self.config_path = config_path
        self.experts = {}
        self.expert_configs = {}
        self.device = self._detect_device()
        self.quantization_enabled = True
        self.initialized = False
        
        # Carregar configuração
        self._load_config()
        
        logger.info(f"Orquestrador inicializado - Device: {self.device}")
    
    def _detect_device(self) -> str:
        """Detecta o dispositivo disponível (GPU/CPU)"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_config(self):
        """Carrega configuração dos especialistas"""
        try:
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.expert_configs = json.load(f)
            else:
                # Configuração padrão
                self.expert_configs = {
                    "sentiment": {
                        "model_name": "microsoft/phi-2",
                        "device": self.device,
                        "max_length": 512,
                        "quantization": self.quantization_enabled
                    },
                    "technical": {
                        "model_name": "microsoft/phi-3-mini-4k-instruct",
                        "device": self.device,
                        "max_length": 1024,
                        "quantization": self.quantization_enabled
                    },
                    "fundamental": {
                        "model_name": "meta-llama/Llama-2-7b-chat-hf",
                        "device": self.device,
                        "max_length": 2048,
                        "quantization": self.quantization_enabled
                    }
                }
            
            logger.info(f"Configuração carregada para {len(self.expert_configs)} especialistas")
            
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            self.expert_configs = {}
    
    def initialize(self, expert_types: Optional[List[str]] = None, **kwargs) -> bool:
        """
        Inicializa o sistema de especialistas
        
        Args:
            expert_types: Lista de tipos de especialistas para inicializar (opcional)
            **kwargs: Parâmetros adicionais
        
        Returns:
            bool: True se inicialização foi bem-sucedida
        """
        try:
            logger.info("Iniciando inicialização do sistema...")
            
            # Descobrir especialistas disponíveis
            available_experts = self._discover_experts()
            logger.info(f"Especialistas descobertos: {list(available_experts.keys())}")
            
            # Usar lista específica se fornecida, senão usar todos descobertos
            experts_to_initialize = expert_types or list(available_experts.keys())
            logger.info(f"Especialistas a inicializar: {experts_to_initialize}")
            
            # Inicializar cada especialista
            success_count = 0
            for expert_name in experts_to_initialize:
                try:
                    # Verificar se o especialista foi descoberto
                    if expert_name in available_experts:
                        if self._initialize_expert(expert_name):
                            success_count += 1
                            logger.info(f"✅ Especialista '{expert_name}' inicializado com sucesso")
                        else:
                            logger.warning(f"⚠️ Falha ao inicializar especialista '{expert_name}'")
                    else:
                        logger.warning(f"⚠️ Especialista '{expert_name}' não foi encontrado")
                except Exception as e:
                    logger.error(f"❌ Erro ao inicializar especialista '{expert_name}': {e}")
            
            self.initialized = True
            
            logger.info(f"Orquestrador inicializado com {success_count}/{len(experts_to_initialize)} especialistas")
            
            # Sistema pode funcionar mesmo com 0 especialistas (modo degradado)
            return True
            
        except Exception as e:
            logger.error(f"Erro crítico na inicialização: {e}")
            return False
    
    def _discover_experts(self) -> Dict[str, str]:
        """
        Descobre especialistas disponíveis
        
        Returns:
            Dict com nome do especialista e caminho
        """
        experts_found = {}
        
        # Verificar pasta experts/
        experts_dir = Path("experts")
        if experts_dir.exists():
            for expert_path in experts_dir.iterdir():
                if expert_path.is_dir():
                    expert_name = expert_path.name
                    if expert_name not in ['__pycache__', '.git']:
                        experts_found[expert_name] = str(expert_path)
        
        # Verificar pasta ezio_experts/
        ezio_experts_dir = Path("ezio_experts")
        if ezio_experts_dir.exists():
            for expert_path in ezio_experts_dir.iterdir():
                if expert_path.is_dir():
                    expert_name = expert_path.name
                    if expert_name not in ['__pycache__', '.git']:
                        experts_found[expert_name] = str(expert_path)
        
        return experts_found
    
    def _initialize_expert(self, expert_name: str) -> bool:
        """
        Inicializa um especialista específico
        
        Args:
            expert_name: Nome do especialista
            
        Returns:
            bool: True se inicialização foi bem-sucedida
        """
        try:
            logger.info(f"Inicializando especialista: {expert_name}")
            
            # Tentar importar dinamicamente o especialista
            expert_instance = self._load_expert_class(expert_name)
            
            if expert_instance:
                # Configurar especialista
                config = self.expert_configs.get(expert_name, {})
                
                # Tentar carregar modelo se necessário
                if hasattr(expert_instance, 'load_model'):
                    model_loaded = self._load_expert_model(expert_instance, config)
                    if not model_loaded:
                        logger.warning(f"Modelo não carregado para {expert_name}, mas continuando...")
                
                # Adicionar aos especialistas ativos
                self.experts[expert_name] = expert_instance
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erro ao inicializar especialista {expert_name}: {e}")
            return False
    
    def _load_expert_class(self, expert_name: str):
        """
        Carrega dinamicamente a classe do especialista
        
        Args:
            expert_name: Nome do especialista
            
        Returns:
            Instância do especialista ou None
        """
        try:
            # Tentar diferentes caminhos de importação
            import_paths = [
                f"experts.{expert_name}.expert",
                f"experts.{expert_name}.{expert_name}_expert", 
                f"ezio_experts.{expert_name}_expert",
                f"core.unified_{expert_name}_expert",
                f"experts.{expert_name}.cache_expert" if expert_name == "cache_expert" else None,
                f"experts.{expert_name}.fallback_data_expert" if expert_name == "fallback_data" else None
            ]
            
            # Filtrar None
            import_paths = [path for path in import_paths if path is not None]
            
            for import_path in import_paths:
                try:
                    module = __import__(import_path, fromlist=[''])
                    
                    # Tentar diferentes nomes de classe
                    class_names = [
                        f"{expert_name.title()}Expert",
                        f"Unified{expert_name.title()}Expert",
                        f"{expert_name.title()}ExpertClass",
                        "Expert",
                        "CacheExpert" if expert_name == "cache_expert" else None,
                        "FallbackDataExpert" if expert_name == "fallback_data" else None,
                        "ClaudeSyncBridge" if expert_name == "claude_sync" else None
                    ]
                    
                    # Filtrar None
                    class_names = [name for name in class_names if name is not None]
                    
                    for class_name in class_names:
                        if hasattr(module, class_name):
                            expert_class = getattr(module, class_name)
                            
                            # Tentar instanciar
                            try:
                                return expert_class(expert_type=expert_name)
                            except TypeError as e:
                                if "system_message" in str(e):
                                    # Tentar sem system_message
                                    try:
                                        return expert_class(expert_type=expert_name)
                                    except:
                                        # Tentar só com expert_type
                                        return expert_class()
                                else:
                                    # Tentar sem parâmetros
                                    try:
                                        return expert_class()
                                    except:
                                        # Tentar com expert_type apenas
                                        return expert_class(expert_type=expert_name)
                                
                except ImportError:
                    continue
            
            logger.warning(f"Não foi possível carregar classe para {expert_name}")
            return None
            
        except Exception as e:
            logger.error(f"Erro ao carregar classe {expert_name}: {e}")
            return None
    
    def _load_expert_model(self, expert_instance, config: Dict) -> bool:
        """
        Carrega modelo para o especialista
        
        Args:
            expert_instance: Instância do especialista
            config: Configuração do especialista
            
        Returns:
            bool: True se modelo foi carregado
        """
        try:
            # Configurar quantização se necessário
            quantization_config = None
            if config.get('quantization', True) and self.device == "cuda":
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                except Exception as e:
                    logger.warning(f"Quantização falhou, usando CPU: {e}")
                    quantization_config = None
            
            # Parâmetros de carregamento
            load_params = {
                'model_name': config.get('model_name', 'microsoft/phi-2'),
                'device': self.device if quantization_config is None else None,
                'quantization_config': quantization_config,
                'max_length': config.get('max_length', 512)
            }
            
            # Chamar método de carregamento
            result = expert_instance.load_model(**load_params)
            
            return result if isinstance(result, bool) else True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            
            # Fallback: tentar carregar sem quantização
            try:
                logger.info("Tentando carregar sem quantização...")
                fallback_params = {
                    'model_name': config.get('model_name', 'microsoft/phi-2'),
                    'device': 'cpu',
                    'quantization_config': None,
                    'max_length': config.get('max_length', 512)
                }
                
                result = expert_instance.load_model(**fallback_params)
                logger.info("✅ Modelo carregado com fallback para CPU")
                return result if isinstance(result, bool) else True
                
            except Exception as e2:
                logger.error(f"Fallback também falhou: {e2}")
                return False
    
    def get_expert_types(self) -> List[str]:
        """
        Retorna lista de tipos de especialistas disponíveis
        
        Returns:
            Lista com nomes dos tipos de especialistas
        """
        return list(self.experts.keys())
    
    def get_experts(self) -> Dict[str, Any]:
        """
        Retorna dicionário com todos os especialistas
        
        Returns:
            Dict com especialistas por nome
        """
        return self.experts.copy()
    
    def get_available_experts(self) -> List[str]:
        """
        Retorna lista de especialistas disponíveis
        
        Returns:
            Lista com nomes dos especialistas
        """
        return list(self.experts.keys())
    
    def query_expert(self, expert_name: str, query: str, **kwargs) -> Optional[Dict]:
        """
        Consulta um especialista específico
        
        Args:
            expert_name: Nome do especialista
            query: Consulta a ser processada
            **kwargs: Parâmetros adicionais
            
        Returns:
            Resultado da consulta ou None
        """
        try:
            if expert_name not in self.experts:
                logger.warning(f"Especialista '{expert_name}' não está disponível")
                return None
            
            expert = self.experts[expert_name]
            
            # Tentar diferentes métodos de consulta
            if hasattr(expert, 'analyze'):
                return expert.analyze(query, **kwargs)
            elif hasattr(expert, 'process'):
                return expert.process(query, **kwargs)
            elif hasattr(expert, 'query'):
                return expert.query(query, **kwargs)
            else:
                logger.warning(f"Especialista '{expert_name}' não tem método de consulta reconhecido")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao consultar especialista {expert_name}: {e}")
            return None
    
    def analyze_with_committee(self, query: str, experts: Optional[List[str]] = None) -> Dict:
        """
        Analisa consulta com comitê de especialistas
        
        Args:
            query: Consulta a ser analisada
            experts: Lista de especialistas (None = todos disponíveis)
            
        Returns:
            Resultado consolidado da análise
        """
        try:
            if not self.experts:
                return {
                    "error": "Nenhum especialista disponível",
                    "status": "no_experts"
                }
            
            # Usar especialistas especificados ou todos disponíveis
            expert_list = experts or list(self.experts.keys())
            results = {}
            
            # Consultar cada especialista
            for expert_name in expert_list:
                if expert_name in self.experts:
                    result = self.query_expert(expert_name, query)
                    if result:
                        results[expert_name] = result
            
            # Consolidar resultados
            return {
                "query": query,
                "experts_consulted": list(results.keys()),
                "results": results,
                "status": "success" if results else "no_results"
            }
            
        except Exception as e:
            logger.error(f"Erro na análise do comitê: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    def get_system_status(self) -> Dict:
        """
        Retorna status do sistema
        
        Returns:
            Dicionário com informações do sistema
        """
        return {
            "initialized": self.initialized,
            "device": self.device,
            "experts_available": len(self.experts),
            "expert_names": list(self.experts.keys()),
            "quantization_enabled": self.quantization_enabled,
            "config_loaded": bool(self.expert_configs)
        }