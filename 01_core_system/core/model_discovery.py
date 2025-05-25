#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ModelDiscovery - Sistema de autodescoberta de modelos LLM locais
----------------------------------------------------------------
Este módulo fornece uma classe para detecção automática de modelos de linguagem locais,
identificação de suas capacidades, e mapeamento para funções de especialistas financeiros.

Componentes principais:
- Escaneamento recursivo de diretórios
- Detecção e validação de modelos
- Identificação de tipos e capacidades de modelos
- Mapeamento inteligente para funções de especialistas
- Resolução de conflitos para múltiplos modelos

Autor: EzioFilho LLMGraph Team
Data: Maio/2025
"""

import os
import sys
import json
import time
import logging
import shutil
import hashlib
import torch
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ModelDiscovery")

# Constantes para detecção de modelos
MODEL_EXTENSIONS = {
    "transformers": [".bin", ".safetensors", ".pt", ".pth"],
    "gguf": [".gguf", ".ggml"],
    "onnx": [".onnx"],
    "pytorch": [".pt", ".pth"],
    "tensorflow": [".pb", ".savedmodel"]
}

MODEL_MARKERS = {
    "transformers": ["config.json", "tokenizer.json", "pytorch_model.bin", "model.safetensors"],
    "gguf": ["*.gguf", "*.ggml"],
    "llamacpp": ["*.gguf", "*.ggml"],
    "onnx": ["*.onnx", "model.onnx"],
    "pytorch": ["*.pt", "*.pth", "model_state_dict.pt"],
    "tensorflow": ["saved_model.pb", "variables"]
}

# Mapeamento de modelos para especialistas financeiros
FINANCIAL_EXPERT_ROLES = [
    "data_analyst",      # Análise de dados financeiros
    "market_predictor",  # Previsão de mercado
    "risk_assessor",     # Avaliação de riscos
    "portfolio_manager", # Gestão de portfólio
    "trend_analyst",     # Análise de tendências
    "news_interpreter",  # Interpretação de notícias financeiras
    "report_generator",  # Geração de relatórios
    "strategy_advisor"   # Conselheiro de estratégias
]

# Nomes de modelos conhecidos e suas características
KNOWN_MODEL_CAPABILITIES = {
    "phi": {
        "capabilities": ["fast", "efficient", "general"],
        "context_length": 2048,
        "suitable_roles": ["data_analyst", "report_generator"]
    },
    "phi-2": {
        "capabilities": ["fast", "efficient", "general"],
        "context_length": 2048,
        "suitable_roles": ["data_analyst", "report_generator"]
    },
    "phi-3": {
        "capabilities": ["powerful", "general", "precise"],
        "context_length": 8192,
        "suitable_roles": ["market_predictor", "trend_analyst", "strategy_advisor"]
    },
    "mistral": {
        "capabilities": ["precise", "general", "long_context"],
        "context_length": 8192,
        "suitable_roles": ["risk_assessor", "portfolio_manager", "market_predictor"]
    },
    "llama": {
        "capabilities": ["general", "versatile"],
        "context_length": 4096,
        "suitable_roles": ["general", "news_interpreter"]
    },
    "llama2": {
        "capabilities": ["general", "versatile"],
        "context_length": 4096,
        "suitable_roles": ["general", "news_interpreter"]
    },
    "llama3": {
        "capabilities": ["powerful", "precise", "general"],
        "context_length": 8192,
        "suitable_roles": ["strategy_advisor", "risk_assessor", "trend_analyst"]
    },
    "codellama": {
        "capabilities": ["code", "technical", "precise"],
        "context_length": 4096,
        "suitable_roles": ["data_analyst"]
    },
    "qwen": {
        "capabilities": ["creative", "general"],
        "context_length": 8192,
        "suitable_roles": ["report_generator", "news_interpreter"]
    },
    "gemma": {
        "capabilities": ["fast", "efficient", "general"],
        "context_length": 4096,
        "suitable_roles": ["data_analyst", "report_generator"]
    }
}

class ModelInfo:
    """Armazena informações sobre um modelo de linguagem detectado."""
    
    def __init__(
        self,
        path: Union[str, Path],
        model_type: str,
        name: Optional[str] = None,
        size_mb: float = 0,
        capabilities: Optional[List[str]] = None,
        context_length: int = 4096,
        expert_roles: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa informações do modelo.
        
        Args:
            path: Caminho para o modelo (arquivo ou diretório)
            model_type: Tipo do modelo (transformers, gguf, etc)
            name: Nome atribuído ao modelo
            size_mb: Tamanho aproximado do modelo em MB
            capabilities: Lista de capacidades do modelo
            context_length: Tamanho máximo de contexto do modelo
            expert_roles: Funções de especialista financeiro adequadas
            parameters: Parâmetros específicos para carregar o modelo
            metadata: Metadados adicionais sobre o modelo
        """
        self.path = Path(path)
        self.model_type = model_type
        self.name = name or self.path.stem
        self.size_mb = size_mb
        self.capabilities = capabilities or ["general"]
        self.context_length = context_length
        self.expert_roles = expert_roles or ["general"]
        self.parameters = parameters or {}
        self.metadata = metadata or {}
        self.hash = self._compute_hash()
        
    def _compute_hash(self) -> str:
        """Gera um hash único para o modelo baseado no caminho."""
        if self.path.is_file():
            # Para arquivos, usa um hash do nome e tamanho
            return hashlib.md5(f"{self.path.name}:{self.size_mb}".encode()).hexdigest()[:8]
        else:
            # Para diretórios, usa um hash do caminho
            return hashlib.md5(str(self.path).encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte as informações do modelo para um dicionário."""
        return {
            "name": self.name,
            "path": str(self.path),
            "model_type": self.model_type,
            "size_mb": self.size_mb,
            "capabilities": self.capabilities,
            "context_length": self.context_length,
            "expert_roles": self.expert_roles,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "hash": self.hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """Cria uma instância a partir de um dicionário."""
        model_info = cls(
            path=data["path"],
            model_type=data["model_type"],
            name=data["name"],
            size_mb=data["size_mb"],
            capabilities=data["capabilities"],
            context_length=data["context_length"],
            expert_roles=data["expert_roles"],
            parameters=data["parameters"],
            metadata=data["metadata"]
        )
        model_info.hash = data.get("hash", model_info.hash)
        return model_info


class ModelDiscovery:
    """
    Sistema de autodescoberta de modelos de linguagem locais.
    
    Esta classe fornece funcionalidades para:
    1. Escanear diretórios recursivamente para encontrar modelos
    2. Detectar e validar modelos encontrados
    3. Identificar suas capacidades e tipos
    4. Mapear modelos para funções de especialistas financeiros
    5. Gerenciar conflitos entre múltiplos modelos
    """
    
    def __init__(
        self,
        base_dirs: Optional[List[Union[str, Path]]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        max_model_size_gb: float = 100.0,
        auto_scan: bool = True,
        gpu_ids: Optional[List[int]] = None
    ):
        """
        Inicializa o sistema de descoberta de modelos.
        
        Args:
            base_dirs: Lista de diretórios base para procurar modelos
            cache_dir: Diretório para cache de configurações
            max_model_size_gb: Tamanho máximo de modelo em GB
            auto_scan: Se deve escanear automáticamente na inicialização
            gpu_ids: Lista de IDs de GPUs disponíveis para uso
        """
        # Diretórios base para procurar modelos
        self.base_dirs = []
        if base_dirs:
            for dir_path in base_dirs:
                path = Path(dir_path).expanduser().absolute()
                if path.exists():
                    self.base_dirs.append(path)
        
        # Adiciona diretórios padrões
        self._add_default_directories()
        
        # Configurações
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "eziofilho" / "models"
        self.max_model_size = max_model_size_gb * 1024  # Conversão para MB
        self.scan_complete = False
        self.gpu_ids = gpu_ids or []
        
        # Armazenamento para modelos encontrados
        self.discovered_models: List[ModelInfo] = []
        self.model_map: Dict[str, ModelInfo] = {}  # Mapeamento nome -> modelo
        self.models = {}  # Compatibilidade com a interface
        
        # Log de GPUs disponíveis
        if self.gpu_ids:
            logger.info(f"GPUs disponíveis para modelos: {self.gpu_ids}")
        else:
            logger.info("Nenhuma GPU especificada, usando CPU para todos os modelos")
        
        # Inicializa cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "model_discovery_cache.json"
        
        # Escaneia automaticamente se solicitado
        if auto_scan:
            self.scan_all()
    
    def _add_default_directories(self):
        """Adiciona diretórios padrões para busca de modelos."""
        # Diretórios padrões a serem verificados
        default_dirs = [
            # Diretórios do projeto
            Path.cwd() / "models",
            Path.cwd() / "models_hf",
            Path.cwd().parent / "models",
            Path.cwd().parent / "models_hf",
            
            # Cache do Hugging Face
            Path.home() / ".cache" / "huggingface" / "hub",
            
            # Diretórios de modelos comuns em diferentes sistemas
            Path.home() / "models",
            Path.home() / "Documents" / "models",
            Path.home() / "AI" / "models",
            Path.home() / "Downloads" / "models",
            
            # Diretórios de aplicativos comuns
            Path("/opt/models"),
            Path("C:/models")
        ]
        
        # Adiciona apenas diretórios que existem
        for dir_path in default_dirs:
            if dir_path.exists() and dir_path not in self.base_dirs:
                self.base_dirs.append(dir_path)
    
    def scan_all(self) -> List[ModelInfo]:
        """
        Escaneia todos os diretórios base em busca de modelos.
        
        Returns:
            Lista de ModelInfo para todos os modelos encontrados
        """
        start_time = time.time()
        logger.info(f"Iniciando escaneamento de modelos em {len(self.base_dirs)} diretórios")
        
        # Tenta carregar do cache primeiro
        cached_models = self._load_from_cache()
        if cached_models:
            self.discovered_models = cached_models
            self._update_model_map()
            logger.info(f"Carregados {len(cached_models)} modelos do cache")
        
        # Escaneia cada diretório base
        for base_dir in self.base_dirs:
            try:
                logger.info(f"Escaneando: {base_dir}")
                self._scan_directory(base_dir)
            except Exception as e:
                logger.error(f"Erro ao escanear {base_dir}: {e}")
        
        # Filtra duplicatas
        self._remove_duplicates()
        
        # Atualiza cache
        self._save_to_cache()
        
        # Marca escaneamento como completo
        self.scan_complete = True
        
        elapsed = time.time() - start_time
        count = len(self.discovered_models)
        logger.info(f"Escaneamento concluído em {elapsed:.2f}s. Encontrados {count} modelos.")
        
        return self.discovered_models
    
    def _scan_directory(self, directory: Path, max_depth: int = 4, current_depth: int = 0):
        """
        Escaneia um diretório em busca de modelos recursivamente.
        
        Args:
            directory: Diretório a ser escaneado
            max_depth: Profundidade máxima de recursão
            current_depth: Profundidade atual da recursão
        """
        # Evita recursão excessiva
        if current_depth > max_depth:
            return
        
        # Ignora diretórios ocultos e especiais
        if directory.name.startswith(".") or directory.name.startswith("__"):
            return
        
        try:
            # Verifica se este diretório contém um modelo Transformers
            if self._is_transformers_model_dir(directory):
                model_info = self._create_model_info(directory, "transformers")
                if model_info:
                    self._add_model(model_info)
                return  # Não precisa continuar a recursão dentro de um modelo
            
            # Para cada item no diretório
            for item in directory.iterdir():
                # Arquivos GGUF/GGML
                if item.is_file() and any(item.name.endswith(ext) for ext in MODEL_EXTENSIONS["gguf"]):
                    model_info = self._create_model_info(item, "gguf")
                    if model_info:
                        self._add_model(model_info)
                
                # Arquivos ONNX
                elif item.is_file() and item.name.endswith(".onnx"):
                    model_info = self._create_model_info(item, "onnx")
                    if model_info:
                        self._add_model(model_info)
                
                # Arquivos PyTorch
                elif item.is_file() and any(item.name.endswith(ext) for ext in MODEL_EXTENSIONS["pytorch"]):
                    # Verifica se é um arquivo de modelo PyTorch independente
                    if "model" in item.name.lower() or "weight" in item.name.lower():
                        model_info = self._create_model_info(item, "pytorch")
                        if model_info:
                            self._add_model(model_info)
                
                # Recursão para subdiretórios
                elif item.is_dir():
                    self._scan_directory(item, max_depth, current_depth + 1)
        
        except Exception as e:
            logger.error(f"Erro ao processar {directory}: {e}")
    
    def _is_transformers_model_dir(self, directory: Path) -> bool:
        """
        Verifica se um diretório contém um modelo Transformers.
        
        Args:
            directory: Diretório a ser verificado
            
        Returns:
            True se for um diretório de modelo Transformers
        """
        # Verificação rápida para arquivos comuns
        config_json = directory / "config.json"
        if config_json.exists():
            # Verifica existência de outros arquivos comuns
            model_files = [
                directory / "pytorch_model.bin",
                directory / "model.safetensors",
                directory / "tf_model.h5",
                directory / "model.onnx"
            ]
            
            tokenizer_files = [
                directory / "tokenizer.json",
                directory / "tokenizer_config.json",
                directory / "special_tokens_map.json",
                directory / "vocab.json"
            ]
            
            has_model = any(f.exists() for f in model_files)
            has_tokenizer = any(f.exists() for f in tokenizer_files)
            
            return has_model or has_tokenizer
        
        return False
    
    def _get_model_size(self, path: Path) -> float:
        """
        Calcula o tamanho aproximado de um modelo em MB.
        
        Args:
            path: Caminho para o arquivo ou diretório do modelo
            
        Returns:
            Tamanho em megabytes
        """
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)  # Bytes para MB
        
        # Para diretórios, soma o tamanho de todos os arquivos relevantes
        total_size = 0
        for ext in [".bin", ".safetensors", ".gguf", ".ggml", ".onnx", ".pt", ".pth"]:
            for file_path in path.glob(f"**/*{ext}"):
                total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)  # Bytes para MB
    
    def _create_model_info(self, path: Path, model_type: str) -> Optional[ModelInfo]:
        """
        Cria um objeto ModelInfo a partir de um caminho detectado.
        
        Args:
            path: Caminho para o modelo (arquivo ou diretório)
            model_type: Tipo do modelo (transformers, gguf, etc)
            
        Returns:
            ModelInfo ou None se o modelo for inválido
        """
        try:
            # Obtém tamanho do modelo
            size_mb = self._get_model_size(path)
            
            # Verifica limite de tamanho
            if size_mb > self.max_model_size:
                logger.warning(f"Modelo muito grande ignorado: {path} ({size_mb:.2f} MB)")
                return None
            
            # Determina nome base do modelo
            name = self._extract_model_name(path)
            
            # Infere capacidades e tamanho de contexto
            capabilities, context_length, expert_roles = self._infer_model_capabilities(name, path, model_type)
            
            # Cria objeto ModelInfo
            return ModelInfo(
                path=path,
                model_type=model_type,
                name=name,
                size_mb=size_mb,
                capabilities=capabilities,
                context_length=context_length,
                expert_roles=expert_roles,
                parameters=self._generate_model_parameters(name, model_type, size_mb),
                metadata={
                    "detection_method": "auto",
                    "detected_time": time.time(),
                    "original_filename": path.name
                }
            )
        
        except Exception as e:
            logger.error(f"Erro ao criar ModelInfo para {path}: {e}")
            return None
    
    def _extract_model_name(self, path: Path) -> str:
        """
        Extrai um nome significativo do caminho do modelo.
        
        Args:
            path: Caminho para o modelo
            
        Returns:
            Nome do modelo
        """
        # Para arquivos, usa o nome base sem extensão
        if path.is_file():
            name = path.stem.lower()
            
            # Remove sufixos de quantização comuns
            name = re.sub(r'-q[2-8]_[kms]', '', name)
            name = re.sub(r'-[0-9]bit', '', name)
            name = re.sub(r'_quantized', '', name)
            
            # Remove prefixos de publishers comuns
            name = re.sub(r'^thebloke-', '', name)
            name = re.sub(r'^microsoft-', '', name)
            name = re.sub(r'^meta-', '', name)
            
            return name
        
        # Para diretórios, tenta extrair de config.json
        config_path = path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if "_name_or_path" in config:
                        return config["_name_or_path"].split("/")[-1]
            except Exception:
                pass
        
        # Fallback para o nome do diretório
        return path.name.lower()
    
    def _infer_model_capabilities(
        self, 
        name: str, 
        path: Path, 
        model_type: str
    ) -> Tuple[List[str], int, List[str]]:
        """
        Infere as capacidades e tamanho de contexto de um modelo.
        
        Args:
            name: Nome do modelo
            path: Caminho para o modelo
            model_type: Tipo do modelo
            
        Returns:
            Tupla com (capacidades, tamanho de contexto, funções de especialista)
        """
        # Padrões
        capabilities = ["general"]
        context_length = 4096  # Valor padrão
        expert_roles = ["general"]
        
        # Verifica modelos conhecidos
        for known_name, info in KNOWN_MODEL_CAPABILITIES.items():
            if known_name in name.lower():
                capabilities = info["capabilities"]
                context_length = info["context_length"]
                expert_roles = info["suitable_roles"]
                break
        
        # Identificação por tipos específicos
        if "codellama" in name or "starcoder" in name or "codebert" in name:
            capabilities = ["code", "technical", "precise"]
            expert_roles = ["data_analyst"]
        
        elif "mistral" in name:
            # Específico para diferentes versões do Mistral
            if "instruct" in name or "chat" in name:
                capabilities = ["precise", "chat", "general"]
                expert_roles = ["market_predictor", "risk_assessor", "strategy_advisor"]
            else:
                capabilities = ["precise", "general"]
                expert_roles = ["market_predictor", "risk_assessor"]
            
            # Tamanho de contexto baseado na versão
            if "7b-" in name:
                context_length = 8192
            elif "8x7b" in name or "large" in name:
                context_length = 32768  # Mistral Large
            
        elif "llama" in name:
            # Específico para diferentes versões do Llama
            if "llama-3" in name or "llama3" in name:
                capabilities = ["powerful", "precise", "general"]
                context_length = 8192
                expert_roles = ["strategy_advisor", "market_predictor", "risk_assessor"]
            elif "llama-2" in name or "llama2" in name:
                capabilities = ["general", "versatile"]
                context_length = 4096
                expert_roles = ["general", "news_interpreter"]
            
            # Ajustes baseados em indicadores no nome
            if "70b" in name:
                context_length = 4096
                capabilities.append("powerful")
            elif "13b" in name:
                context_length = 4096
                capabilities.append("balanced")
            elif "7b" in name:
                context_length = 4096
                capabilities.append("efficient")
        
        elif "phi" in name:
            if "phi-3" in name:
                capabilities = ["powerful", "general", "precise"]
                context_length = 8192
                expert_roles = ["market_predictor", "trend_analyst", "strategy_advisor"]
            else:
                capabilities = ["fast", "efficient", "general"]
                context_length = 2048
                expert_roles = ["data_analyst", "report_generator"]
        
        # Infere pelo tamanho do modelo
        if model_type == "gguf":
            if path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                
                # Adiciona capacidades baseadas no tamanho
                if size_mb < 2000:  # < 2GB
                    capabilities.append("fast")
                    capabilities.append("efficient")
                elif size_mb > 10000:  # > 10GB
                    capabilities.append("powerful")
                    expert_roles.append("strategy_advisor")
                
                # Infere quantização pelo nome
                if "q4_k_m" in name or "q4_0" in name:
                    capabilities.append("efficient")
                elif "q5_k_m" in name or "q5_0" in name:
                    capabilities.append("balanced")
                elif "q8_0" in name or "f16" in name:
                    capabilities.append("precise")
        
        # Remove duplicatas
        capabilities = list(dict.fromkeys(capabilities))
        expert_roles = list(dict.fromkeys(expert_roles))
        
        return capabilities, context_length, expert_roles
    
    def _generate_model_parameters(self, name: str, model_type: str, size_mb: float) -> Dict[str, Any]:
        """
        Gera parâmetros ótimos para o modelo baseados no nome e tipo.
        
        Args:
            name: Nome do modelo
            model_type: Tipo do modelo
            size_mb: Tamanho do modelo em MB
            
        Returns:
            Dicionário de parâmetros
        """
        params = {}
        
        # Configurações comuns
        params["use_gpu"] = True
        params["trust_remote_code"] = True
        
        # Configurações específicas para GGUF/GGML
        if model_type == "gguf":
            # Otimização baseada no tamanho do modelo
            if size_mb < 2000:  # < 2GB
                params["n_threads"] = min(4, os.cpu_count() or 4)
                params["n_batch"] = 512
            elif size_mb < 5000:  # < 5GB
                params["n_threads"] = min(6, os.cpu_count() or 4)
                params["n_batch"] = 1024
            else:  # >= 5GB
                params["n_threads"] = min(8, os.cpu_count() or 4)
                params["n_batch"] = 2048
            
            # Configuração de GPU para modelos maiores
            if size_mb > 3000 and self._has_gpu():
                gpu_vram = self._get_gpu_vram_mb()
                if gpu_vram > 0:
                    # Determina quanta VRAM pode ser usada (70%)
                    usable_vram = int(gpu_vram * 0.7)
                    
                    # Se há VRAM suficiente
                    if usable_vram > size_mb:
                        params["n_gpu_layers"] = -1  # Todas as camadas na GPU
                    else:
                        # Carrega camadas proporcional à VRAM disponível
                        params["n_gpu_layers"] = max(1, int((usable_vram / size_mb) * 40))
        
        # Configurações específicas para Transformers
        elif model_type == "transformers":
            # Device mapping baseado no tamanho
            if self._has_gpu():
                if size_mb > 10000:  # > 10GB
                    params["device_map"] = "auto"
                else:
                    params["device_map"] = "cuda:0"
            else:
                params["device_map"] = "cpu"
            
            # Otimizações de memória
            if size_mb > 5000:  # > 5GB
                params["low_cpu_mem_usage"] = True
                params["torch_dtype"] = "auto"
            
            # Configurações para modelos específicos
            if "phi-3" in name:
                params["torch_dtype"] = "bfloat16"
            elif "mistral" in name or "llama" in name:
                params["torch_dtype"] = "float16"
        
        return params
    
    def _has_gpu(self) -> bool:
        """Verifica se há uma GPU disponível."""
        try:
            return torch.cuda.is_available()
        except Exception:
            return False
    
    def _get_gpu_vram_mb(self) -> int:
        """Obtém a memória VRAM disponível na GPU em MB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            return 0
        except Exception:
            return 0
    
    def _add_model(self, model_info: ModelInfo):
        """
        Adiciona um modelo à lista de modelos descobertos.
        
        Args:
            model_info: Informações do modelo a ser adicionado
        """
        # Verifica se o modelo já existe (pelo hash)
        for existing in self.discovered_models:
            if existing.hash == model_info.hash:
                return
        
        # Adiciona à lista
        self.discovered_models.append(model_info)
        
        # Atualiza o mapa de modelos
        self._update_model_map()
    
    def _update_model_map(self):
        """Atualiza o mapeamento de nome para modelo."""
        self.model_map = {model.name: model for model in self.discovered_models}
    
    def _remove_duplicates(self):
        """Remove modelos duplicados da lista de descobertos."""
        # Usa hash para detectar duplicatas
        unique_models = {}
        for model in self.discovered_models:
            # Se houver colisão de hash, mantém o maior/mais recente
            if model.hash in unique_models:
                existing = unique_models[model.hash]
                # Prefere modelos maiores/mais recentes
                if model.size_mb > existing.size_mb:
                    unique_models[model.hash] = model
            else:
                unique_models[model.hash] = model
        
        # Atualiza lista
        self.discovered_models = list(unique_models.values())
        self._update_model_map()
    
    def _save_to_cache(self):
        """Salva modelos descobertos no cache."""
        try:
            models_data = [model.to_dict() for model in self.discovered_models]
            cache_data = {
                "timestamp": time.time(),
                "models": models_data
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Cache salvo: {len(models_data)} modelos em {self.cache_file}")
        except Exception as e:
            logger.error(f"Erro ao salvar cache: {e}")
    
    def _load_from_cache(self) -> List[ModelInfo]:
        """
        Carrega modelos descobertos do cache.
        
        Returns:
            Lista de ModelInfo carregados do cache
        """
        try:
            if not self.cache_file.exists():
                return []
            
            # Verifica idade do cache (máximo 7 dias)
            max_age = 7 * 24 * 60 * 60  # 7 dias em segundos
            cache_age = time.time() - self.cache_file.stat().st_mtime
            if cache_age > max_age:
                logger.info(f"Cache expirado ({cache_age / 86400:.1f} dias). Reescaneando...")
                return []
            
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            models = []
            for model_data in cache_data.get("models", []):
                model_path = Path(model_data["path"])
                
                # Verifica se o modelo ainda existe
                if model_path.exists():
                    models.append(ModelInfo.from_dict(model_data))
                else:
                    logger.debug(f"Modelo em cache não encontrado: {model_path}")
            
            return models
        
        except Exception as e:
            logger.error(f"Erro ao carregar cache: {e}")
            return []
    
    def get_best_model_for_role(self, role: str) -> Optional[ModelInfo]:
        """
        Encontra o melhor modelo para um papel de especialista específico.
        
        Args:
            role: Papel de especialista financeiro
            
        Returns:
            ModelInfo do melhor modelo ou None
        """
        candidates = []
        
        # Coleta modelos adequados para o papel
        for model in self.discovered_models:
            if role in model.expert_roles:
                candidates.append(model)
        
        if not candidates:
            # Sem correspondências exatas, procure por "general"
            for model in self.discovered_models:
                if "general" in model.expert_roles:
                    candidates.append(model)
        
        if not candidates:
            return None
        
        # Pontuação para cada modelo candidato
        scores = []
        for model in candidates:
            score = 0
            
            # Pontuação por ter o papel exato
            if role in model.expert_roles:
                score += 50
            
            # Pontuação por capacidades
            if "precise" in model.capabilities:
                score += 20
            if "powerful" in model.capabilities:
                score += 15
            
            # Tamanho do contexto
            score += min(30, model.context_length / 1000)
            
            # Penalidade por modelos muito grandes em hardware limitado
            if model.size_mb > 5000 and not self._has_gpu():
                score -= 20
            
            scores.append((model, score))
        
        # Retorna o modelo com maior pontuação
        return max(scores, key=lambda x: x[1])[0]
    
    def get_models_by_capability(self, capability: str) -> List[ModelInfo]:
        """
        Filtra modelos por uma capacidade específica.
        
        Args:
            capability: Capacidade desejada
            
        Returns:
            Lista de modelos com a capacidade
        """
        return [model for model in self.discovered_models 
                if capability in model.capabilities]
    
    def get_models_by_type(self, model_type: str) -> List[ModelInfo]:
        """
        Filtra modelos por tipo.
        
        Args:
            model_type: Tipo de modelo (transformers, gguf, etc)
            
        Returns:
            Lista de modelos do tipo especificado
        """
        return [model for model in self.discovered_models 
                if model.model_type == model_type]
    
    def get_fastest_model(self) -> Optional[ModelInfo]:
        """
        Encontra o modelo mais rápido disponível.
        
        Returns:
            ModelInfo do modelo mais rápido ou None
        """
        # Filtra modelos com capacidade "fast"
        fast_models = self.get_models_by_capability("fast")
        if fast_models:
            # Ordena por tamanho (menor = mais rápido geralmente)
            return min(fast_models, key=lambda m: m.size_mb)
        
        # Fallback: O menor modelo disponível
        if self.discovered_models:
            return min(self.discovered_models, key=lambda m: m.size_mb)
        
        return None
    
    def generate_config_file(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Gera um arquivo de configuração JSON com os modelos descobertos.
        
        Args:
            output_path: Caminho para o arquivo de saída
            
        Returns:
            Caminho do arquivo de configuração gerado
        """
        # Cria configuração
        config = {
            "generation_time": time.time(),
            "models": [model.to_dict() for model in self.discovered_models],
            "roles": {}
        }
        
        # Mapeia os melhores modelos para cada papel
        for role in FINANCIAL_EXPERT_ROLES:
            best_model = self.get_best_model_for_role(role)
            if best_model:
                config["roles"][role] = best_model.name
        
        # Determina modelo padrão (mais rápido e geral)
        fastest = self.get_fastest_model()
        if fastest:
            config["default_model"] = fastest.name
        elif self.discovered_models:
            config["default_model"] = self.discovered_models[0].name
        
        # Define caminho de saída
        if not output_path:
            output_path = Path.cwd() / "model_discovery_config.json"
        else:
            output_path = Path(output_path)
        
        # Salva arquivo
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuração gerada: {output_path}")
        return str(output_path)
    
    def register_model_manually(
        self, 
        path: Union[str, Path], 
        model_type: str, 
        name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        context_length: int = 4096,
        expert_roles: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """
        Registra um modelo manualmente.
        
        Args:
            path: Caminho para o modelo
            model_type: Tipo do modelo
            name: Nome do modelo
            capabilities: Lista de capacidades
            context_length: Tamanho do contexto
            expert_roles: Papéis de especialista
            parameters: Parâmetros adicionais
            
        Returns:
            ModelInfo do modelo registrado
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Caminho do modelo não existe: {path}")
        
        # Obter nome se não fornecido
        if not name:
            name = self._extract_model_name(path)
        
        # Obter tamanho
        size_mb = self._get_model_size(path)
        
        # Criar ModelInfo
        model_info = ModelInfo(
            path=path,
            model_type=model_type,
            name=name,
            size_mb=size_mb,
            capabilities=capabilities or ["general"],
            context_length=context_length,
            expert_roles=expert_roles or ["general"],
            parameters=parameters or self._generate_model_parameters(name, model_type, size_mb),
            metadata={
                "detection_method": "manual",
                "registration_time": time.time()
            }
        )
        
        # Adicionar à lista
        self._add_model(model_info)
        
        # Atualizar cache
        self._save_to_cache()
        
        return model_info

    def clear_cache(self):
        """Limpa o cache de modelos descobertos."""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
            logger.info("Cache de modelos limpo")
        except Exception as e:
            logger.error(f"Erro ao limpar cache: {e}")


# ======= Utilitários de uso rápido =======

def discover_local_models() -> List[Dict[str, Any]]:
    """
    Função utilitária para descobrir rapidamente modelos locais.
    
    Returns:
        Lista de dicionários com informações dos modelos
    """
    discovery = ModelDiscovery(auto_scan=True)
    return [model.to_dict() for model in discovery.discovered_models]

def get_best_model_for_financial_role(role: str) -> Dict[str, Any]:
    """
    Encontra o melhor modelo para um papel financeiro.
    
    Args:
        role: Papel de especialista financeiro
        
    Returns:
        Dicionário com informações do modelo ou vazio se não encontrado
    """
    discovery = ModelDiscovery(auto_scan=True)
    model = discovery.get_best_model_for_role(role)
    return model.to_dict() if model else {}

def generate_models_config(output_path: Optional[str] = None) -> str:
    """
    Gera um arquivo de configuração para todos os modelos locais.
    
    Args:
        output_path: Caminho para o arquivo de saída
        
    Returns:
        Caminho do arquivo de configuração gerado
    """
    discovery = ModelDiscovery(auto_scan=True)
    return discovery.generate_config_file(output_path)


# ======= Uso como script =======

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema de descoberta de modelos LLM")
    parser.add_argument("--scan", action="store_true", help="Escanear por modelos locais")
    parser.add_argument("--config", "-c", help="Gerar arquivo de configuração")
    parser.add_argument("--dir", "-d", action="append", help="Diretório adicional para escanear")
    parser.add_argument("--clear-cache", action="store_true", help="Limpar cache de modelos")
    parser.add_argument("--verbose", "-v", action="store_true", help="Modo verboso")
    
    args = parser.parse_args()
    
    # Configura logging
    if args.verbose:
        logging.getLogger("ModelDiscovery").setLevel(logging.DEBUG)
    
    # Inicializa descoberta
    discovery = ModelDiscovery(base_dirs=args.dir, auto_scan=False)
    
    # Limpa cache se solicitado
    if args.clear_cache:
        discovery.clear_cache()
    
    # Escaneia modelos
    if args.scan or args.config:
        discovery.scan_all()
        
        # Exibe resultados
        print(f"\nModelos encontrados: {len(discovery.discovered_models)}")
        for i, model in enumerate(discovery.discovered_models, 1):
            print(f"{i}. {model.name} ({model.model_type}): {model.size_mb:.1f}MB")
            print(f"   Capacidades: {', '.join(model.capabilities)}")
            print(f"   Contexto: {model.context_length} tokens")
            print(f"   Especialista: {', '.join(model.expert_roles)}")
    
    # Gera configuração
    if args.config:
        config_path = discovery.generate_config_file(args.config)
        print(f"\nConfiguração gerada: {config_path}")
