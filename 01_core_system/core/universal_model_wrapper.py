#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UniversalModelWrapper - Interface unificada para modelos de linguagem locais
---------------------------------------------------------------------------
Fornece uma camada de abstração para diferentes tipos de modelos de linguagem,
com recursos avançados como:
- Interface unificada para diferentes modelos (Transformers, GGUF, ONNX)
- Detecção automática de hardware e otimização
- Carregamento preguiçoso (lazy loading) para gerenciamento de memória
- Mecanismos de fallback em caso de falha

Autor: EzioFilho LLMGraph Team
Data: Maio/2025
"""

import gc
import json
import logging
import os
import sys
import time
import re
import warnings
import threading
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable, TypeVar

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("UniversalModelWrapper")

# Decoradores
T = TypeVar('T')

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
         exceptions: Tuple[Exception, ...] = (Exception,)) -> Callable:
    """
    Decorador para retry com backoff exponencial.
    
    Args:
        max_attempts: Número máximo de tentativas
        delay: Atraso inicial entre tentativas (segundos)
        backoff: Fator multiplicador para o atraso entre tentativas
        exceptions: Tupla de exceções a serem capturadas
        
    Returns:
        Decorador para função
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            mtries, mdelay = max_attempts, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Tentativa falhou: {str(e)}, tentando novamente em {mdelay}s...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator

def lazy_import(module_name: str) -> Optional[Any]:
    """
    Importa um módulo apenas quando necessário.
    
    Args:
        module_name: Nome do módulo a ser importado
        
    Returns:
        Módulo importado ou None se não disponível
    """
    try:
        return __import__(module_name)
    except ImportError:
        return None

# Constantes
MODEL_TYPES = [
    "transformers",  # Modelos HuggingFace Transformers
    "gguf",          # GGUF (sucessor do GGML)
    "onnx",          # ONNX Runtime
    "pytorch",       # PyTorch nativo
    "tensorflow",    # TensorFlow nativo
    "ctransformers"  # CTransformers
]

# Capacidades dos modelos
CAPABILITIES = [
    "general",       # Capacidade geral
    "fast",          # Otimizado para velocidade
    "precise",       # Alta precisão
    "creative",      # Geração criativa
    "code",          # Geração de código
    "chat",          # Formato de chat
    "math",          # Raciocínio matemático
    "finance",       # Análise financeira
    "long_context"   # Suporte a contexto longo
]

class HardwareDetector:
    """Detector de hardware para otimização automática de modelos."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Implementação de singleton."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(HardwareDetector, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Inicializa o detector de hardware."""
        if not self._initialized:
            self._detect_hardware()
            self._initialized = True
    
    def _detect_hardware(self):
        """Detecta e armazena informações sobre o hardware disponível."""
        # CPU
        self.cpu_count = os.cpu_count() or 4
        
        # Memoria RAM
        try:
            import psutil
            self.total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
            self.free_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        except ImportError:
            # Fallback se psutil não estiver disponível
            self.total_memory_mb = 8192  # 8GB assumidos
            self.free_memory_mb = 4096   # 4GB assumidos
        
        # GPU - CUDA (NVIDIA)
        self.has_cuda = False
        self.cuda_device_name = ""
        self.cuda_device_count = 0
        self.cuda_device_memory_mb = 0
        
        try:
            if torch.cuda.is_available():
                self.has_cuda = True
                self.cuda_device_count = torch.cuda.device_count()
                if self.cuda_device_count > 0:
                    self.cuda_device_name = torch.cuda.get_device_name(0)
                    self.cuda_device_memory_mb = (
                        torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    )
        except Exception as e:
            logger.warning(f"Erro ao detectar CUDA: {e}")
        
        # GPU - Metal (Apple Silicon)
        self.has_mps = False
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.has_mps = True
        except Exception as e:
            logger.warning(f"Erro ao detectar MPS: {e}")
        
        # ROCm (AMD)
        self.has_rocm = False
        try:
            # Tenta detectar ROCm através de variáveis de ambiente
            if "ROCM_PATH" in os.environ or "HIP_PATH" in os.environ:
                import torch.utils.cpp_extension
                if hasattr(torch.utils.cpp_extension, "ROCM_HOME"):
                    self.has_rocm = True
        except Exception:
            pass
        
        # Logging das informações
        logger.info(f"CPU: {self.cpu_count} cores")
        logger.info(f"RAM: {self.total_memory_mb:.0f}MB total, {self.free_memory_mb:.0f}MB livre")
        
        if self.has_cuda:
            logger.info(f"GPU: CUDA disponível - {self.cuda_device_name} " +
                      f"({self.cuda_device_memory_mb:.0f}MB)")
        elif self.has_mps:
            logger.info("GPU: Apple Metal (MPS) disponível")
        elif self.has_rocm:
            logger.info("GPU: AMD ROCm disponível")
        else:
            logger.info("GPU: Não detectada")
    
    def get_optimal_device(self) -> str:
        """
        Determina o melhor dispositivo para execução de modelos.
        
        Returns:
            String representando o dispositivo ("cuda", "mps", "cpu", etc)
        """
        if self.has_cuda:
            return "cuda"
        elif self.has_mps:
            return "mps"
        elif self.has_rocm:
            return "rocm"
        else:
            return "cpu"
    
    def get_optimal_threads(self, model_size_mb: float = 0) -> int:
        """
        Determina o número ótimo de threads para o modelo.
        
        Args:
            model_size_mb: Tamanho do modelo em MB
            
        Returns:
            Número ótimo de threads
        """
        # Se modelo for pequeno, use menos threads
        if model_size_mb < 1000:  # < 1GB
            return min(4, max(2, self.cpu_count // 2))
        # Se modelo for grande, use mais threads
        elif model_size_mb > 5000:  # > 5GB
            return min(12, max(4, self.cpu_count))
        # Valor padrão para modelos médios
        else:
            return min(8, max(4, self.cpu_count // 2))
    
    def can_fit_on_gpu(self, model_size_mb: float) -> bool:
        """
        Verifica se um modelo cabe na GPU.
        
        Args:
            model_size_mb: Tamanho do modelo em MB
            
        Returns:
            True se o modelo provavelmente cabe na GPU
        """
        if not (self.has_cuda or self.has_mps):
            return False
        
        # Espaço necessário (modelo + buffer de operações)
        # Usa fator de 1.5x para considerar overhead de operações
        required_mb = model_size_mb * 1.5
        
        # Verifica se há memória suficiente
        if self.has_cuda:
            return required_mb <= self.cuda_device_memory_mb * 0.85  # 85% da VRAM
        
        # Para Apple Silicon (sem informação precisa de memória)
        # Assume 70% da RAM disponível para GPU integrada
        if self.has_mps:
            return required_mb <= self.free_memory_mb * 0.7
        
        return False
    
    def refresh(self):
        """Atualiza as informações de hardware."""
        self._detect_hardware()


class ModelLoader:
    """Carregador de modelos compatível com diferentes formatos."""
    
    def __init__(self):
        """Inicializa o carregador de modelos."""
        self.hardware = HardwareDetector()
        self.loaded_models = {}
        self.locks = {}
        self.references = {}
    
    def _is_transformers_available(self) -> bool:
        """Verifica se a biblioteca Transformers está disponível."""
        try:
            import transformers
            return True
        except ImportError:
            return False
    
    def _is_gguf_available(self) -> bool:
        """Verifica se a biblioteca llama-cpp-python está disponível."""
        try:
            import llama_cpp
            return True
        except ImportError:
            return False
    
    def _is_onnx_available(self) -> bool:
        """Verifica se o ONNX Runtime está disponível."""
        try:
            import onnxruntime
            return True
        except ImportError:
            return False
    
    def _is_ctransformers_available(self) -> bool:
        """Verifica se a biblioteca CTransformers está disponível."""
        try:
            import ctransformers
            return True
        except ImportError:
            return False
    
    def _load_transformers_model(self, path: Path, parameters: Dict[str, Any]):
        """
        Carrega um modelo Transformers.
        
        Args:
            path: Caminho para o modelo
            parameters: Parâmetros adicionais para carregamento
            
        Returns:
            Tuple com (modelo, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # Configura device e tipo apropriados
        device = parameters.get("device_map", self.hardware.get_optimal_device())
        dtype = parameters.get("torch_dtype", "auto")
        
        # Parâmetros padrão, podem ser sobrescritos pelos fornecidos
        default_params = {
            "device_map": device,
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "use_safetensors": True,
            "low_cpu_mem_usage": True
        }
        
        # Mesclar com parâmetros fornecidos, dando prioridade aos fornecidos
        load_params = {**default_params, **parameters}
        
        # Carrega tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                path,
                trust_remote_code=load_params.get("trust_remote_code", True),
                use_fast=True
            )
        except Exception as e:
            logger.warning(f"Erro ao carregar tokenizer específico: {e}")
            # Fallback para tokenizer básico
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/phi-2",  # Tokenizer genérico como fallback
                    trust_remote_code=True
                )
            except Exception:
                raise ValueError("Falha ao carregar tokenizer, mesmo com fallback")
        
        # Garante token de padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Carrega modelo
        try:
            # Configura dtype específico se necessário
            if dtype == "auto":
                # Seleciona dtype baseado no dispositivo e nome do modelo
                if "cuda" in str(device) or device == "mps":
                    model_name = str(path).lower()
                    if "phi-3" in model_name:
                        load_params["torch_dtype"] = torch.bfloat16
                    else:
                        load_params["torch_dtype"] = torch.float16
            
            model = AutoModelForCausalLM.from_pretrained(
                path,
                **load_params
            )
            
            # Marca o modelo para verificar vazamentos
            model._wrapper_id = id(model)
            
            return model, tokenizer
        
        except Exception as e:
            # Se falhar com configuração automática, tenta abordagem mais simples
            logger.warning(f"Erro ao carregar modelo otimizado: {e}. Tentando fallback...")
            
            # Fallback para carregamento mais simples
            fallback_params = {
                "device_map": "auto",
                "trust_remote_code": True
            }
            
            model = AutoModelForCausalLM.from_pretrained(
                path,
                **fallback_params
            )
            
            return model, tokenizer
    
    def _load_gguf_model(self, path: Path, parameters: Dict[str, Any]):
        """
        Carrega um modelo GGUF/GGML usando llama-cpp-python.
        
        Args:
            path: Caminho para o arquivo do modelo
            parameters: Parâmetros adicionais para carregamento
            
        Returns:
            Modelo llama-cpp-python
        """
        import llama_cpp
        
        # Parâmetros padrão
        default_params = {
            "n_ctx": parameters.get("context_length", 4096),
            "n_threads": self.hardware.get_optimal_threads(parameters.get("size_mb", 0)),
            "n_batch": parameters.get("n_batch", 512),
        }
        
        # GPU se disponível
        if self.hardware.has_cuda:
            # Tenta estimar número de camadas na GPU
            model_size_mb = parameters.get("size_mb", 0)
            if self.hardware.can_fit_on_gpu(model_size_mb):
                default_params["n_gpu_layers"] = -1  # Todas as camadas na GPU
            else:
                # Carrega uma parte proporcional à VRAM disponível
                vram = self.hardware.cuda_device_memory_mb
                if vram > 0:
                    # 40 é um número típico de camadas em modelos grandes
                    default_params["n_gpu_layers"] = max(1, int((vram * 0.7 / model_size_mb) * 40))
        
        # Mesclar com parâmetros fornecidos
        load_params = {**default_params, **parameters}
        
        # Remove parâmetros que não são suportados por llama-cpp
        for key in list(load_params.keys()):
            if key not in ["n_ctx", "n_parts", "seed", "f16_kv", "use_mlock", 
                         "vocab_only", "n_threads", "n_batch", "last_n_tokens_size",
                         "lora_path", "lora_scale", "n_gpu_layers", "verbose"]:
                if key not in ["context_length", "size_mb"]:  # Nossos parâmetros internos
                    logger.debug(f"Removendo parâmetro não suportado para GGUF: {key}")
                load_params.pop(key)
        
        # Carrega o modelo
        try:
            model = llama_cpp.Llama(
                model_path=str(path),
                **load_params
            )
            return model
        except Exception as e:
            # Tenta com menos parâmetros se falhar
            logger.warning(f"Erro ao carregar modelo GGUF: {e}. Tentando com fallback...")
            
            try:
                minimal_params = {
                    "n_ctx": 2048,
                    "n_threads": min(4, os.cpu_count() or 4),
                }
                
                model = llama_cpp.Llama(
                    model_path=str(path),
                    **minimal_params
                )
                return model
            except Exception as e2:
                raise RuntimeError(f"Falha ao carregar modelo GGUF mesmo com fallback: {e2}")
    
    def _load_onnx_model(self, path: Path, parameters: Dict[str, Any]):
        """
        Carrega um modelo ONNX.
        
        Args:
            path: Caminho para o arquivo ou diretório do modelo
            parameters: Parâmetros adicionais para carregamento
            
        Returns:
            Sessão ONNX Runtime
        """
        import onnxruntime as ort
        
        # Determina o melhor provedor
        providers = []
        if self.hardware.has_cuda:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        # Opções da sessão
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = True
        options.enable_mem_pattern = True
        
        # Número de threads
        if "num_threads" in parameters:
            options.intra_op_num_threads = parameters["num_threads"]
        
        # Carrega o modelo
        path_str = str(path)
        try:
            session = ort.InferenceSession(
                path_str,
                providers=providers,
                sess_options=options
            )
            return session
        except Exception as e:
            logger.error(f"Erro ao carregar modelo ONNX: {e}")
            raise
    
    def _load_ctransformers_model(self, path: Path, parameters: Dict[str, Any]):
        """
        Carrega um modelo usando CTransformers.
        
        Args:
            path: Caminho para o arquivo do modelo
            parameters: Parâmetros adicionais para carregamento
            
        Returns:
            Modelo CTransformers
        """
        from ctransformers import AutoModelForCausalLM
        
        # Determina o tipo de modelo
        model_type = None
        path_str = str(path).lower()
        
        if "llama" in path_str:
            model_type = "llama"
        elif "mistral" in path_str:
            model_type = "mistral"
        elif "falcon" in path_str:
            model_type = "falcon"
        elif "gpt" in path_str:
            model_type = "gpt2"
        elif "mpt" in path_str:
            model_type = "mpt"
        elif any(x in path_str for x in ["phi", "qwen", "bloom"]):
            # Tenta llama como fallback para vários modelos
            model_type = "llama"
        else:
            # Tenta inferir pelo nome do arquivo
            name = Path(path).name.lower()
            if "-llama-" in name or "llama2" in name:
                model_type = "llama"
            else:
                model_type = "auto"
        
        # Verifica se o tipo é suportado
        if model_type == "auto" and self._is_transformers_available():
            from transformers import AutoConfig
            try:
                config = AutoConfig.from_pretrained(path)
                model_type = config.model_type
            except Exception:
                model_type = "llama"  # Fallback comum
        
        # Parâmetros para carregamento
        config = {
            "model_type": model_type,
            "context_length": parameters.get("context_length", 4096),
            "gpu_layers": 0  # Padrão para CPU
        }
        
        # Ajusta para GPU se disponível
        if self.hardware.has_cuda:
            model_size_mb = parameters.get("size_mb", 0)
            if self.hardware.can_fit_on_gpu(model_size_mb):
                config["gpu_layers"] = -1  # Todas as camadas
            else:
                # Carrega uma parte proporcional à VRAM disponível
                vram = self.hardware.cuda_device_memory_mb
                if vram > 0:
                    # Estima camadas proporcionalmente
                    config["gpu_layers"] = max(1, int((vram * 0.7 / model_size_mb) * 40))
        
        # Carrega o modelo
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(path),
                **config
            )
            return model
        except Exception as e:
            # Tenta com menos parâmetros
            logger.warning(f"Erro ao carregar CTransformers: {e}. Tentando com configuração básica...")
            
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    str(path),
                    model_type=model_type
                )
                return model
            except Exception as e2:
                raise RuntimeError(f"Falha ao carregar CTransformers mesmo com fallback: {e2}")
    
    @retry(max_attempts=2)
    def load_model(self, path: Union[str, Path], model_type: str, 
                 parameters: Dict[str, Any]) -> Tuple[Any, Optional[Any]]:
        """
        Carrega um modelo com a biblioteca apropriada.
        
        Args:
            path: Caminho para o arquivo ou diretório do modelo
            model_type: Tipo do modelo
            parameters: Parâmetros adicionais para carregamento
            
        Returns:
            Tuple com (modelo, tokenizer) ou (modelo, None)
        """
        path = Path(path)
        
        # Verifica se já está carregado
        model_key = str(path)
        if model_key in self.loaded_models:
            logger.info(f"Modelo já carregado: {path.name}")
            self.references[model_key] = self.references.get(model_key, 0) + 1
            return self.loaded_models[model_key]
        
        # Cria um lock se não existir
        if model_key not in self.locks:
            self.locks[model_key] = threading.Lock()
        
        # Carrega com lock exclusivo
        with self.locks[model_key]:
            # Garante que outro thread não carregou enquanto esperava
            if model_key in self.loaded_models:
                self.references[model_key] = self.references.get(model_key, 0) + 1
                return self.loaded_models[model_key]
            
            if not path.exists():
                raise FileNotFoundError(f"Caminho do modelo não existe: {path}")
            
            logger.info(f"Carregando modelo: {path.name} ({model_type})")
            start_time = time.time()
            
            # Escolhe o método de carregamento adequado
            model_and_tokenizer = None
            try:
                if model_type == "transformers":
                    if self._is_transformers_available():
                        model_and_tokenizer = self._load_transformers_model(path, parameters)
                    else:
                        raise ImportError("Biblioteca 'transformers' não está disponível")
                
                elif model_type == "gguf":
                    if self._is_gguf_available():
                        model = self._load_gguf_model(path, parameters)
                        model_and_tokenizer = (model, None)
                    else:
                        raise ImportError("Biblioteca 'llama-cpp-python' não está disponível")
                
                elif model_type == "onnx":
                    if self._is_onnx_available():
                        model = self._load_onnx_model(path, parameters)
                        model_and_tokenizer = (model, None)
                    else:
                        raise ImportError("Biblioteca 'onnxruntime' não está disponível")
                
                elif model_type == "ctransformers":
                    if self._is_ctransformers_available():
                        model = self._load_ctransformers_model(path, parameters)
                        model_and_tokenizer = (model, None)
                    else:
                        raise ImportError("Biblioteca 'ctransformers' não está disponível")
                
                else:
                    raise ValueError(f"Tipo de modelo não suportado: {model_type}")
                
            except Exception as e:
                logger.error(f"Erro ao carregar modelo {path.name}: {e}")
                if isinstance(e, ImportError):
                    logger.error(f"Biblioteca necessária não instalada para {model_type}")
                raise
            
            # Armazena o modelo carregado
            self.loaded_models[model_key] = model_and_tokenizer
            self.references[model_key] = 1
            
            # Limpa memória após carregamento
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            elapsed = time.time() - start_time
            logger.info(f"Modelo carregado em {elapsed:.2f}s: {path.name}")
            
            return model_and_tokenizer
    
    def unload_model(self, path: Union[str, Path]):
        """
        Descarrega um modelo da memória.
        
        Args:
            path: Caminho para o modelo
        """
        model_key = str(Path(path))
        
        # Decrementa referência
        if model_key in self.references:
            self.references[model_key] -= 1
            
            # Se não há mais referências, descarrega
            if self.references[model_key] <= 0:
                with self.locks.get(model_key, threading.Lock()):
                    if model_key in self.loaded_models:
                        logger.info(f"Descarregando modelo: {Path(path).name}")
                        
                        model_and_tokenizer = self.loaded_models[model_key]
                        
                        # Limpa referências específicas por tipo
                        if isinstance(model_and_tokenizer, tuple):
                            model, tokenizer = model_and_tokenizer
                            
                            # Pytorch/Transformers
                            if hasattr(model, "to"):
                                try:
                                    model.to("cpu")
                                except Exception:
                                    pass
                            
                            # Libera CUDA
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        # Remove do dicionário
                        del self.loaded_models[model_key]
                        del self.references[model_key]
                        
                        # Coleta de lixo explícita
                        gc.collect()
                        
                        logger.info(f"Modelo descarregado: {Path(path).name}")
    
    def unload_all_models(self):
        """Descarrega todos os modelos da memória."""
        model_keys = list(self.loaded_models.keys())
        for model_key in model_keys:
            self.references[model_key] = 0
            self.unload_model(model_key)


class UniversalModelWrapper:
    """
    Wrapper universal para modelos de linguagem locais.
    
    Fornece uma interface unificada para diferentes tipos de modelos:
    - Transformers (HuggingFace)
    - GGUF (sucessor do GGML)
    - ONNX
    - CTransformers
    
    Recursos:
    - Carregamento preguiçoso
    - Otimização automática para hardware
    - Fallback para modelos alternativos
    - Segurança na geração
    """
    
    def __init__(
        self,
        model_path: Union[str, Path] = None,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        context_length: int = 4096,
        fallback_model: Optional[Union[str, Path]] = None,
        lazy_load: bool = True,
        gpu_id: Optional[int] = None,
        gpu_monitor: Optional[Any] = None,
        model_name_or_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        quantization: Optional[str] = None,
        max_length: Optional[int] = None
    ):
        """
        Inicializa o wrapper universal.
        
        Args:
            model_path: Caminho para o arquivo ou diretório do modelo
            model_name_or_path: Alias para model_path (compatibilidade)
            model_type: Tipo do modelo (transformers, gguf, etc)
            model_name: Nome amigável do modelo
            capabilities: Lista de capacidades do modelo
            parameters: Parâmetros específicos para o modelo
            context_length: Tamanho máximo do contexto em tokens
            fallback_model: Modelo alternativo em caso de falha
            lazy_load: Se o modelo deve ser carregado apenas quando necessário
            gpu_id: ID da GPU para carregar o modelo (None para auto-detecção)
            gpu_monitor: Instância do GPUMonitor para monitoramento de memória
            device: Device específico para carregar o modelo
            load_in_8bit: Carregamento em 8 bits
            load_in_4bit: Carregamento em 4 bits
            quantization: Tipo de quantização
            max_length: Comprimento máximo para geração
        """
        # Compatibilidade: aceita model_name_or_path como alias para model_path
        if model_name_or_path is not None and model_path is None:
            model_path = model_name_or_path
        elif model_path is None and model_name_or_path is None:
            raise ValueError("Um dos parâmetros 'model_path' ou 'model_name_or_path' deve ser fornecido")
        
        # Normalização do caminho
        self.model_path = Path(model_path)
        self.model_type = self._determine_model_type(model_type)
        self.model_name = model_name or self.model_path.stem
        self.capabilities = capabilities or ["general"]
        self.context_length = max_length or context_length
        self.fallback_model_path = Path(fallback_model) if fallback_model else None
        self.gpu_id = gpu_id
        self.gpu_monitor = gpu_monitor
        self.device = device
        self.quantization = quantization
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # Detecta tamanho do modelo
        try:
            self.model_size_mb = self._get_model_size()
        except Exception:
            self.model_size_mb = 0
        
        # Parâmetros para o modelo
        self.parameters = parameters or {}
        if "size_mb" not in self.parameters:
            self.parameters["size_mb"] = self.model_size_mb
        if "context_length" not in self.parameters:
            self.parameters["context_length"] = self.context_length
        
        # Estado interno
        self._model = None
        self._tokenizer = None
        self._loader = ModelLoader()
        self._hardware = HardwareDetector()
        self._last_used = time.time()
        self._is_loaded = False
        
        # Contador de tokens
        self.total_tokens_generated = 0
        self.total_calls = 0
        
        # Carrega o modelo imediatamente se não for lazy
        if not lazy_load:
            self._load_model()
    
    def _determine_model_type(self, model_type: Optional[str]) -> str:
        """
        Determina o tipo de modelo com base no caminho e tipo fornecido.
        
        Args:
            model_type: Tipo de modelo opcional
            
        Returns:
            Tipo de modelo determinado
        """
        if model_type:
            return model_type.lower()
        
        # Tenta determinar pelo caminho/extensão
        path = self.model_path
        if path.is_file():
            # Verificação por extensão
            if path.name.endswith(".gguf") or path.name.endswith(".ggml"):
                return "gguf"
            elif path.name.endswith(".onnx"):
                return "onnx"
            elif path.name.endswith(".pt") or path.name.endswith(".pth"):
                return "pytorch"
            elif path.name.endswith(".pb"):
                return "tensorflow"
        
        # Para diretórios, procura por indicadores
        elif path.is_dir():
            if (path / "config.json").exists():
                # Diretório Transformers
                return "transformers"
        
        # Padrão para transformers (mais compatível)
        return "transformers"
    
    def _get_model_size(self) -> float:
        """
        Calcula o tamanho aproximado do modelo em MB.
        
        Returns:
            Tamanho em megabytes
        """
        path = self.model_path
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)  # Bytes para MB
        
        # Para diretórios, soma tamanhos dos arquivos relevantes
        total_size = 0
        for ext in [".bin", ".safetensors", ".gguf", ".ggml", ".onnx", ".pt", ".pth"]:
            for file_path in path.glob(f"**/*{ext}"):
                total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)  # Bytes para MB
    
    def _load_model(self):
        """Carrega o modelo se ainda não estiver carregado."""
        if not self._is_loaded:
            # Verifica memória da GPU se for especificada e o monitor estiver disponível
            original_gpu_id = self.gpu_id
            device_target = "cuda"
            
            if self.gpu_id is not None and self.gpu_monitor is not None:
                memory_needed = self.model_size_mb * 1.5  # Fator de 1.5x para overhead
                logger.info(f"Verificando disponibilidade de memória na GPU {self.gpu_id} para modelo {self.model_name} (necessário: {memory_needed:.2f}MB)")
                
                try:
                    # Verifica se a GPU está sobrecarregada
                    if self.gpu_monitor.is_gpu_overloaded(self.gpu_id, threshold_percent=85.0):
                        logger.warning(f"GPU {self.gpu_id} está sobrecarregada (>85%). Buscando alternativa.")
                        
                        # Tenta encontrar outra GPU com memória disponível
                        best_gpu = self.gpu_monitor.get_best_gpu_for_model(memory_needed)
                        if best_gpu is not None and best_gpu != self.gpu_id:
                            logger.info(f"Redirecionando modelo para GPU {best_gpu} com mais memória disponível")
                            self.gpu_id = best_gpu
                        else:
                            # Se não houver GPU com memória suficiente, muda para CPU
                            logger.warning(f"Sem GPU disponível com memória suficiente para o modelo {self.model_name}. Usando CPU.")
                            self.gpu_id = None
                            device_target = "cpu"
                    else:
                        # Verifica se a GPU tem memória suficiente
                        metrics = self.gpu_monitor.get_current_metrics()
                        if self.gpu_id in metrics and "memory_free_mb" in metrics[self.gpu_id]:
                            free_memory = metrics[self.gpu_id]["memory_free_mb"]
                            if free_memory < memory_needed:
                                logger.warning(f"GPU {self.gpu_id} tem apenas {free_memory:.2f}MB livre, mas o modelo precisa de {memory_needed:.2f}MB. Buscando alternativa.")
                                
                                # Tenta encontrar outra GPU com memória disponível
                                best_gpu = self.gpu_monitor.get_best_gpu_for_model(memory_needed)
                                if best_gpu is not None and best_gpu != self.gpu_id:
                                    logger.info(f"Redirecionando modelo para GPU {best_gpu} com mais memória disponível")
                                    self.gpu_id = best_gpu
                                else:
                                    # Se não houver GPU com memória suficiente, muda para CPU
                                    logger.warning(f"Sem GPU disponível com memória suficiente para o modelo {self.model_name}. Usando CPU.")
                                    self.gpu_id = None
                                    device_target = "cpu"
                            else:
                                logger.info(f"GPU {self.gpu_id} tem {free_memory:.2f}MB livre, suficiente para o modelo")
                except Exception as e:
                    logger.error(f"Erro ao verificar memória da GPU: {e}. Usando configuração original.")
                    self.gpu_id = original_gpu_id
            
            # Atualiza os parâmetros com o device escolhido
            if self.gpu_id is not None:
                if "device_map" not in self.parameters:
                    self.parameters["device_map"] = f"cuda:{self.gpu_id}"
                if self.model_type == "gguf" and "n_gpu_layers" not in self.parameters:
                    self.parameters["n_gpu_layers"] = -1  # Todas as camadas na GPU
            else:
                # Configuração para CPU
                if "device_map" not in self.parameters:
                    self.parameters["device_map"] = "cpu"
                if self.model_type == "gguf":
                    self.parameters["n_gpu_layers"] = 0  # Nenhuma camada na GPU
            
            # Carrega o modelo com os parâmetros ajustados
            try:
                self._model, self._tokenizer = self._loader.load_model(
                    self.model_path, self.model_type, self.parameters
                )
                self._is_loaded = True
                self._last_used = time.time()
                logger.info(f"Modelo {self.model_name} carregado com sucesso no dispositivo: {device_target}")
                
                # Restaura o gpu_id original para manter a referência inicial
                self.gpu_id = original_gpu_id
            except Exception as e:
                logger.error(f"Erro ao carregar modelo primário: {e}")
                
                # Tenta o modelo de fallback se disponível
                if self.fallback_model_path and self.fallback_model_path.exists():
                    logger.warning(f"Tentando modelo de fallback: {self.fallback_model_path.name}")
                    try:
                        fallback_type = self._determine_model_type(None)
                        self._model, self._tokenizer = self._loader.load_model(
                            self.fallback_model_path, fallback_type, self.parameters
                        )
                        self._is_loaded = True
                        self._last_used = time.time()
                    except Exception as fallback_error:
                        raise RuntimeError(f"Falha ao carregar modelo primário e fallback: {fallback_error}")
                else:
                    raise
    
    def unload(self):
        """Descarrega o modelo da memória."""
        if self._is_loaded:
            self._loader.unload_model(self.model_path)
            self._model = None
            self._tokenizer = None
            self._is_loaded = False
    
    def _ensure_loaded(self):
        """Garante que o modelo está carregado antes do uso."""
        if not self._is_loaded:
            self._load_model()
    
    def _format_prompt(self, prompt: str) -> str:
        """
        Formata o prompt adequadamente para o modelo.
        
        Args:
            prompt: Prompt de entrada
            
        Returns:
            Prompt formatado para o modelo
        """
        # Formatação para modelos específicos
        model_name = self.model_name.lower()
        
        # Detecta se é um modelo de chat
        is_chat_model = any(chat_type in model_name 
                          for chat_type in ["chat", "instruct", "phi", "mistral", "llama"])
        
        if not is_chat_model:
            return prompt
        
        # Formatações específicas por tipo de modelo
        if "phi-3" in model_name:
            return f"<|user|>\n{prompt}\n<|assistant|>\n"
        elif "phi-2" in model_name:
            return f"Instruct: {prompt}\nOutput: "
        elif "qwen" in model_name:
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif any(x in model_name for x in ["llama2", "llama-2", "mistral"]):
            return f"<s>[INST] {prompt} [/INST] "
        elif "falcon" in model_name:
            return f"User: {prompt}\nAssistant: "
        else:
            return f"Pergunta: {prompt}\nResposta: "
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estima o número de tokens em um texto.
        
        Args:
            text: Texto a ser estimado
            
        Returns:
            Número estimado de tokens
        """
        if self._tokenizer and hasattr(self._tokenizer, "encode"):
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback: estimativa simples baseada em palavras
        return int(len(re.findall(r'\b\w+\b', text)) * 1.33)
    
    def _post_process_output(self, output: str, prompt: str) -> str:
        """
        Processa a saída do modelo.
        
        Args:
            output: Texto gerado pelo modelo
            prompt: Prompt original
            
        Returns:
            Texto processado
        """
        # Alguns modelos retornam o prompt junto com a resposta
        if prompt in output:
            output = output[output.find(prompt) + len(prompt):]
        
        # Remove tokens especiais comuns
        output = output.replace("<|endoftext|>", "")
        output = output.replace("<|im_end|>", "")
        output = output.replace("<|assistant|>", "")
        
        # Remove prefixos comuns de resposta
        prefixes = ["Assistant:", "AI:", "Response:", "Output:", "Answer:"]
        for prefix in prefixes:
            if output.startswith(prefix):
                output = output[len(prefix):].lstrip()
        
        return output.strip()
    
    def _generate_with_transformers(self, prompt: str, generation_config: Dict[str, Any]) -> str:
        """
        Gera texto usando HuggingFace Transformers.
        
        Args:
            prompt: Prompt de entrada
            generation_config: Configuração para geração
            
        Returns:
            Texto gerado
        """
        import torch
        
        model, tokenizer = self._model, self._tokenizer
        
        # Configuração padrão
        config = {
            "max_new_tokens": generation_config.pop("max_tokens", 512),
            "temperature": generation_config.pop("temperature", 0.7),
            "top_p": generation_config.pop("top_p", 0.95),
            "top_k": generation_config.pop("top_k", 40),
            "repetition_penalty": generation_config.pop("repetition_penalty", 1.1),
            "do_sample": generation_config.pop("do_sample", True),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # Adiciona configurações extras se fornecidas
        config.update(generation_config)
        
        # Processa o prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        
        # Move para o dispositivo do modelo
        device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
        input_ids = input_ids.to(device)
        
        # Gera a resposta
        with torch.no_grad():
            output_ids = model.generate(input_ids, **config)
        
        # Decodifica a resposta
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Remove o prompt da resposta
        return self._post_process_output(output, prompt)
    
    def _generate_with_gguf(self, prompt: str, generation_config: Dict[str, Any]) -> str:
        """
        Gera texto usando llama-cpp-python (GGUF).
        
        Args:
            prompt: Prompt de entrada
            generation_config: Configuração para geração
            
        Returns:
            Texto gerado
        """
        model = self._model
        
        # Configuração padrão
        config = {
            "max_tokens": generation_config.pop("max_tokens", 512),
            "temperature": generation_config.pop("temperature", 0.7),
            "top_p": generation_config.pop("top_p", 0.95),
            "top_k": generation_config.pop("top_k", 40),
            "repeat_penalty": generation_config.pop("repetition_penalty", 1.1),
        }
        
        # Adiciona configurações extras se fornecidas
        config.update(generation_config)
        
        # Processa stop tokens
        stop = config.pop("stop", [])
        if isinstance(stop, str):
            stop = [stop]
        
        # Gera a resposta
        output = model(
            prompt,
            **config,
            stop=stop or None
        )
        
        # Extrai o texto gerado
        if isinstance(output, dict) and "choices" in output:
            generated_text = output["choices"][0]["text"]
        elif isinstance(output, dict) and "text" in output:
            generated_text = output["text"]
        else:
            generated_text = str(output)
        
        # Remove o prompt da resposta
        return self._post_process_output(generated_text, prompt)
    
    def _generate_with_ctransformers(self, prompt: str, generation_config: Dict[str, Any]) -> str:
        """
        Gera texto usando CTransformers.
        
        Args:
            prompt: Prompt de entrada
            generation_config: Configuração para geração
            
        Returns:
            Texto gerado
        """
        model = self._model
        
        # Configuração para geração
        config = {
            "max_new_tokens": generation_config.pop("max_tokens", 512),
            "temperature": generation_config.pop("temperature", 0.7),
            "top_p": generation_config.pop("top_p", 0.95),
            "top_k": generation_config.pop("top_k", 40),
            "repetition_penalty": generation_config.pop("repetition_penalty", 1.1),
        }
        
        # Gera a resposta
        output = model(prompt, **config)
        
        # Remove o prompt da resposta
        return self._post_process_output(output, prompt)
    
    def _generate_with_onnx(self, prompt: str, generation_config: Dict[str, Any]) -> str:
        """
        Gera texto usando ONNX Runtime.
        
        Args:
            prompt: Prompt de entrada
            generation_config: Configuração para geração
            
        Returns:
            Texto gerado
        """
        # Esta implementação é um placeholder para modelos ONNX
        session = self._model
        
        # Implementação completa exigiria:
        # 1. Preparar entradas no formato correto para o modelo
        # 2. Executar a inferência usando session.run()
        # 3. Processar saídas para texto
        
        # Como isso é altamente específico para o modelo, aqui está uma implementação simplificada
        logger.warning("Geração com ONNX ainda não totalmente implementada")
        
        # Retorna um texto de aviso
        return (f"[Geração com modelo ONNX não implementada completamente. "
              f"Prompt recebido: '{prompt[:30]}...']")
    
    @retry(max_attempts=2)
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Gera texto a partir de um prompt.
        
        Args:
            prompt: Prompt de entrada
            max_tokens: Número máximo de tokens a gerar
            temperature: Temperatura para amostragem (0-1)
            top_p: Probabilidade cumulativa para amostragem nucleus
            top_k: Número de tokens mais prováveis para considerar
            repetition_penalty: Penalidade para repetição de tokens
            stop_sequences: Sequências que indicam fim da geração
            **kwargs: Parâmetros adicionais específicos do modelo
            
        Returns:
            Texto gerado
        """
        # Garante que o modelo está carregado
        self._ensure_loaded()
        self._last_used = time.time()
        
        # Formata o prompt apropriadamente
        formatted_prompt = self._format_prompt(prompt)
        
        # Configura parâmetros para geração
        generation_config = {}
        
        # Adiciona parâmetros se fornecidos
        if max_tokens is not None:
            generation_config["max_tokens"] = max_tokens
        if temperature is not None:
            generation_config["temperature"] = temperature
        if top_p is not None:
            generation_config["top_p"] = top_p
        if top_k is not None:
            generation_config["top_k"] = top_k
        if repetition_penalty is not None:
            generation_config["repetition_penalty"] = repetition_penalty
        if stop_sequences:
            generation_config["stop"] = stop_sequences
        
        # Adiciona outros parâmetros
        generation_config.update(kwargs)
        
        # Mede tempo de execução
        start_time = time.time()
        
        # Gera texto com base no tipo de modelo
        try:
            if self.model_type == "transformers":
                output = self._generate_with_transformers(formatted_prompt, generation_config)
            elif self.model_type == "gguf":
                output = self._generate_with_gguf(formatted_prompt, generation_config)
            elif self.model_type == "ctransformers":
                output = self._generate_with_ctransformers(formatted_prompt, generation_config)
            elif self.model_type == "onnx":
                output = self._generate_with_onnx(formatted_prompt, generation_config)
            else:
                raise ValueError(f"Tipo de modelo não suportado: {self.model_type}")
        except Exception as e:
            logger.error(f"Erro na geração: {e}")
            # Fallback para geração simples
            if self.model_type == "transformers" and self._tokenizer:
                logger.info("Tentando fallback para geração simplificada")
                try:
                    input_ids = self._tokenizer(formatted_prompt, return_tensors="pt").input_ids
                    device = next(self._model.parameters()).device
                    input_ids = input_ids.to(device)
                    
                    output_ids = self._model.generate(
                        input_ids,
                        max_new_tokens=generation_config.get("max_tokens", 512),
                        do_sample=True
                    )
                    
                    output = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    output = self._post_process_output(output, formatted_prompt)
                except Exception as fallback_error:
                    logger.error(f"Fallback também falhou: {fallback_error}")
                    output = "[Erro na geração de texto]"
            else:
                output = f"[Erro: {str(e)}]"
        
        # Calcula estatísticas
        self.total_calls += 1
        tokens_generated = self._estimate_tokens(output)
        self.total_tokens_generated += tokens_generated
        
        elapsed = time.time() - start_time
        tokens_per_second = tokens_generated / elapsed if elapsed > 0 else 0
        
        logger.debug(f"Geração: {tokens_generated} tokens em {elapsed:.2f}s ({tokens_per_second:.1f} tokens/s)")
        
        return output
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Gera resposta para uma conversa de chat.
        
        Args:
            messages: Lista de mensagens no formato [{role, content}, ...]
            max_tokens: Número máximo de tokens a gerar
            temperature: Temperatura para geração
            **kwargs: Parâmetros adicionais
            
        Returns:
            Resposta gerada
        """
        # Formata mensagens no formato de chat
        if not messages:
            return ""
        
        # Formata conversa com base no tipo de modelo
        model_name = self.model_name.lower()
        system_message = ""
        prompt = ""
        
        # Extrai mensagem do sistema, se presente
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
                break
        
        # Formato específico por modelo
        if "phi-3" in model_name:
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    prompt += f"<|system|>\n{content}\n"
                elif role == "user":
                    prompt += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    prompt += f"<|assistant|>\n{content}\n"
            
            prompt += "<|assistant|>\n"
        
        elif "phi-2" in model_name:
            prompt = ""
            if system_message:
                prompt += f"System: {system_message}\n\n"
            
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role != "system":
                    if role == "user":
                        prompt += f"Instruct: {content}\n"
                    elif role == "assistant":
                        prompt += f"Output: {content}\n"
            
            prompt += "Output: "
        
        elif any(x in model_name for x in ["llama2", "llama-2", "mistral"]):
            prompt = ""
            
            if system_message:
                prompt += f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n"
            else:
                prompt += "<s>[INST] "
            
            # Adiciona mensagens alternando entre usuário e assistente
            for i, msg in enumerate(messages):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    continue
                
                if i > 0 and role == "user" and messages[i-1].get("role") == "assistant":
                    prompt += f" [/INST] {content} [INST] "
                elif role == "user":
                    prompt += content
                elif role == "assistant":
                    prompt += f" [/INST] {content} [INST] "
            
            prompt += " [/INST] "
        
        else:
            # Formato genérico para outros modelos
            prompt = ""
            if system_message:
                prompt += f"Contexto: {system_message}\n\n"
            
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role != "system":
                    if role == "user":
                        prompt += f"Pergunta: {content}\n"
                    elif role == "assistant":
                        prompt += f"Resposta: {content}\n"
            
            prompt += "Resposta: "
        
        # Gera resposta usando o prompt formatado
        return self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokeniza um texto usando o tokenizer do modelo.
        
        Args:
            text: Texto a ser tokenizado
            
        Returns:
            Lista de tokens (IDs)
        """
        self._ensure_loaded()
        
        # Usa tokenizer se disponível
        if self._tokenizer and hasattr(self._tokenizer, "encode"):
            try:
                return self._tokenizer.encode(text)
            except Exception as e:
                logger.warning(f"Erro ao tokenizar com tokenizer: {e}")
        
        # Estimativa para modelos sem tokenizer
        return list(range(self._estimate_tokens(text)))
    
    def embedding(self, text: str) -> List[float]:
        """
        Gera embedding para um texto.
        
        Args:
            text: Texto para gerar embedding
            
        Returns:
            Lista de valores float representando o embedding
        """
        self._ensure_loaded()
        
        # Implementação para Transformers
        if self.model_type == "transformers" and hasattr(self._model, "get_input_embeddings"):
            try:
                import torch
                
                # Tokeniza o texto
                input_ids = self._tokenizer.encode(text, return_tensors="pt")
                
                # Obtém o dispositivo do modelo
                device = next(self._model.parameters()).device
                input_ids = input_ids.to(device)
                
                # Obtém embeddings
                with torch.no_grad():
                    # Obtém camada de embedding
                    embedding_layer = self._model.get_input_embeddings()
                    
                    # Gera embeddings
                    embeddings = embedding_layer(input_ids)
                    
                    # Calcula média dos embeddings (para representação do texto)
                    mean_embedding = embeddings.mean(dim=1)
                    
                    # Normaliza o embedding
                    normalized = torch.nn.functional.normalize(mean_embedding, p=2, dim=1)
                    
                    # Converte para lista de Python
                    return normalized[0].cpu().tolist()
            
            except Exception as e:
                logger.warning(f"Erro ao gerar embedding com transformers: {e}")
        
        # Fallback - gera embedding aleatório (placeholder)
        # Em produção, deve-se implementar uma solução mais robusta
        logger.warning("Usando embedding aleatório (fallback)")
        import numpy as np
        
        # Seed baseada no hash do texto para consistência
        seed = hash(text) % 2**32
        np.random.seed(seed)
        
        # Gera embedding aleatório com dimensão típica
        return np.random.normal(0, 0.1, 768).tolist()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtém métricas sobre o uso do modelo.
        
        Returns:
            Dicionário com métricas
        """
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "model_size_mb": self.model_size_mb,
            "is_loaded": self._is_loaded,
            "total_tokens_generated": self.total_tokens_generated,
            "total_calls": self.total_calls,
            "last_used": self._last_used,
            "capabilities": self.capabilities,
            "context_length": self.context_length
        }
    
    def __del__(self):
        """Cleanup ao destruir o objeto."""
        try:
            if self._is_loaded:
                self.unload()
        except Exception:
            pass


# Funções auxiliares

def create_universal_wrapper(
    model_path: Union[str, Path],
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
    capabilities: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    context_length: int = 4096,
    fallback_model: Optional[Union[str, Path]] = None,
    lazy_load: bool = True
) -> UniversalModelWrapper:
    """
    Cria uma instância do UniversalModelWrapper para o caminho especificado.
    
    Args:
        model_path: Caminho para o modelo
        model_type: Tipo do modelo
        model_name: Nome amigável do modelo
        capabilities: Lista de capacidades
        parameters: Parâmetros adicionais
        context_length: Tamanho máximo do contexto
        fallback_model: Caminho para modelo de fallback
        lazy_load: Se deve carregar apenas quando necessário
        
    Returns:
        Instância de UniversalModelWrapper
    """
    return UniversalModelWrapper(
        model_path=model_path,
        model_type=model_type,
        model_name=model_name,
        capabilities=capabilities,
        parameters=parameters,
        context_length=context_length,
        fallback_model=fallback_model,
        lazy_load=lazy_load
    )

def get_optimal_wrapper_params(
    model_path: Union[str, Path],
    model_size_mb: Optional[float] = None
) -> Dict[str, Any]:
    """
    Obtém parâmetros ótimos para um modelo com base no hardware.
    
    Args:
        model_path: Caminho para o modelo
        model_size_mb: Tamanho do modelo em MB
        
    Returns:
        Dicionário com parâmetros otimizados
    """
    # Detecta hardware
    hardware = HardwareDetector()
    
    # Calcula tamanho se não fornecido
    if model_size_mb is None:
        path = Path(model_path)
        if path.is_file():
            model_size_mb = path.stat().st_size / (1024 * 1024)
        else:
            # Estimativa para diretório
            model_size_mb = 5000  # 5GB por padrão
    
    # Parâmetros base
    params = {
        "use_gpu": hardware.has_cuda or hardware.has_mps,
        "trust_remote_code": True,
    }
    
    # Ajusta com base no dispositivo
    if hardware.has_cuda:
        # Configura parâmetros para GPU CUDA
        if model_size_mb < 3000:  # < 3GB
            params["device_map"] = "cuda:0"
            params["torch_dtype"] = "float16"
        else:
            params["device_map"] = "auto"
            params["low_cpu_mem_usage"] = True
            
            # Para modelos grandes, usar carregamento otimizado
            if model_size_mb > 8000:  # > 8GB
                params["torch_dtype"] = "bfloat16" if torch.cuda.is_available() and \
                                      torch.cuda.get_device_capability()[0] >= 8 else "float16"
            else:
                params["torch_dtype"] = "float16"
    
    elif hardware.has_mps:
        # Configura para Apple Silicon
        params["device_map"] = "mps"
        params["torch_dtype"] = "float16"
    
    else:
        # Configura para CPU
        params["device_map"] = "cpu"
        
        # Otimizações para CPU
        params["n_threads"] = hardware.get_optimal_threads(model_size_mb)
    
    return params


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="UniversalModelWrapper")
    parser.add_argument("--path", required=True, help="Caminho para o modelo")
    parser.add_argument("--type", help="Tipo de modelo (transformers, gguf, etc)")
    parser.add_argument("--prompt", default="Quem é você?", help="Prompt para geração")
    parser.add_argument("--max-tokens", type=int, default=512, help="Tokens máximos")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperatura")
    
    args = parser.parse_args()
    
    # Cria wrapper
    wrapper = create_universal_wrapper(
        model_path=args.path,
        model_type=args.type,
        parameters=get_optimal_wrapper_params(args.path)
    )
    
    # Gera resposta
    print(f"Modelo: {wrapper.model_name} ({wrapper.model_type})")
    print(f"Prompt: {args.prompt}")
    print("-" * 40)
    
    response = wrapper.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    print(response)
    print("-" * 40)
    
    # Métricas
    metrics = wrapper.get_metrics()
    print(f"Tokens gerados: {metrics['total_tokens_generated']}")
    
    # Descarrega modelo
    wrapper.unload()
