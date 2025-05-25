"""
EzioBaseExpert - Classe base unificada para todos os especialistas no sistema EzioFilho
Fornece funcionalidades comuns de carregamento de modelo, inferência e métricas
"""
import os
import json
import time
import logging
import uuid
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [%(name)s] | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class EzioBaseExpert:
    """
    Classe base unificada para todos os especialistas do sistema EzioFilho.
    Gerencia carregamento de modelos, tokenizadores, inferência e métricas.
    """
    
    # Versão do framework base de especialistas
    VERSION = "3.0.0"
    
    def __init__(self, 
                expert_type: str,
                config_path: Optional[Union[str, Path]] = None, 
                gpu_id: Optional[int] = None,
                gpu_ids: Optional[List[int]] = None,
                gpu_monitor: Optional[Any] = None,
                model_path_override: Optional[str] = None,
                system_message: Optional[str] = None,
                quantization: Optional[str] = None):
        """
        Inicializa o especialista base
        
        Args:
            expert_type: Tipo de especialista (sentiment, factcheck, etc.)
            config_path: Caminho para arquivo de configuração
            gpu_id: ID da GPU a ser utilizada
            gpu_ids: Lista de IDs das GPUs disponíveis
            gpu_monitor: Instância do GPUMonitor para gerenciamento de memória
            model_path_override: Sobrescrição do caminho do modelo na configuração
            system_message: Mensagem de sistema para o especialista
            quantization: Método de quantização (4bit, 8bit, None)
        """
        # Rastreamento de status para o orquestrador
        self.is_initialized = False
        self.initialization_error = None
        self.initialization_time = time.time()
        
        # Propriedades básicas do especialista
        self.expert_type = expert_type
        self.expert_id = f"{expert_type}_{self._generate_id()}"
        self.logger = logging.getLogger(f"Expert_{self.expert_id}")
        
        # Carregar configuração
        config_path = Path(config_path) if config_path else Path("models_config.json")
        try:
            self.config = self._load_config(config_path)
            self.logger.info(f"📄 Configuração carregada para {expert_type}")
        except Exception as e:
            error_msg = f"❌ Erro ao carregar configuração: {e}"
            self.logger.error(error_msg)
            self.config = {"models": {}}
            self.is_initialized = False
            self.initialization_error = error_msg
        
        # Configurar GPU
        self.gpu_id = self._select_gpu(gpu_id)
        self.gpu_ids = gpu_ids
        self.gpu_monitor = gpu_monitor
        
        if self.gpu_id is not None:
            try:
                gpu_name = torch.cuda.get_device_name(self.gpu_id)
                gpu_mem = torch.cuda.get_device_properties(self.gpu_id).total_memory / (1024**3)
                self.logger.info(f"🖥️ GPU selecionada: {gpu_name} (ID: {self.gpu_id}, Memória: {gpu_mem:.2f} GB)")
                self.device = f"cuda:{self.gpu_id}"
            except Exception as e:
                self.logger.warning(f"Erro ao obter informações da GPU {self.gpu_id}: {e}")
                self.device = "cpu"
        else:
            self.logger.info(f"💻 Usando CPU para inferência")
            self.device = "cpu"
        
        # Configurar mensagem do sistema
        self.system_message = system_message
        
        # Inicializar modelo
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.model_name = "unknown"
        
        # Obter caminho do modelo da configuração ou sobrescrito
        model_config = self.config.get("models", {}).get(expert_type, {})
        self.model_path = model_path_override or model_config.get("path")
        
        # Configurar quantização
        self.quantization = quantization or model_config.get("quantization")
        
        # Métricas de desempenho
        self.metrics = {
            "model_load_time": 0,
            "inference_count": 0,
            "total_inference_time": 0,
            "total_tokens_generated": 0,
            "errors": 0,
            "oom_events": 0,
            "execution_times": {}
        }
        
        # Tentar carregar o modelo se o caminho estiver disponível
        if self.model_path:
            self._load_model()
    
    def _generate_id(self) -> str:
        """Gera um ID único para esta instância do especialista"""
        return uuid.uuid4().hex[:8]
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """
        Carrega configuração do arquivo especificado
        
        Args:
            config_path: Caminho para o arquivo de configuração
            
        Returns:
            Dicionário com a configuração carregada
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar configuração: {e}")
            raise
    
    def _select_gpu(self, gpu_id: Optional[int]) -> Optional[int]:
        """
        Seleciona GPU para uso, com fallback para autodetecção
        
        Args:
            gpu_id: ID específico de GPU para usar ou None para autodetecção
            
        Returns:
            ID da GPU selecionada ou None se usando CPU
        """
        if not torch.cuda.is_available():
            self.logger.warning("⚠️ CUDA não disponível, usando CPU")
            return None
        
        if gpu_id is not None and gpu_id < torch.cuda.device_count():
            return gpu_id
        
        # Autodetecção baseada em memória disponível
        if torch.cuda.device_count() > 0:
            free_memory = []
            for i in range(torch.cuda.device_count()):
                free_mem = (torch.cuda.get_device_properties(i).total_memory - 
                           torch.cuda.memory_allocated(i)) / (1024**3)
                free_memory.append((i, free_mem))
            
            # Escolher GPU com mais memória livre
            selected_gpu = max(free_memory, key=lambda x: x[1])[0]
            return selected_gpu
        
        return None
    
    def _load_model(self) -> None:
        """Carrega o modelo a partir do caminho especificado"""
        start_time = time.time()
        original_gpu_id = self.gpu_id
        
        try:
            # Importar aqui para evitar carregar transformers no tempo de importação do módulo
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Log de carregamento do modelo
            self.logger.info(f"⏳ Carregando modelo de {self.model_path}")
            
            # Verificar memória disponível usando o GPUMonitor se disponível
            if self.gpu_monitor and self.gpu_id is not None:
                # Estimar tamanho do modelo (regra prática: 4 bytes por parâmetro)
                # Podemos melhorar essa estimativa no futuro
                model_size_mb = 5000  # Valor padrão estimado em MB
                
                try:
                    # Verificar se o modelo cabe na GPU atual
                    metrics = self.gpu_monitor.get_current_metrics()
                    if self.gpu_id in metrics and "memory_free_mb" in metrics[self.gpu_id]:
                        free_memory = metrics[self.gpu_id]["memory_free_mb"]
                        required_memory = model_size_mb * 1.5  # Fator de segurança de 1.5x
                        
                        self.logger.info(f"Verificando memória na GPU {self.gpu_id}: {free_memory:.2f}MB livre, necessário ~{required_memory:.2f}MB")
                        
                        if free_memory < required_memory:
                            self.logger.warning(f"Memória insuficiente na GPU {self.gpu_id}. Buscando alternativa.")
                            
                            # Tentar encontrar outra GPU com memória suficiente
                            best_gpu = self.gpu_monitor.get_best_gpu_for_model(required_memory)
                            if best_gpu is not None and best_gpu != self.gpu_id:
                                self.logger.info(f"Redirecionando para GPU {best_gpu} com mais memória disponível")
                                self.gpu_id = best_gpu
                                self.device = f"cuda:{self.gpu_id}"
                            else:
                                # Se não houver GPU adequada, usar CPU
                                self.logger.warning(f"Sem GPU disponível com memória suficiente. Usando CPU.")
                                self.gpu_id = None
                                self.device = "cpu"
                except Exception as e:
                    self.logger.error(f"Erro ao verificar memória via GPUMonitor: {e}")
            
            # Log de memória GPU antes do carregamento
            if self.gpu_id is not None:
                free_mem = (torch.cuda.get_device_properties(self.gpu_id).total_memory - 
                           torch.cuda.memory_allocated(self.gpu_id)) / (1024**3)
                self.logger.info(f"Memória GPU disponível antes de carregar: {free_mem:.2f} GB")
            
            # Configurações de quantização
            kwargs = {
                "device_map": self.device,
                "torch_dtype": torch.float16
            }
            
            if self.quantization == "4bit":
                kwargs["load_in_4bit"] = True
            elif self.quantization == "8bit":
                kwargs["load_in_8bit"] = True
            
            # Carregar modelo
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
            
            # Carregar tokenizador
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Definir flags de status
            self.model_loaded = True
            self.model_name = Path(self.model_path).name if '/' in self.model_path else self.model_path
            
            # Verificar uso de memória após carregamento
            if self.gpu_id is not None:
                mem_used = torch.cuda.memory_allocated(self.gpu_id) / (1024**3)
                self.logger.info(f"✅ Modelo carregado, usando {mem_used:.2f} GB de memória GPU")
                
            # Atualizar métricas
            self.metrics["model_load_time"] = time.time() - start_time
            
            # Log de sucesso
            self.logger.info(f"✅ Especialista inicializado com sucesso: {self.expert_type} ({self.model_path})")
            
            # Status para o orquestrador
            self.is_initialized = True
            self.initialization_error = None
            
            # Restaurar o gpu_id original para manter a referência
            self.gpu_id = original_gpu_id
            
        except Exception as e:
            error_msg = f"❌ Erro ao carregar modelo: {e}"
            self.logger.error(error_msg)
            self.model_loaded = False
            self.metrics["errors"] += 1
            
            # Status para o orquestrador
            self.is_initialized = False
            self.initialization_error = error_msg
    
    def unload_model(self) -> bool:
        """
        Descarrega o modelo da memória para liberar recursos.
        Útil quando o sistema necessita de gerenciamento dinâmico de memória.
        
        Returns:
            True se o modelo foi descarregado com sucesso, False caso contrário
        """
        if not self.model_loaded or self.model is None:
            self.logger.warning("Tentativa de descarregar modelo que não está carregado")
            return False
            
        try:
            # Log de recurso sendo liberado
            self.logger.info(f"Descarregando modelo {self.model_name} para liberar recursos")
            
            # Verificar uso de memória antes de descarregar
            if hasattr(torch.cuda, "memory_allocated") and self.gpu_id is not None:
                try:
                    mem_before = torch.cuda.memory_allocated(self.gpu_id) / (1024**3)
                    self.logger.debug(f"Memória GPU antes de descarregar: {mem_before:.2f} GB")
                except Exception:
                    mem_before = 0
            else:
                mem_before = 0
            
            # Descarrega modelo
            if hasattr(self.model, "to"):
                self.model.to("cpu")
                
            # Libera referências
            del self.model
            self.model = None
            
            # Executa coleta de lixo explícita
            import gc
            gc.collect()
            
            # Limpar cache CUDA
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
            
            # Verificar uso de memória após descarregar
            if hasattr(torch.cuda, "memory_allocated") and self.gpu_id is not None and mem_before > 0:
                try:
                    mem_after = torch.cuda.memory_allocated(self.gpu_id) / (1024**3)
                    mem_freed = mem_before - mem_after
                    self.logger.info(f"Memória GPU liberada: {mem_freed:.2f} GB")
                except Exception as e:
                    self.logger.warning(f"Erro ao calcular memória liberada: {e}")
            
            # Atualiza estado
            self.model_loaded = False
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao descarregar modelo: {e}")
            return False
    
    def analyze(self, 
               text: str,
               max_tokens: int = 512, 
               context: Optional[str] = None,
               temperature: float = 0.1,
               system_message_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Analisa texto com o modelo carregado
        
        Args:
            text: Texto a ser analisado
            max_tokens: Número máximo de tokens a serem gerados
            context: Contexto adicional
            temperature: Temperatura para geração
            system_message_override: Sobrescrição da mensagem do sistema
            
        Returns:
            Dicionário com resultados da análise
        """
        # Verificar se o modelo está carregado
        if not self.model_loaded:
            return {
                "status": "error",
                "error": "Modelo não carregado",
                "response": None
            }
            
        # Rastrear métricas
        start_time = time.time()
        self.metrics["inference_count"] += 1
        
        try:
            # Preparar prompt
            if context:
                prompt = f"{context}\n\n{text}"
            else:
                prompt = text
                
            # Adicionar mensagem do sistema se disponível
            system = system_message_override or self.system_message
            if system:
                if "<text>" in system:
                    prompt = system.replace("<text>", prompt)
                else:
                    prompt = f"{system}\n\n{prompt}"
            
            # Tokenizar entrada
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_tokens = len(inputs.input_ids[0])
            
            # Gerar resposta
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=(temperature > 0),
                    temperature=max(temperature, 1e-5),
                    top_p=0.95,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decodificar saída
            output_text = self.tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True)
            
            # Calcular tokens gerados
            tokens_generated = len(outputs[0]) - input_tokens
            
            # Atualizar métricas
            inference_time = time.time() - start_time
            self.metrics["total_inference_time"] += inference_time
            self.metrics["total_tokens_generated"] += tokens_generated
            
            return {
                "status": "success",
                "response": output_text,
                "tokens_prompt": input_tokens,
                "tokens_generated": tokens_generated,
                "latency": inference_time
            }
            
        except RuntimeError as e:
            # Verificar erro de memória
            if "out of memory" in str(e).lower():
                self.logger.error(f"❌ GPU sem memória: {e}")
                self.metrics["oom_events"] += 1
            else:
                self.logger.error(f"❌ Erro de execução: {e}")
                
            self.metrics["errors"] += 1
            
            return {
                "status": "error",
                "error": str(e),
                "response": None
            }
            
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas do especialista
        
        Returns:
            Dicionário com métricas de desempenho
        """
        # Calcular métricas derivadas
        if self.metrics["inference_count"] > 0:
            self.metrics["avg_inference_time"] = (
                self.metrics["total_inference_time"] / self.metrics["inference_count"]
            )
            self.metrics["avg_tokens_per_inference"] = (
                self.metrics["total_tokens_generated"] / self.metrics["inference_count"]
            )
        
        return self.metrics
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retorna informações de status do especialista
        
        Returns:
            Dicionário com informações de status
        """
        status = {
            "expert_type": self.expert_type,
            "expert_id": self.expert_id,
            "is_initialized": self.is_initialized,
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "model_name": self.model_name,
            "device": self.device,
            "gpu_id": self.gpu_id,
            "quantization": self.quantization,
            "metrics": self.get_metrics(),
            "uptime_seconds": time.time() - self.initialization_time,
            "version": self.VERSION
        }
        
        if not self.is_initialized and self.initialization_error:
            status["initialization_error"] = self.initialization_error
            
        return status
    
    def reload_model(self) -> bool:
        """
        Recarrega o modelo, útil após descarregamento ou para mudar para outra GPU.
        
        Returns:
            True se o modelo foi recarregado com sucesso, False caso contrário
        """
        try:
            if self.model_loaded:
                self.logger.warning("Tentativa de recarregar modelo que já está carregado")
                return True
                
            self.logger.info(f"Recarregando modelo {self.model_name}...")
            
            # Registrar tempo de início para medir performance
            start_time = time.time()
            
            # Tentar carregar o modelo
            self._load_model()
            
            # Verificar se o carregamento foi bem-sucedido
            if not self.model_loaded:
                self.logger.error("Falha ao recarregar modelo")
                return False
                
            # Registrar tempo de carregamento
            load_time = time.time() - start_time
            self.logger.info(f"Modelo recarregado com sucesso em {load_time:.2f} segundos")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao recarregar modelo: {e}")
            return False
            
    def move_to_gpu(self, target_gpu_id: int) -> bool:
        """
        Move o modelo para outra GPU.
        
        Args:
            target_gpu_id: ID da GPU de destino
            
        Returns:
            True se a operação foi bem-sucedida, False caso contrário
        """
        try:
            if not self.model_loaded:
                self.logger.warning("Tentativa de mover modelo que não está carregado")
                return False
                
            # Registrar detalhes
            original_gpu = self.gpu_id
            self.logger.info(f"Movendo modelo da GPU {original_gpu} para GPU {target_gpu_id}")
            
            # Verificar se a GPU alvo existe
            try:
                if not torch.cuda.is_available():
                    self.logger.error("CUDA não disponível para mover modelo")
                    return False
                    
                if target_gpu_id >= torch.cuda.device_count():
                    self.logger.error(f"GPU alvo {target_gpu_id} não existe. Total de GPUs: {torch.cuda.device_count()}")
                    return False
            except Exception as e:
                self.logger.error(f"Erro ao verificar disponibilidade da GPU: {e}")
                return False
                
            # Verificar memória disponível na GPU de destino se tiver monitor
            if self.gpu_monitor:
                metrics = self.gpu_monitor.get_current_metrics()
                if target_gpu_id in metrics:
                    free_memory = metrics[target_gpu_id].get("memory_free_mb", 0)
                    # Estimar tamanho do modelo (4 bytes por parâmetro)
                    model_size_mb = getattr(self, "model_size_mb", 5000)  # Valor padrão se não tiver estimativa precisa
                    
                    if free_memory < model_size_mb * 1.2:  # 20% de margem de segurança
                        self.logger.error(f"GPU {target_gpu_id} não tem memória suficiente: {free_memory:.2f}MB livre, necessário ~{model_size_mb * 1.2:.2f}MB")
                        return False
            
            # Registrar uso de memória antes da operação
            if hasattr(torch.cuda, "memory_allocated") and self.gpu_id is not None:
                try:
                    mem_before = torch.cuda.memory_allocated(self.gpu_id) / (1024**3)
                    self.logger.debug(f"Memória GPU {self.gpu_id} antes da operação: {mem_before:.2f} GB")
                except Exception:
                    pass
            
            # Realizar a movimentação do modelo
            target_device = f"cuda:{target_gpu_id}"
            self.model.to(target_device)
            
            # Atualizar o ID da GPU
            self.gpu_id = target_gpu_id
            self.device = target_device
            
            # Limpar cache CUDA na GPU original
            torch.cuda.empty_cache()
            
            self.logger.info(f"Modelo movido com sucesso para GPU {target_gpu_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao mover modelo para GPU {target_gpu_id}: {e}")
            return False
