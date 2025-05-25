"""
EzioBaseExpert - Base class for all specialized experts in the EzioFilho system
"""
import os
import json
import time
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [%(name)s] | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class EzioBaseExpert:
    """Base class for all Ezio experts"""
    
    # Versão base do framework de especialistas
    VERSION = "2.0.0"
    
    def __init__(self, 
                expert_type: str,
                config_path: Optional[Path] = None, 
                gpu_id: Optional[int] = None,
                model_path_override: Optional[str] = None,
                system_message: Optional[str] = None,
                quantization: Optional[str] = None):
        """Initialize the expert
        
        Args:
            expert_type: Type of expert (sentiment, factcheck, etc.)
            config_path: Path to config file
            gpu_id: GPU ID to use
            model_path_override: Override model path from config
            system_message: System message for the expert
            quantization: Quantization method (4bit, 8bit, None)
        """
        # Status tracking para o orquestrador
        self.is_initialized = False
        self.initialization_error = None
        self.initialization_time = time.time()
        
        # Basic expert properties
        self.expert_type = expert_type
        self.expert_id = f"{expert_type}_{self._generate_id()}"
        self.logger = logging.getLogger(f"Expert_{self.expert_id}")
        
        # Load configuration
        config_path = config_path or Path("models_config.json")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self.logger.info(f"📄 Configuração carregada para {expert_type}")
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar configuração: {e}")
            self.config = {"models": {}}
            self.initialization_error = f"Erro ao carregar configuração: {e}"
        
        # Set GPU device
        self.gpu_id = self._select_gpu(gpu_id)
        if self.gpu_id is not None:
            gpu_name = torch.cuda.get_device_name(self.gpu_id)
            gpu_mem = torch.cuda.get_device_properties(self.gpu_id).total_memory / (1024**3)
            self.logger.info(f"🖥️ GPU selecionada: {gpu_name} (ID: {self.gpu_id}, Free: {gpu_mem:.2f} GB)")
            self.device = f"cuda:{self.gpu_id}"
        else:
            self.logger.info(f"💻 Usando CPU para inferência")
            self.device = "cpu"
        
        # Set system message
        self.system_message = system_message
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.model_name = "unknown"
        
        # Get model path from config or override
        model_config = self.config.get("models", {}).get(expert_type, {})
        self.model_path = model_path_override or model_config.get("path", None)
        
        # Set quantization
        self.quantization = quantization or model_config.get("quantization", None)
        
        # Performance metrics
        self.metrics = {
            "model_load_time": 0,
            "inference_count": 0,
            "total_inference_time": 0,
            "total_tokens_generated": 0,
            "errors": 0,
            "oom_events": 0
        }
        
        # Try to load model
        if self.model_path:
            self._load_model()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for this expert instance"""
        import uuid
        return uuid.uuid4().hex[:8]
    
    def _select_gpu(self, gpu_id: Optional[int]) -> Optional[int]:
        """Select the best GPU or use the specified one
        
        Args:
            gpu_id: Specified GPU ID or None for auto-selection
            
        Returns:
            Selected GPU ID or None if no GPU available
        """
        if not torch.cuda.is_available():
            return None
            
        if gpu_id is not None:
            if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
                self.logger.warning(f"⚠️ GPU ID {gpu_id} inválido. Selecionando automaticamente.")
                gpu_id = None
            else:
                return gpu_id
                
        # Auto-select GPU with most free memory
        if gpu_id is None and torch.cuda.device_count() > 0:
            max_free = 0
            best_gpu = 0
            
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                free_mem = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                if free_mem > max_free:
                    max_free = free_mem
                    best_gpu = i
                    
            return best_gpu
            
        return None
    
    def _load_model(self) -> None:
        """Load the model from the specified path"""
        start_time = time.time()
        
        try:
            # Log model loading
            self.logger.info(f"⏳ Carregando modelo de {self.model_path}")
            
            # Log GPU memory before loading
            if self.gpu_id is not None:
                free_mem = (torch.cuda.get_device_properties(self.gpu_id).total_memory - 
                           torch.cuda.memory_allocated(self.gpu_id)) / (1024**3)
                self.logger.info(f"Memória GPU disponível antes de carregar: {free_mem:.2f} GB")
            
            # Import here to avoid loading transformers at module import time
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Quantization settings
            if self.quantization == "4bit":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=self.device,
                    torch_dtype=torch.float16,
                    load_in_4bit=True
                )
            elif self.quantization == "8bit":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=self.device,
                    torch_dtype=torch.float16,
                    load_in_8bit=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=self.device,
                    torch_dtype=torch.float16
                )
                
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Set model loaded flag
            self.model_loaded = True
            self.model_name = self.model_path.split("/")[-1]
            
            # Check memory usage after loading
            if self.gpu_id is not None:
                mem_used = torch.cuda.memory_allocated(self.gpu_id) / (1024**3)
                self.logger.info(f"✅ Modelo carregado, usando {mem_used:.2f} GB de memória GPU")
                
            # Update metrics
            self.metrics["model_load_time"] = time.time() - start_time
            
            # Log success
            self.logger.info(f"✅ Especialista inicializado com sucesso: {self.expert_type} ({self.model_path})")
            
            # Status para o orquestrador
            self.is_initialized = True
            self.initialization_error = None
            
        except Exception as e:
            error_msg = f"❌ Erro ao carregar modelo: {e}"
            self.logger.error(error_msg)
            self.model_loaded = False
            self.metrics["errors"] += 1
            
            # Status para o orquestrador
            self.is_initialized = False
            self.initialization_error = error_msg
    
    def analyze(self, 
               text: str,
               max_tokens: int = 512, 
               context: Optional[str] = None,
               temperature: float = 0.1) -> Dict[str, Any]:
        """Analyze text with the loaded model
        
        Args:
            text: Text to analyze
            max_tokens: Maximum tokens to generate
            context: Additional context
            temperature: Temperature for generation
            
        Returns:
            Dictionary with analysis results
        """
        # Check if model is loaded
        if not self.model_loaded:
            return {
                "status": "error",
                "error": "Model not loaded",
                "response": None
            }
            
        # Track metrics
        start_time = time.time()
        self.metrics["inference_count"] += 1
        
        try:
            # Prepare prompt
            if context:
                prompt = f"{context}\n\n{text}"
            else:
                prompt = text
                
            # Add system message if available
            if self.system_message:
                if "<text>" in self.system_message:
                    prompt = self.system_message.replace("<text>", prompt)
                else:
                    prompt = f"{self.system_message}\n\n{prompt}"
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_tokens = len(inputs.input_ids[0])
            
            # Generate response
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
                
            # Decode output
            output_text = self.tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True)
            
            # Calculate tokens generated
            tokens_generated = len(outputs[0]) - input_tokens
            
            # Update metrics
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
            # Check for out of memory error
            if "out of memory" in str(e).lower():
                self.logger.error(f"❌ GPU out of memory: {e}")
                self.metrics["oom_events"] += 1
            else:
                self.logger.error(f"❌ Runtime error: {e}")
                
            self.metrics["errors"] += 1
            
            return {
                "status": "error",
                "error": str(e),
                "response": None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error during inference: {e}")
            self.metrics["errors"] += 1
            
            return {
                "status": "error",
                "error": str(e),
                "response": None
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics
        
        Returns:
            Dictionary with metrics
        """
        return self.metrics
        
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the expert
        
        Returns:
            Dictionary with status information
        """
        status = {
            "expert_type": self.expert_type,
            "expert_id": self.expert_id,
            "is_initialized": self.is_initialized,
            "model_path": getattr(self, "model_path", "unknown"),
            "model_name": getattr(self, "model_name", "unknown"),
            "device": getattr(self, "device", "unknown"),
            "metrics": self.metrics,
            "uptime_seconds": time.time() - self.initialization_time,
            "version": getattr(self, "VERSION", "1.0.0")
        }
        
        if not self.is_initialized and self.initialization_error:
            status["initialization_error"] = self.initialization_error
            
        return status