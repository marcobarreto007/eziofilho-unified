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

    def __init__(self, 
                 expert_type: str,
                 config_path: Optional[Path] = None, 
                 gpu_id: Optional[int] = None,
                 model_path_override: Optional[str] = None,
                 system_message: Optional[str] = None,
                 quantization: Optional[str] = None):
        self.expert_type = expert_type
        self.expert_id = f"{expert_type}_{self._generate_id()}"
        self.logger = logging.getLogger(f"Expert_{self.expert_id}")

        config_path = config_path or Path("models_config.json")
        self.config = self._load_config(config_path)

        self.gpu_id = self._select_gpu(gpu_id)
        self.device = f"cuda:{self.gpu_id}" if self.gpu_id is not None else "cpu"
        self.system_message = system_message

        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.model_name = "unknown"

        model_config = self.config.get("models", {}).get(expert_type, {})
        self.model_path = model_path_override or model_config.get("path")
        self.quantization = quantization or model_config.get("quantization")

        self.metrics = {
            "model_load_time": 0,
            "inference_count": 0,
            "total_inference_time": 0,
            "total_tokens_generated": 0,
            "errors": 0,
            "oom_events": 0
        }

        if self.model_path:
            self._load_model()

    def _generate_id(self) -> str:
        import uuid
        return uuid.uuid4().hex[:8]

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"📄 Configuração carregada de {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar configuração: {e}")
            return {"models": {}}

    def _select_gpu(self, gpu_id: Optional[int]) -> Optional[int]:
        if not torch.cuda.is_available():
            return None
        if gpu_id is not None and 0 <= gpu_id < torch.cuda.device_count():
            return gpu_id

        best_gpu = None
        max_free = 0
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            free_mem = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            if free_mem > max_free:
                max_free = free_mem
                best_gpu = i
        return best_gpu

    def _load_model(self) -> None:
        start_time = time.time()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.logger.info(f"⏳ Carregando modelo: {self.model_path}")

            kwargs = {
                "device_map": self.device,
                "torch_dtype": torch.float16
            }
            if self.quantization == "4bit":
                kwargs["load_in_4bit"] = True
            elif self.quantization == "8bit":
                kwargs["load_in_8bit"] = True

            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model_loaded = True
            self.model_name = Path(self.model_path).name

            if self.gpu_id is not None:
                used_mem = torch.cuda.memory_allocated(self.gpu_id) / (1024**3)
                self.logger.info(f"✅ Modelo carregado, usando {used_mem:.2f} GB na GPU")
            self.metrics["model_load_time"] = time.time() - start_time
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar modelo: {e}")
            self.model_loaded = False
            self.metrics["errors"] += 1

    def analyze(self, text: str, max_tokens: int = 512, 
                context: Optional[str] = None, temperature: float = 0.1) -> Dict[str, Any]:
        if not self.model_loaded:
            return {"status": "error", "error": "Model not loaded", "response": None}

        self.metrics["inference_count"] += 1
        start_time = time.time()

        try:
            prompt = f"{context}\n\n{text}" if context else text
            if self.system_message:
                prompt = self.system_message.replace("<text>", prompt) if "<text>" in self.system_message else f"{self.system_message}\n\n{prompt}"

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_tokens = len(inputs.input_ids[0])

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    top_p=0.95,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            output_text = self.tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True)
            tokens_generated = len(outputs[0]) - input_tokens
            latency = time.time() - start_time

            self.metrics["total_inference_time"] += latency
            self.metrics["total_tokens_generated"] += tokens_generated

            return {
                "status": "success",
                "response": output_text,
                "tokens_prompt": input_tokens,
                "tokens_generated": tokens_generated,
                "latency": latency
            }
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.error("❌ GPU out of memory")
                self.metrics["oom_events"] += 1
            self.metrics["errors"] += 1
            return {"status": "error", "error": str(e), "response": None}
        except Exception as e:
            self.logger.error(f"❌ Erro na inferência: {e}")
            self.metrics["errors"] += 1
            return {"status": "error", "error": str(e), "response": None}

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics
