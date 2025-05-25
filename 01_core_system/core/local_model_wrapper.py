#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LocalModelWrapper - Wrapper unificado para modelos de linguagem locais
----------------------------------------------------------------------
Interface simplificada para uso de modelos de linguagem em diferentes formatos,
com foco especial em modelos HuggingFace Transformers.
"""

import gc
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Configuração de logging com cores no console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s"
)
logger = logging.getLogger("LocalModelWrapper")

# Constantes
TRANSFORMERS_INDICATORS = ["config.json", "pytorch_model.bin", "model.safetensors"]
GGUF_EXTENSIONS = [".gguf", ".bin", ".ggml"]
CHAT_INDICATORS = ["chat", "instruct", "assistant", "phi", "qwen", "falcon", "llama2", "mistral"]

class LocalModelWrapper:
    """Interface unificada para modelos locais."""
    
    def __init__(
        self,
        model_path: str,
        model_type: Optional[str] = None,
        context_length: int = 4096,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        use_gpu: bool = True,
        trust_remote_code: bool = True,
        **kwargs
    ):
        """
        Inicializa o wrapper para um modelo local.
        
        Args:
            model_path: Caminho para o modelo (arquivo ou diretório)
            model_type: Tipo de modelo ('transformers' ou 'gguf')
            context_length: Comprimento máximo do contexto em tokens
            temperature: Temperatura para a geração de texto (0.0-1.0)
            max_tokens: Número máximo de tokens a gerar
            use_gpu: Se deve usar GPU quando disponível
            trust_remote_code: Confia em código remoto (para modelos HF)
        """
        # Configura atributos básicos
        self.model_path = os.path.expanduser(model_path)
        self.model_type = model_type.lower() if model_type else self._detect_model_type()
        self.context_length = context_length
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_gpu = use_gpu
        self.trust_remote_code = trust_remote_code
        
        # Armazena configuração adicional
        self.config = {
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 40),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "stop_sequences": kwargs.get("stop_sequences", []),
        }
        
        # Estado interno
        self.model = None
        self.tokenizer = None
        self.is_chat_model = self._is_chat_model()
        self.loaded = False
        
        # Estatísticas
        self.total_tokens = 0
        self.total_calls = 0
        self.last_used = 0
        
        logger.info(f"Wrapper inicializado para: {os.path.basename(self.model_path)}")
        logger.info(f"Tipo: {self.model_type}, Chat: {self.is_chat_model}")
    
    def _detect_model_type(self) -> str:
        """Detecta automaticamente o tipo de modelo com base no caminho."""
        path = Path(self.model_path)
        
        # Verifica se é um arquivo GGUF
        if path.is_file() and any(str(path).lower().endswith(ext) for ext in GGUF_EXTENSIONS):
            return "gguf"
        
        # Verifica se é um diretório de modelo HuggingFace
        if path.is_dir() and any((path / indicator).exists() for indicator in TRANSFORMERS_INDICATORS):
            return "transformers"
        
        # Fallback para transformers (mais compatível com o inventário de modelos)
        logger.info(f"Tipo de modelo não detectado explicitamente, assumindo 'transformers'")
        return "transformers"
    
    def _is_chat_model(self) -> bool:
        """Detecta se o modelo é um modelo de chat com base no nome/caminho."""
        model_path_lower = self.model_path.lower()
        return any(indicator in model_path_lower for indicator in CHAT_INDICATORS)
    
    def _check_transformers(self) -> bool:
        """Verifica se a biblioteca Transformers está instalada."""
        try:
            import transformers
            return True
        except ImportError:
            logger.error("Biblioteca 'transformers' não está instalada!")
            logger.error("Instale com: pip install transformers")
            return False
    
    def _check_llama_cpp(self) -> bool:
        """Verifica se llama-cpp-python está instalada."""
        try:
            import llama_cpp
            return True
        except ImportError:
            logger.error("Biblioteca 'llama-cpp-python' não está instalada!")
            logger.error("Instale com: pip install llama-cpp-python")
            return False
    
    def _format_prompt(self, prompt: str) -> str:
        """Formata o prompt de acordo com o tipo de modelo."""
        if not self.is_chat_model:
            return prompt
        
        # Detecta o tipo específico de modelo para formatação adequada
        model_name = os.path.basename(self.model_path).lower()
        
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
    
    def _load_transformers_model(self):
        """Carrega um modelo usando a biblioteca HuggingFace Transformers."""
        if not self._check_transformers():
            raise ImportError("Transformers não disponível")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # Determina o dispositivo a ser utilizado
        device = "cpu"
        if self.use_gpu:
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Usando GPU CUDA: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                logger.info("Usando GPU Apple Silicon MPS")
        
        # Suprime avisos específicos
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Carrega o tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=self.trust_remote_code,
                    use_fast=True
                )
            except Exception as e:
                logger.warning(f"Erro ao carregar tokenizer padrão: {e}")
                logger.info("Tentando com fallback para tokenizer básico...")
                
                # Tenta com outro tokenizer como fallback
                try:
                    from transformers import AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "microsoft/phi-2",  # Um tokenizer genérico como fallback
                        trust_remote_code=True
                    )
                except Exception as e2:
                    logger.error(f"Erro no tokenizer de fallback: {e2}")
                    raise
            
            # Configura padding se necessário
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Carrega o modelo
            try:
                # Tenta com carregamento otimizado primeiro
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto" if device != "cpu" else None,
                    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                    trust_remote_code=self.trust_remote_code,
                    low_cpu_mem_usage=True
                )
            except Exception as e:
                logger.warning(f"Erro no carregamento otimizado: {e}")
                logger.info("Tentando com configuração simples...")
                
                # Tenta com configuração básica
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                
                # Move para o dispositivo apropriado
                if device != "cpu":
                    self.model = self.model.to(device)
        
        self.loaded = True
        logger.info(f"Modelo Transformers carregado: {os.path.basename(self.model_path)}")
    
    def _load_gguf_model(self):
        """Carrega um modelo usando a biblioteca llama.cpp."""
        if not self._check_llama_cpp():
            raise ImportError("llama-cpp-python não disponível")
        
        import llama_cpp
        
        # Configuração para GPU
        n_gpu_layers = -1 if self.use_gpu else 0
        
        # Carrega o modelo
        self.model = llama_cpp.Llama(
            model_path=self.model_path,
            n_ctx=self.context_length,
            n_threads=os.cpu_count() // 2,  # Usa metade dos núcleos disponíveis
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        
        self.loaded = True
        logger.info(f"Modelo GGUF carregado: {os.path.basename(self.model_path)}")
    
    def load(self):
        """Carrega o modelo na memória."""
        if self.loaded:
            return True
        
        try:
            if self.model_type == "transformers":
                self._load_transformers_model()
            elif self.model_type == "gguf":
                self._load_gguf_model()
            else:
                raise ValueError(f"Tipo de modelo não suportado: {self.model_type}")
            
            return self.loaded
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            return False
    
    def unload(self):
        """Libera o modelo da memória."""
        if not self.loaded:
            return True
        
        try:
            # Limpa referências
            self.model = None
            self.tokenizer = None
            
            # Força liberação de memória
            gc.collect()
            
            # Libera memória CUDA se disponível
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            self.loaded = False
            logger.info(f"Modelo descarregado: {os.path.basename(self.model_path)}")
            return True
        except Exception as e:
            logger.error(f"Erro ao descarregar modelo: {str(e)}")
            return False
    
    def _generate_with_transformers(self, prompt: str) -> str:
        """Gera texto usando modelo HuggingFace."""
        import torch
        
        # Formata o prompt se necessário
        formatted_prompt = self._format_prompt(prompt)
        
        # Tokeniza o input
        input_ids = self.tokenizer.encode(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_length - self.max_tokens
        )
        
        # Move para GPU se modelo estiver na GPU
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        
        # Configura parâmetros de geração
        gen_params = {
            "temperature": max(0.1, self.temperature),  # Evita divisão por zero
            "top_p": self.config["top_p"],
            "top_k": self.config["top_k"],
            "max_new_tokens": self.max_tokens,
            "repetition_penalty": self.config["repetition_penalty"],
            "do_sample": True if self.temperature > 0 else False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Executa a geração
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                **gen_params
            )
        
        # Decodifica a saída
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove o prompt da saída
        if decoded_output.startswith(formatted_prompt):
            response = decoded_output[len(formatted_prompt):]
        else:
            response = decoded_output
        
        # Remove padrões típicos de terminação
        for stop in ["<|endoftext|>", "</s>", "<|im_end|>"]:
            if stop in response:
                response = response.split(stop)[0]
        
        return response.strip()
    
    def _generate_with_gguf(self, prompt: str) -> str:
        """Gera texto usando modelo GGUF com llama.cpp."""
        # Configura parâmetros de geração
        params = {
            "temperature": max(0.1, self.temperature),
            "top_p": self.config["top_p"],
            "top_k": self.config["top_k"],
            "max_tokens": self.max_tokens,
            "repeat_penalty": self.config["repetition_penalty"],
        }
        
        # Acrescenta stop_sequences se existirem
        if self.config["stop_sequences"]:
            params["stop"] = self.config["stop_sequences"]
        
        # Executa a geração
        result = self.model(prompt, **params)
        
        # Processa o resultado
        if isinstance(result, dict) and "choices" in result:
            return result["choices"][0]["text"].strip()
        elif isinstance(result, str):
            return result.strip()
        else:
            return str(result).strip()
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Gera texto com base no prompt fornecido.
        
        Args:
            prompt: Texto de entrada para o modelo
            **kwargs: Parâmetros opcionais que substituem os padrões
            
        Returns:
            Texto gerado pelo modelo
        """
        # Verifica prompt vazio
        if not prompt or not prompt.strip():
            return ""
        
        # Atualiza configuração com parâmetros opcionais
        gen_config = self.config.copy()
        gen_config.update(kwargs)
        
        # Carrega o modelo se necessário
        if not self.loaded:
            self.load()
        
        # Registra o uso
        self.total_calls += 1
        self.last_used = time.time()
        
        # Tempo inicial para métricas
        start_time = time.time()
        
        try:
            # Gera o texto com base no tipo de modelo
            if self.model_type == "transformers":
                response = self._generate_with_transformers(prompt)
            elif self.model_type == "gguf":
                response = self._generate_with_gguf(prompt)
            else:
                raise ValueError(f"Tipo de modelo não suportado: {self.model_type}")
            
            # Atualiza estatísticas
            self.total_tokens += len(response.split())
            generation_time = time.time() - start_time
            
            logger.info(f"Texto gerado em {generation_time:.2f}s ({len(response.split())} tokens)")
            return response
            
        except Exception as e:
            logger.error(f"Erro na geração de texto: {str(e)}")
            return f"[Erro na geração: {str(e)}]"
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do modelo."""
        return {
            "name": os.path.basename(self.model_path),
            "type": self.model_type,
            "loaded": self.loaded,
            "calls": self.total_calls,
            "tokens": self.total_tokens,
            "last_used": self.last_used
        }
    
    def __str__(self) -> str:
        """Representação em string."""
        return f"LocalModelWrapper({os.path.basename(self.model_path)}, {self.model_type})"


def create_model_wrapper(
    model_path: str,
    model_type: Optional[str] = None,
    **kwargs
) -> LocalModelWrapper:
    """
    Cria uma instância de LocalModelWrapper com os parâmetros fornecidos.
    
    Args:
        model_path: Caminho para o modelo
        model_type: Tipo de modelo (autodetectado se None)
        **kwargs: Parâmetros adicionais
        
    Returns:
        Instância configurada de LocalModelWrapper
    """
    return LocalModelWrapper(
        model_path=model_path,
        model_type=model_type,
        **kwargs
    )


if __name__ == "__main__":
    # Teste simples se executado diretamente
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        prompt = sys.argv[2] if len(sys.argv) > 2 else "Explique o que é inteligência artificial."
        
        try:
            wrapper = create_model_wrapper(model_path)
            print(f"Testando modelo: {wrapper}")
            response = wrapper.generate(prompt)
            print(f"\nPrompt: {prompt}")
            print(f"\nResposta:\n{response}")
        except Exception as e:
            print(f"Erro no teste: {e}")
    else:
        print("Uso: python local_model_wrapper.py <caminho_do_modelo> [prompt]")