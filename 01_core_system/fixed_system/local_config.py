#!/usr/bin/env python3
"""
Configuração para rodar modelos locais com llama-cpp-python
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Tenta importar llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("⚠️ llama-cpp-python não instalado. Instale com:")
    print("pip install llama-cpp-python")

logger = logging.getLogger(__name__)

class LocalModelServer:
    """Servidor simples para modelos GGUF locais"""
    
    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 0):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.model: Optional[Llama] = None
        
    def load_model(self):
        """Carrega o modelo GGUF"""
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python não está instalado!")
            
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
            
        logger.info(f"🔄 Carregando modelo: {self.model_path.name}")
        
        try:
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            logger.info("✅ Modelo carregado com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Gera resposta do modelo"""
        if not self.model:
            raise RuntimeError("Modelo não carregado!")
            
        try:
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            logger.error(f"❌ Erro na geração: {e}")
            return ""

class ModelManager:
    """Gerenciador de modelos locais"""
    
    def __init__(self):
        self.models_dir = Path.home() / ".cache" / "models"
        self.available_models = self._scan_models()
        
    def _scan_models(self) -> Dict[str, Path]:
        """Busca modelos GGUF disponíveis"""
        models = {}
        
        if self.models_dir.exists():
            for file in self.models_dir.glob("*.gguf"):
                model_name = file.stem.lower()
                models[model_name] = file
                logger.info(f"📦 Modelo encontrado: {model_name}")
                
        return models
    
    def get_model_path(self, name: str) -> Optional[Path]:
        """Retorna caminho do modelo"""
        # Busca exata
        if name in self.available_models:
            return self.available_models[name]
            
        # Busca parcial
        for model_name, path in self.available_models.items():
            if name.lower() in model_name:
                return path
                
        return None
    
    def list_models(self):
        """Lista modelos disponíveis"""
        if not self.available_models:
            logger.warning("⚠️ Nenhum modelo GGUF encontrado!")
            logger.info(f"📁 Diretório de busca: {self.models_dir}")
            return
            
        logger.info("📋 Modelos disponíveis:")
        for i, (name, path) in enumerate(self.available_models.items(), 1):
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"  {i}. {name} ({size_mb:.1f} MB)")

def test_local_model():
    """Testa modelo local"""
    manager = ModelManager()
    manager.list_models()
    
    if not manager.available_models:
        return
        
    # Pega primeiro modelo disponível
    model_name = list(manager.available_models.keys())[0]
    model_path = manager.get_model_path(model_name)
    
    logger.info(f"\n🧪 Testando modelo: {model_name}")
    
    server = LocalModelServer(model_path)
    if server.load_model():
        response = server.generate("O que é inteligência artificial? Responda em uma frase.")
        logger.info(f"🤖 Resposta: {response}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    test_local_model()