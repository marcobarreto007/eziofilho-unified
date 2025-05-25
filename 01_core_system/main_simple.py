#!/usr/bin/env python3
"""
Sistema AutoGen Simplificado e Robusto
Versão 2.0 - Reescrito para máxima eficiência e confiabilidade
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class AutoGenSystem:
    """Sistema AutoGen simplificado e eficiente"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.absolute()
        self.models_cache = Path.home() / ".cache" / "models"
        self._setup_paths()
        
    def _setup_paths(self):
        """Configura caminhos do sistema"""
        sys.path.insert(0, str(self.project_root))
        
    def check_dependencies(self) -> bool:
        """Verifica dependências essenciais"""
        try:
            import autogen
            logger.info(f"AutoGen v{autogen.__version__} encontrado")
            return True
        except ImportError:
            logger.error("AutoGen não encontrado. Execute: pip install pyautogen")
            return False
    
    def get_available_models(self) -> List[Dict]:
        """Retorna lista de modelos disponíveis"""
        models = []
        
        # Modelos padrão conhecidos
        default_models = [
            {"name": "phi2", "file": "phi-2.gguf"},
            {"name": "mistral", "file": "mistral-7b.gguf"},
            {"name": "llama", "file": "llama-2-7b.gguf"},
        ]
        
        for model in default_models:
            model_path = self.models_cache / model["file"]
            if model_path.exists():
                models.append({
                    "name": model["name"],
                    "path": str(model_path),
                    "size": model_path.stat().st_size
                })
                logger.info(f"✓ Modelo encontrado: {model['name']}")
        
        if not models:
            logger.warning("Nenhum modelo encontrado no cache")
        
        return models
    
    def create_router(self) -> Optional[object]:
        """Cria roteador de modelos simples"""
        models = self.get_available_models()
        
        if not models:
            logger.error("Nenhum modelo disponível")
            return None
            
        # Router simples sem dependências externas
        class SimpleRouter:
            def __init__(self, models):
                self.models = {m["name"]: m for m in models}
                self.default = models[0]["name"]
                
            def get_model(self, name=None):
                return self.models.get(name or self.default)
                
        return SimpleRouter(models)
    
    def run_simple_chat(self, prompt: str = "Olá! Como você pode me ajudar?") -> bool:
        """Executa chat simples com AutoGen"""
        try:
            import autogen
            
            # Configuração mínima para funcionar
            config_list = [{
                "model": "gpt-3.5-turbo",
                "api_key": "sk-fake-key-for-local",
                "base_url": "http://localhost:1234/v1"
            }]
            
            # Agente assistente
            assistant = autogen.AssistantAgent(
                name="assistant",
                system_message="Você é um assistente útil.",
                llm_config={"config_list": config_list, "temperature": 0.7}
            )
            
            # Proxy do usuário (sem is_human para compatibilidade)
            user_proxy = autogen.UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False
            )
            
            logger.info(f"Enviando prompt: {prompt}")
            
            # Inicia conversa
            user_proxy.initiate_chat(
                assistant,
                message=prompt
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Erro no chat: {e}")
            return False
    
    def run_diagnostic(self) -> bool:
        """Executa diagnóstico completo do sistema"""
        logger.info("=== DIAGNÓSTICO DO SISTEMA ===")
        
        # 1. Verificar dependências
        if not self.check_dependencies():
            return False
        
        # 2. Verificar estrutura do projeto
        core_dir = self.project_root / "core"
        if core_dir.exists():
            core_files = list(core_dir.glob("*.py"))
            logger.info(f"Arquivos core encontrados: {len(core_files)}")
        else:
            logger.warning("Diretório core/ não encontrado")
        
        # 3. Verificar modelos
        models = self.get_available_models()
        logger.info(f"Modelos disponíveis: {len(models)}")
        
        # 4. Testar imports
        try:
            from core import model_router
            logger.info("✓ Imports do core funcionando")
        except ImportError:
            logger.warning("✗ Imports do core falharam")
        
        # 5. Verificar cache
        cache_dir = self.project_root / ".cache"
        if cache_dir.exists():
            logger.info("✓ Diretório cache encontrado")
        else:
            logger.warning("✗ Diretório cache não encontrado")
        
        logger.info("=== DIAGNÓSTICO COMPLETO ===")
        return True


def main():
    """Função principal simplificada"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema AutoGen Simplificado")
    parser.add_argument("--prompt", default="Explique o que é IA em 2 parágrafos.",
                       help="Prompt para o modelo")
    parser.add_argument("--diagnostic", action="store_true",
                       help="Executa diagnóstico do sistema")
    parser.add_argument("--debug", action="store_true",
                       help="Ativa logs detalhados")
    
    args = parser.parse_args()
    
    # Configura logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Inicializa sistema
    system = AutoGenSystem()
    
    try:
        # Modo diagnóstico
        if args.diagnostic:
            return 0 if system.run_diagnostic() else 1
        
        # Verifica dependências
        if not system.check_dependencies():
            logger.error("Dependências não satisfeitas")
            return 1
        
        # Executa chat
        logger.info("Iniciando sistema AutoGen...")
        success = system.run_simple_chat(args.prompt)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuário")
        return 0
    except Exception as e:
        logger.exception(f"Erro fatal: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())