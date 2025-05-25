#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Model Manager for LLM Selection and Configuration
---------------------------------------------------------
Ferramenta para descoberta, configuração e gerenciamento de modelos LLM locais.
Suporta modelos nos formatos GGUF (llama.cpp) e Transformers (HuggingFace).

Uso:
  python advanced_model_manager.py discover --dirs "caminho/para/modelos" [--recursive] [--output "config.json"]
  python advanced_model_manager.py list --config "config.json"
  python advanced_model_manager.py validate --config "config.json"
  python advanced_model_manager.py test --config "config.json" --model "nome_modelo"
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Configuração de logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s – %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("model_manager")

# Constantes
GGUF_EXTENSIONS = [".gguf", ".bin"]
TRANSFORMERS_INDICATORS = ["config.json", "pytorch_model.bin", "model.safetensors"]
MODEL_TYPES = ["gguf", "transformers"]
DEFAULT_CONFIG = {
    "name": "",
    "path": "",
    "type": "",
    "context_length": 4096,
    "capabilities": ["general"],
    "performance": 5,
    "priority": 50
}

def setup_logging(log_file: Optional[str] = None) -> None:
    """Configura o sistema de logging com opção de arquivo."""
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)
        logger.info(f"Logging também para arquivo: {log_file}")

def is_gguf_model(path: Path) -> bool:
    """Verifica se o caminho é um modelo GGUF."""
    return path.is_file() and any(str(path).lower().endswith(ext) for ext in GGUF_EXTENSIONS)

def is_transformers_model(path: Path) -> bool:
    """Verifica se o diretório contém um modelo Transformers."""
    if not path.is_dir():
        return False
    
    files = [f.name for f in path.iterdir() if f.is_file()]
    return any(indicator in files for indicator in TRANSFORMERS_INDICATORS)

def detect_model_type(path: Path) -> Optional[str]:
    """Detecta o tipo de modelo baseado no caminho."""
    if is_gguf_model(path):
        return "gguf"
    elif is_transformers_model(path):
        return "transformers"
    return None

def guess_model_name(path: Path) -> str:
    """Tenta inferir um nome adequado para o modelo."""
    if path.is_file():
        # Para arquivos GGUF, use o nome do arquivo sem a extensão
        base_name = path.stem
        # Remove sufixos comuns de quantização
        for suffix in ["-q4_k_m", "-q5_k_m", "-q8_0", "-q4_0", "-q4_k", "-q6_k"]:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
        return base_name.replace("-", "_").lower()
    else:
        # Para diretórios de modelos Transformers, use o último componente do caminho
        return path.name.replace("-", "_").lower()

def discover_models(directories: List[str], recursive: bool = False) -> List[Dict[str, Any]]:
    """
    Descobre modelos nos diretórios especificados.
    
    Args:
        directories: Lista de diretórios para procurar modelos
        recursive: Se deve procurar recursivamente em subdiretórios
        
    Returns:
        Lista de configurações de modelos descobertos
    """
    models = []
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Diretório não encontrado: {directory}")
            continue
            
        logger.info(f"Procurando modelos em: {directory}")
        
        # Primeiro, verifica se o próprio diretório é um modelo Transformers
        model_type = detect_model_type(dir_path)
        if model_type == "transformers":
            model_name = guess_model_name(dir_path)
            logger.info(f"✓ Modelo Transformers encontrado: {model_name} em {dir_path}")
            models.append({
                **DEFAULT_CONFIG,
                "name": model_name,
                "path": str(dir_path),
                "type": "transformers"
            })
            
        # Procura por arquivos e subdiretórios
        if dir_path.is_dir():
            # Função recursiva para explorar diretórios
            def explore_dir(current_path: Path, depth: int = 0):
                # Limita a profundidade da recursão para evitar problemas
                if depth > 10:  
                    return
                
                for item in current_path.iterdir():
                    # Verifica se é um arquivo GGUF
                    if is_gguf_model(item):
                        model_name = guess_model_name(item)
                        logger.info(f"✓ Modelo GGUF encontrado: {model_name} em {item}")
                        models.append({
                            **DEFAULT_CONFIG,
                            "name": model_name,
                            "path": str(item),
                            "type": "gguf"
                        })
                    
                    # Verifica se é um diretório Transformers
                    elif item.is_dir() and is_transformers_model(item):
                        model_name = guess_model_name(item)
                        logger.info(f"✓ Modelo Transformers encontrado: {model_name} em {item}")
                        models.append({
                            **DEFAULT_CONFIG,
                            "name": model_name,
                            "path": str(item),
                            "type": "transformers"
                        })
                    
                    # Recursão para subdiretórios se habilitado
                    elif recursive and item.is_dir():
                        explore_dir(item, depth + 1)
            
            # Inicia a exploração do diretório
            explore_dir(dir_path)
    
    return models

def enrich_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enriquece a configuração do modelo com informações adicionais baseadas em heurísticas.
    
    Args:
        config: Configuração básica do modelo
        
    Returns:
        Configuração enriquecida com valores estimados
    """
    model_name = config["name"].lower()
    model_path = config["path"]
    
    # Estima o tamanho do contexto baseado no nome do modelo
    if "32k" in model_name or "32000" in model_name:
        config["context_length"] = 32000
    elif "16k" in model_name or "16000" in model_name:
        config["context_length"] = 16000
    elif "8k" in model_name or "8000" in model_name:
        config["context_length"] = 8000
    
    # Atribui capacidades baseadas no nome do modelo
    capabilities = ["general"]
    
    if any(x in model_name for x in ["code", "coder", "coding", "starcoder", "wizard-coder"]):
        capabilities.append("code")
        
    if any(x in model_name for x in ["instruct", "chat", "assistant"]):
        capabilities.append("chat")
        
    if any(x in model_name for x in ["math", "qwen"]):
        capabilities.append("math")

    config["capabilities"] = list(set(capabilities))  # Remove duplicatas
    
    # Estima a performance com base no tamanho do arquivo (somente para GGUF)
    if config["type"] == "gguf":
        path = Path(model_path)
        if path.exists() and path.is_file():
            size_gb = path.stat().st_size / (1024 * 1024 * 1024)
            
            if size_gb < 2:  # Modelos pequenos (< 2GB)
                config["performance"] = 3
                config["priority"] = 80  # Prioridade alta por ser rápido
            elif size_gb < 5:  # Modelos médios (2-5GB)
                config["performance"] = 5
                config["priority"] = 60
            elif size_gb < 10:  # Modelos grandes (5-10GB)
                config["performance"] = 7
                config["priority"] = 40
            else:  # Modelos muito grandes (>10GB)
                config["performance"] = 9
                config["priority"] = 20  # Prioridade baixa por ser lento
    
    # Ajusta prioridade com base no nome do modelo
    if "phi" in model_name or "gemma" in model_name:
        config["priority"] += 10  # Modelos mais recentes têm prioridade maior
        
    return config

def save_config(models: List[Dict[str, Any]], output_file: str) -> None:
    """
    Salva a configuração de modelos em um arquivo JSON.
    
    Args:
        models: Lista de configurações de modelos
        output_file: Caminho para o arquivo de saída
    """
    # Enriquece cada configuração
    enriched_models = [enrich_model_config(model) for model in models]
    
    # Ordena por prioridade (maior primeiro)
    sorted_models = sorted(enriched_models, key=lambda x: x.get("priority", 0), reverse=True)
    
    # Cria o objeto de configuração com metadata
    config = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "models_count": len(sorted_models),
            "version": "1.0"
        },
        "models": sorted_models
    }
    
    # Garante que o diretório exista
    output_path = Path(output_file)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Salva o arquivo
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Configuração salva em: {output_file}")
    logger.info(f"Total de modelos encontrados: {len(sorted_models)}")

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Carrega a configuração de modelos de um arquivo JSON.
    
    Args:
        config_file: Caminho para o arquivo de configuração
        
    Returns:
        Configuração carregada
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Arquivo de configuração não encontrado: {config_file}")
        return {"metadata": {}, "models": []}
    except json.JSONDecodeError:
        logger.error(f"Erro ao decodificar JSON do arquivo: {config_file}")
        return {"metadata": {}, "models": []}

def validate_models(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Valida se os modelos na configuração existem e são acessíveis.
    
    Args:
        config: Configuração de modelos
        
    Returns:
        Lista de modelos válidos
    """
    valid_models = []
    
    for model in config.get("models", []):
        path = Path(model["path"])
        model_type = model["type"]
        
        if model_type == "gguf" and path.is_file() and path.exists():
            logger.info(f"✓ Modelo GGUF válido: {model['name']} em {path}")
            valid_models.append(model)
        elif model_type == "transformers" and path.is_dir() and path.exists() and is_transformers_model(path):
            logger.info(f"✓ Modelo Transformers válido: {model['name']} em {path}")
            valid_models.append(model)
        else:
            logger.warning(f"✗ Modelo inválido ou não encontrado: {model['name']} em {path}")
    
    return valid_models

def list_models(config: Dict[str, Any]) -> None:
    """
    Lista os modelos na configuração em formato tabular.
    
    Args:
        config: Configuração de modelos
    """
    models = config.get("models", [])
    
    if not models:
        logger.info("Nenhum modelo encontrado na configuração")
        return
    
    # Determina o tamanho máximo de cada coluna
    name_width = max(len(model["name"]) for model in models) + 2
    type_width = max(len(model["type"]) for model in models) + 2
    ctx_width = len("Contexto") + 2
    perf_width = len("Perf.") + 2
    path_width = min(50, max(len(str(model["path"])) for model in models) + 2)
    
    # Imprime o cabeçalho
    header = (
        f"{'Nome'.ljust(name_width)} "
        f"{'Tipo'.ljust(type_width)} "
        f"{'Contexto'.ljust(ctx_width)} "
        f"{'Perf.'.ljust(perf_width)} "
        f"{'Caminho'.ljust(path_width)} "
        f"Capacidades"
    )
    separator = "-" * (name_width + type_width + ctx_width + perf_width + path_width + 20)
    
    print("\nModelos disponíveis:")
    print(separator)
    print(header)
    print(separator)
    
    # Imprime cada modelo
    for model in models:
        path_str = str(model["path"])
        if len(path_str) > path_width - 2:
            path_str = path_str[:path_width - 5] + "..."
            
        row = (
            f"{model['name'].ljust(name_width)} "
            f"{model['type'].ljust(type_width)} "
            f"{str(model.get('context_length', '-')).ljust(ctx_width)} "
            f"{str(model.get('performance', '-')).ljust(perf_width)} "
            f"{path_str.ljust(path_width)} "
            f"{', '.join(model.get('capabilities', []))}"
        )
        print(row)
    
    print(separator)
    print(f"Total: {len(models)} modelo(s)")
    
    # Mostra metadata
    metadata = config.get("metadata", {})
    if metadata:
        print("\nMetadados:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

def test_model(config: Dict[str, Any], model_name: str, prompt: str = "Explique o que é um modelo de linguagem.") -> None:
    """
    Testa um modelo específico carregando-o e gerando uma resposta.
    
    Args:
        config: Configuração de modelos
        model_name: Nome do modelo a ser testado
        prompt: Prompt para testar o modelo
    """
    models = config.get("models", [])
    target_model = None
    
    for model in models:
        if model["name"].lower() == model_name.lower():
            target_model = model
            break
    
    if not target_model:
        logger.error(f"Modelo não encontrado: {model_name}")
        return
    
    try:
        # Aqui você poderia integrar com seu wrapper de modelo
        logger.info(f"Testando modelo: {target_model['name']} ({target_model['type']})")
        logger.info(f"Caminho: {target_model['path']}")
        logger.info(f"Este é apenas um teste simulado, não está realmente carregando o modelo.")
        logger.info(f"Para testar o modelo real, utilize seu LocalModelWrapper do 'core/local_model_wrapper.py'")
        
        # Implementação simulada
        print(f"\nPrompt: {prompt}")
        print("\nSimulando resposta (não é gerada pelo modelo real)...")
        print(f"Este é um teste simulado do modelo {target_model['name']}.")
        print("Para obter respostas reais, integre este gerenciador com seu LocalModelWrapper.")
        
    except Exception as e:
        logger.error(f"Erro ao testar modelo: {e}")

def main() -> None:
    """Função principal do script."""
    parser = argparse.ArgumentParser(description="Gerenciador avançado de modelos LLM.")
    subparsers = parser.add_subparsers(dest="command", help="Comando a ser executado")
    
    # Comando: discover
    discover_parser = subparsers.add_parser("discover", help="Descobre modelos em diretórios")
    discover_parser.add_argument("--dirs", nargs="+", required=True, help="Diretórios para procurar modelos")
    discover_parser.add_argument("--recursive", action="store_true", help="Procurar recursivamente em subdiretórios")
    discover_parser.add_argument("--output", default="models_config.json", help="Arquivo de saída para a configuração")
    discover_parser.add_argument("--log-file", help="Arquivo para salvar logs")
    
    # Comando: list
    list_parser = subparsers.add_parser("list", help="Lista modelos configurados")
    list_parser.add_argument("--config", required=True, help="Arquivo de configuração")
    list_parser.add_argument("--log-file", help="Arquivo para salvar logs")
    
    # Comando: validate
    validate_parser = subparsers.add_parser("validate", help="Valida modelos configurados")
    validate_parser.add_argument("--config", required=True, help="Arquivo de configuração")
    validate_parser.add_argument("--output", help="Arquivo de saída para configuração validada")
    validate_parser.add_argument("--log-file", help="Arquivo para salvar logs")
    
    # Comando: test
    test_parser = subparsers.add_parser("test", help="Testa um modelo específico")
    test_parser.add_argument("--config", required=True, help="Arquivo de configuração")
    test_parser.add_argument("--model", required=True, help="Nome do modelo a ser testado")
    test_parser.add_argument("--prompt", default="Explique o que é um modelo de linguagem.", help="Prompt para testar")
    test_parser.add_argument("--log-file", help="Arquivo para salvar logs")
    
    args = parser.parse_args()
    
    # Configura o logging
    if hasattr(args, "log_file") and args.log_file:
        setup_logging(args.log_file)
    
    # Executa o comando especificado
    if args.command == "discover":
        logger.info(f"Iniciando descoberta de modelos em: {args.dirs}")
        models = discover_models(args.dirs, args.recursive)
        save_config(models, args.output)
        
    elif args.command == "list":
        config = load_config(args.config)
        list_models(config)
        
    elif args.command == "validate":
        config = load_config(args.config)
        valid_models = validate_models(config)
        
        if args.output:
            # Cria nova configuração apenas com modelos válidos
            new_config = {
                "metadata": {
                    **config.get("metadata", {}),
                    "validated_at": datetime.now().isoformat(),
                    "original_count": len(config.get("models", [])),
                    "valid_count": len(valid_models)
                },
                "models": valid_models
            }
            save_config(valid_models, args.output)
            
        logger.info(f"Total de modelos: {len(config.get('models', []))}")
        logger.info(f"Modelos válidos: {len(valid_models)}")
        
    elif args.command == "test":
        config = load_config(args.config)
        test_model(config, args.model, args.prompt)
        
    else:
        parser.print_help()
        
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Operação interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erro não tratado: {e}", exc_info=True)
        sys.exit(1)