"""
Transformers Model Config Generator

Este script detecta modelos Transformers disponíveis no cache do Hugging Face
ou em diretórios especificados e gera uma configuração apropriada.
"""

import os
import sys
import json
import glob
from pathlib import Path

def find_huggingface_models(extra_dirs=None):
    """
    Localiza modelos Hugging Face no cache padrão ou em diretórios adicionais
    
    Args:
        extra_dirs: Lista de diretórios adicionais para procurar
        
    Returns:
        Lista de tuplas (nome_modelo, caminho)
    """
    models = []
    
    # Diretórios para buscar
    search_dirs = []
    
    # Cache padrão do Hugging Face
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        search_dirs.append(hf_cache)
        
    # Adiciona diretórios extras
    if extra_dirs:
        for dir_path in extra_dirs:
            path = Path(dir_path).expanduser()
            if path.exists():
                search_dirs.append(path)
    
    # Procura em cada diretório
    for base_dir in search_dirs:
        print(f"Procurando modelos em: {base_dir}")
        
        # Busca padrão no cache HF
        if str(base_dir).endswith("huggingface/hub"):
            # No cache do HF, os modelos são armazenados em subdiretórios com nome específico
            model_dirs = list(base_dir.glob("models--*"))
            
            for model_dir in model_dirs:
                # Extrai o nome do modelo do nome do diretório
                model_name = model_dir.name.replace("models--", "").replace("--", "/")
                
                # Verifica se é um modelo de linguagem
                is_llm = False
                snapshots = list(model_dir.glob("snapshots/*"))
                
                if snapshots:
                    # Verifica arquivos comuns em modelos de linguagem
                    llm_files = [
                        "config.json", 
                        "pytorch_model.bin", 
                        "model.safetensors", 
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "generation_config.json"
                    ]
                    
                    for snapshot in snapshots:
                        if any((snapshot / file).exists() for file in llm_files):
                            is_llm = True
                            
                            # Usa o último snapshot
                            models.append((model_name, str(snapshot)))
                            print(f"Encontrado: {model_name} em {snapshot}")
                            break
        
        # Busca em diretórios personalizados
        else:
            # Em diretórios personalizados, os modelos podem estar em subdiretórios diretos
            for subdir in base_dir.iterdir():
                if subdir.is_dir():
                    # Verifica se parece um modelo de linguagem
                    llm_files = [
                        "config.json", 
                        "pytorch_model.bin",
                        "model.safetensors",
                        "tokenizer.json", 
                        "tokenizer_config.json"
                    ]
                    
                    if any((subdir / file).exists() for file in llm_files):
                        # Usa o nome do diretório como nome do modelo
                        model_name = subdir.name
                        models.append((model_name, str(subdir)))
                        print(f"Encontrado: {model_name} em {subdir}")
    
    return models

def create_config_for_transformers(models):
    """
    Cria um arquivo de configuração para modelos Transformers
    
    Args:
        models: Lista de tuplas (nome_modelo, caminho)
        
    Returns:
        Caminho para o arquivo de configuração gerado
    """
    model_configs = []
    default_model = None
    
    for model_name, model_path in models:
        # Configuração específica baseada no nome do modelo
        capabilities = ["general"]
        context_size = 4096
        
        # Determina capacidades e tamanho de contexto com base no nome
        if "mistral" in model_name.lower():
            capabilities = ["precise", "general", "creative"]
            context_size = 8192
            # Bom modelo padrão
            default_model = model_name
        elif "llama" in model_name.lower() and "chat" in model_name.lower():
            capabilities = ["chat", "general", "creative"]
            context_size = 4096
        elif "code" in model_name.lower() or "codellama" in model_name.lower():
            capabilities = ["code", "technical"]
            context_size = 4096
        elif "phi" in model_name.lower():
            capabilities = ["fast", "general"]
            context_size = 2048
            # Phi-2 é um bom modelo padrão por ser rápido
            if "phi-2" in model_name.lower():
                default_model = model_name
        
        # Cria configuração para este modelo
        model_configs.append({
            "name": model_name.replace("/", "-").lower(),
            "path": model_path,
            "model_type": "transformers",
            "capabilities": capabilities,
            "min_prompt_tokens": 0,
            "max_prompt_tokens": context_size,
            "params": {
                "device_map": "auto",
                "trust_remote_code": False,
                "low_cpu_mem_usage": True
            }
        })
    
    # Define modelo padrão se necessário
    if not default_model and model_configs:
        default_model = model_configs[0]["name"]
    
    # Cria objeto de configuração
    config = {
        "models": model_configs,
        "default_model": default_model
    }
    
    # Escreve arquivo de configuração
    config_path = "transformers_models.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    return config_path

def main():
    """Função principal"""
    # Diretórios adicionais para buscar
    extra_dirs = [
        Path.cwd() / "models_hf",
        Path.cwd().parent / "modelos_hf",
        "C:/Users/anapa/SuperIA/EzioFilhoUnified/models_hf",
        "C:/Users/anapa/SuperIA/modelos_hf",
    ]
    
    # Busca modelos
    print("Buscando modelos Transformers...")
    models = find_huggingface_models(extra_dirs)
    
    if not models:
        print("\nNenhum modelo Transformers encontrado.")
        print("Opções:")
        print("1. Verifique se os modelos estão no cache do Hugging Face:")
        print("   C:\\Users\\anapa\\.cache\\huggingface\\hub")
        print("2. Especifique um diretório personalizado com modelos:")
        print("   python config_manual.py --dir \"C:\\caminho\\para\\modelos\"")
        print("3. Execute o sistema em modo de simulação:")
        print("   python add_simulate_mode.py")
        print("   python main.py --simulate --prompt \"Sua pergunta aqui\"")
        return 1
    
    # Cria configuração
    config_path = create_config_for_transformers(models)
    
    print(f"\n✅ Configuração gerada: {config_path}")
    print(f"Modelos configurados: {len(models)}")
    print("\nPara usar esta configuração:")
    print(f"python main.py --config {config_path} --prompt \"Sua pergunta aqui\"")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())