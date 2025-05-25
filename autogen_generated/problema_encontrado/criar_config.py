"""
Model Config Generator - Configuração para modelos existentes
"""

import os
import sys
import json
from pathlib import Path

def create_transformers_config():
    """Cria configuração para modelos Transformers existentes"""
    
    # Diretórios específicos para procurar, em ordem de prioridade
    model_dirs = [
        Path("C:/Users/anapa/EzioFilhoUnified/modelos_hf"),
        Path("C:/Users/anapa/modelos_hf"),
        Path("C:/Users/anapa/.eziofilho/models/models--microsoft--phi-3-mini-4k-instruct/snapshots"),
        Path("C:/Users/anapa/models_cache/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots"),
        Path("C:/Users/anapa/phi-2/models--microsoft--phi-2/snapshots"),
        Path.home() / ".cache/huggingface/hub"
    ]
    
    # Modelos específicos a procurar, por ordem de prioridade
    target_models = [
        "microsoft--phi-3-mini-4k-instruct",
        "microsoft--phi-2",
        "Qwen--Qwen1.5-1.8B-Chat",
        "tiiuae--falcon-7b-instruct",
        "facebook--opt-125m"
    ]
    
    print("Procurando modelos nos diretórios especificados...")
    
    # Estrutura para armazenar os modelos encontrados: nome -> caminho
    models_found = {}
    
    # 1. Primeiro, procure pelos modelos-alvo nos diretórios específicos
    for model_dir in model_dirs:
        if not model_dir.exists():
            print(f"Diretório não encontrado: {model_dir}")
            continue
            
        print(f"Verificando: {model_dir}")
        
        # Se este é o diretório de cache do Hugging Face
        if str(model_dir).endswith("huggingface/hub"):
            # Procurar por diretórios models--*
            for target in target_models:
                pattern = f"models--{target.replace('--', '-')}"
                model_paths = list(model_dir.glob(pattern))
                
                if model_paths:
                    model_path = model_paths[0]
                    # Procurar por snapshots
                    snapshot_dir = model_path / "snapshots"
                    if snapshot_dir.exists():
                        snapshots = list(snapshot_dir.glob("*"))
                        if snapshots:
                            snapshot = sorted(snapshots)[-1]  # último snapshot
                            model_name = target.split("--")[-1]
                            models_found[model_name] = str(snapshot)
                            print(f"✓ Encontrado: {model_name} em {snapshot}")
        else:
            # Procurar por diretórios específicos em diretório customizado
            for target in target_models:
                # Tenta várias variações do nome
                variations = [
                    target,
                    target.replace("--", "/"),
                    target.split("--")[-1]
                ]
                
                for variation in variations:
                    model_path = model_dir / variation
                    if model_path.exists() and (model_path / "config.json").exists():
                        model_name = target.split("--")[-1]
                        models_found[model_name] = str(model_path)
                        print(f"✓ Encontrado: {model_name} em {model_path}")
                        break
    
    # Se não encontrou nenhum modelo específico, procure por qualquer modelo nos diretórios
    if not models_found:
        print("Nenhum modelo alvo encontrado. Procurando por qualquer modelo...")
        
        for model_dir in model_dirs:
            if not model_dir.exists():
                continue
                
            # Procure por qualquer diretório que contenha config.json
            for path in model_dir.glob("**/config.json"):
                model_path = path.parent
                if "snapshots" in str(model_path):
                    # Está em um snapshot, pega o nome do diretório pai
                    model_name = model_path.parent.parent.name.replace("models--", "").split("--")[-1]
                else:
                    # Pega o nome do diretório atual
                    model_name = model_path.name
                    
                models_found[model_name] = str(model_path)
                print(f"✓ Encontrado: {model_name} em {model_path}")
    
    # Cria a configuração para os modelos encontrados
    model_configs = []
    default_model = None
    
    for model_name, model_path in models_found.items():
        # Define capacidades e contexto baseado no nome do modelo
        capabilities = ["general"]
        context_size = 4096
        use_as_default = False
        
        # Modelos menores têm preferência como padrão
        if "phi-3-mini" in model_name.lower():
            capabilities = ["fast", "general", "creative"]
            context_size = 4096
            use_as_default = True  # Melhor para uso geral
        elif "phi-2" in model_name.lower():
            capabilities = ["fast", "general"]
            context_size = 2048
            use_as_default = True  # Alternativa para default
        elif "qwen1.5-1.8b" in model_name.lower():
            capabilities = ["fast", "general", "chat"]
            context_size = 2048
            use_as_default = True  # Alternativa para default
        elif "falcon-7b" in model_name.lower():
            capabilities = ["creative", "general", "precise"]
            context_size = 4096
        elif "opt-125m" in model_name.lower():
            capabilities = ["fast", "general"]
            context_size = 2048
            
        # Adiciona à configuração
        model_configs.append({
            "name": model_name.lower(),
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
        
        # Define modelo padrão se aplicável
        if use_as_default and not default_model:
            default_model = model_name.lower()
    
    # Se nenhum modelo foi definido como padrão, use o primeiro
    if not default_model and model_configs:
        default_model = model_configs[0]["name"]
    
    # Verifica se encontrou algum modelo
    if not model_configs:
        print("Nenhum modelo Transformers encontrado.")
        return None
    
    # Cria a configuração completa
    config = {
        "models": model_configs,
        "default_model": default_model
    }
    
    # Salva o arquivo
    config_path = "modelos_existentes.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Configuração gerada: {config_path}")
    print(f"Modelos configurados: {', '.join(m['name'] for m in model_configs)}")
    print(f"Modelo padrão: {default_model}")
    print("\nPara usar esta configuração:")
    print(f"python main.py --config {config_path} --prompt \"Sua pergunta aqui\"")
    
    return config_path

if __name__ == "__main__":
    create_transformers_config()