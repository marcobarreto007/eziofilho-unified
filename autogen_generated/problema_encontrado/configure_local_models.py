"""
Config Generator - Configuração específica para modelos existentes
"""

import os
import sys
import json
from pathlib import Path

def main():
    """Função principal"""
    # Diretório base dos modelos
    models_base = Path("modelos_hf")
    
    # Verifique se o diretório existe
    if not models_base.exists():
        print(f"Erro: Diretório {models_base} não encontrado!")
        print(f"Este script deve ser executado do diretório pai de 'modelos_hf'")
        return 1
    
    # Procure por modelos GGUF (melhor suporte com llama-cpp-python)
    gguf_folders = [
        "TheBloke--Mistral-7B-Instruct-v0.1-GGUF",
        "TheBloke--Llama-2-7B-Chat-GGUF",
        "TheBloke--CodeLlama-7B-Instruct-GGUF",
        "microsoft-phi-2"  # Asumindo que este diretório contém uma versão GGUF de Phi-2
    ]
    
    # Configuração de modelos
    models = []
    default_model = None
    
    # Verifique cada pasta de modelo
    for folder_name in gguf_folders:
        folder_path = models_base / folder_name
        
        if not folder_path.exists():
            print(f"Aviso: Pasta não encontrada: {folder_path}")
            continue
        
        # Procure por arquivos .gguf na pasta
        gguf_files = list(folder_path.glob("*.gguf"))
        
        if not gguf_files:
            # Se não encontrar arquivos .gguf, tente encontrar outros formatos compatíveis
            bin_files = list(folder_path.glob("*.bin"))
            safetensors_files = list(folder_path.glob("*.safetensors"))
            
            if bin_files or safetensors_files:
                print(f"Aviso: Encontrados arquivos de modelo mas não no formato GGUF em {folder_path}")
                print(f"       Estes modelos podem funcionar com o wrapper Transformers")
            else:
                print(f"Aviso: Nenhum arquivo de modelo encontrado em {folder_path}")
            
            continue
        
        # Escolha o arquivo GGUF com melhor quantização
        # Prefira Q4_K_M por ser um bom equilíbrio entre tamanho e qualidade
        best_file = None
        for file in gguf_files:
            if "Q4_K_M" in file.name:
                best_file = file
                break
        
        # Se não encontrar Q4_K_M, use o primeiro arquivo GGUF
        if not best_file and gguf_files:
            best_file = gguf_files[0]
        
        if best_file:
            # Determine o nome e as capacidades com base no nome da pasta
            if "Mistral" in folder_name:
                model_name = "mistral"
                capabilities = ["precise", "general", "creative"]
                context_size = 8192
            elif "Llama-2" in folder_name and "Chat" in folder_name:
                model_name = "llama2"
                capabilities = ["general", "creative", "chat"]
                context_size = 4096
            elif "CodeLlama" in folder_name:
                model_name = "codellama"
                capabilities = ["code", "precise", "technical"]
                context_size = 4096
            elif "phi-2" in folder_name:
                model_name = "phi2"
                capabilities = ["fast", "general"]
                context_size = 2048
            else:
                # Nome genérico baseado no arquivo
                model_name = best_file.stem.lower().replace("-", "_")
                capabilities = ["general"]
                context_size = 4096
            
            # Adiciona o modelo à configuração
            models.append({
                "name": model_name,
                "path": str(best_file),
                "model_type": "gguf",
                "capabilities": capabilities,
                "min_prompt_tokens": 0,
                "max_prompt_tokens": context_size,
                "params": {
                    "n_threads": 4,  # Ajuste conforme necessário
                    "n_ctx": context_size
                }
            })
            
            # Define o modelo padrão (prefira modelos menores e mais rápidos)
            if not default_model or model_name == "phi2":
                default_model = model_name
    
    # Verifique se encontramos algum modelo
    if not models:
        print("Erro: Nenhum modelo GGUF encontrado nas pastas especificadas.")
        print("Dica: Você pode adicionar manualmente modelos usando o script config_generator.py")
        return 1
    
    # Crie o arquivo de configuração
    config = {
        "models": models,
        "default_model": default_model or models[0]["name"]
    }
    
    # Escreva o arquivo
    output_path = Path("ezio_models.json")
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Configuração gerada com sucesso: {output_path}")
    print(f"Modelos configurados: {', '.join(m['name'] for m in models)}")
    print(f"Modelo padrão: {config['default_model']}")
    print(f"\nPara usar esta configuração:")
    print(f"python main.py --config {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())