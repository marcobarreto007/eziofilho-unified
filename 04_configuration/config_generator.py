"""
Model Config Generator - Utilitário para criar configuração de modelos locais
"""

import os
import sys
import json
import argparse
from pathlib import Path

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Gerador de configuração para modelos locais")
    parser.add_argument("--phi2", help="Caminho completo para o modelo Phi-2 GGUF")
    parser.add_argument("--mistral", help="Caminho completo para o modelo Mistral GGUF")
    parser.add_argument("--tinyllama", help="Caminho completo para o modelo TinyLlama GGUF")
    parser.add_argument("--other", nargs=2, action='append', 
                      help="Outros modelos no formato: --other nome caminho")
    parser.add_argument("--output", "-o", default="model_config.json",
                      help="Caminho para o arquivo de configuração de saída")
    
    args = parser.parse_args()
    
    # Verifica se pelo menos um modelo foi especificado
    if not (args.phi2 or args.mistral or args.tinyllama or args.other):
        print("Erro: Especifique pelo menos um modelo.")
        print("Exemplo: python config_generator.py --phi2 \"C:\\caminho\\para\\phi-2.gguf\"")
        print("         python config_generator.py --mistral \"C:\\caminho\\para\\mistral.gguf\"")
        print("         python config_generator.py --other modelo1 \"C:\\caminho\\para\\modelo1.gguf\"")
        return 1
    
    # Prepara a configuração
    models = []
    default_model = None
    
    # Adiciona Phi-2 se especificado
    if args.phi2:
        path = Path(args.phi2)
        if not path.exists():
            print(f"Aviso: Arquivo não encontrado: {path}")
        else:
            models.append({
                "name": "phi2",
                "path": str(path),
                "model_type": "gguf",
                "capabilities": ["fast", "general"],
                "min_prompt_tokens": 0,
                "max_prompt_tokens": 2048
            })
            default_model = "phi2"
    
    # Adiciona Mistral se especificado
    if args.mistral:
        path = Path(args.mistral)
        if not path.exists():
            print(f"Aviso: Arquivo não encontrado: {path}")
        else:
            models.append({
                "name": "mistral",
                "path": str(path),
                "model_type": "gguf",
                "capabilities": ["precise", "general", "creative"],
                "min_prompt_tokens": 500,
                "max_prompt_tokens": 4096
            })
            default_model = default_model or "mistral"
    
    # Adiciona TinyLlama se especificado
    if args.tinyllama:
        path = Path(args.tinyllama)
        if not path.exists():
            print(f"Aviso: Arquivo não encontrado: {path}")
        else:
            models.append({
                "name": "tinyllama",
                "path": str(path),
                "model_type": "gguf",
                "capabilities": ["fast", "general"],
                "min_prompt_tokens": 0,
                "max_prompt_tokens": 1024
            })
            default_model = default_model or "tinyllama"
    
    # Adiciona outros modelos especificados
    if args.other:
        for name, path_str in args.other:
            path = Path(path_str)
            if not path.exists():
                print(f"Aviso: Arquivo não encontrado: {path}")
            else:
                models.append({
                    "name": name,
                    "path": str(path),
                    "model_type": "gguf",
                    "capabilities": ["general"],
                    "min_prompt_tokens": 0,
                    "max_prompt_tokens": 4096
                })
                default_model = default_model or name
    
    # Verifica se pelo menos um modelo válido foi encontrado
    if not models:
        print("Erro: Nenhum arquivo de modelo válido encontrado.")
        return 1
    
    # Cria o arquivo de configuração
    config = {
        "models": models,
        "default_model": default_model
    }
    
    # Escreve o arquivo
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Configuração gerada com sucesso: {output_path}")
    print(f"Modelos configurados: {', '.join(m['name'] for m in models)}")
    print(f"Modelo padrão: {default_model}")
    print(f"\nPara usar esta configuração:")
    print(f"python main.py --config {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())