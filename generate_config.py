# Caminho do projeto: C:\Users\anapa\SuperIA\EzioFilhoUnified\generate_config.py
"""
REM Gera arquivo JSON com todos os modelos locais detectados na pasta 'modelos_hf'
REM Salva a configuração em 'modelos_existentes.json'
REM Compatível com ModelRouter e main.py
"""

import os
import json

REM_MODELS_PATH = "modelos_hf"  # Pasta padrão dos modelos
REM_OUTPUT_FILE = "modelos_existentes.json"

# Dicionário de mapeamento de nomes amigáveis para nomes internos
MODEL_PATTERNS = {
    "phi-3-mini-4k-instruct": "microsoft--phi-3-mini-4k-instruct",
    "phi-2": "microsoft--phi-2",
    "qwen1.5-1.8b-chat": "Qwen--Qwen1.5-1.8B-Chat",
    "falcon-7b-instruct": "tiiuae--falcon-7b-instruct",
    "opt-125m": "facebook--opt-125m",
    "llama-2-7b-chat": "TheBloke--Llama-2-7B-Chat-GGUF",
    "llama-2-13b-chat": "TheBloke--Llama-2-13B-chat-GGUF",
    "mistral-7b-instruct": "TheBloke--Mistral-7B-Instruct-v0.1-GGUF",
    # Adicione mais padrões conforme necessidade
}

def find_models(base_path):
    """
    REM Busca todos os modelos válidos no diretório de modelos
    """
    found = {}
    if not os.path.isdir(base_path):
        print(f"REM Caminho '{base_path}' não encontrado!")
        return found

    for name, folder in MODEL_PATTERNS.items():
        model_dir = os.path.join(base_path, folder)
        if os.path.isdir(model_dir):
            # Procura arquivos de modelo
            files = [f for f in os.listdir(model_dir) if f.endswith(".gguf") or f.endswith(".safetensors") or f.endswith(".bin")]
            if files:
                found[name] = os.path.join(model_dir, files[0])
    return found

def main():
    """
    REM Roda varredura, gera e salva o JSON de configuração
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))
    modelos_dir = os.path.join(project_dir, REM_MODELS_PATH)

    print(f"REM Buscando modelos em: {modelos_dir}")

    found_models = find_models(modelos_dir)

    if not found_models:
        print("REM Nenhum modelo encontrado. Verifique o diretório 'modelos_hf'.")
        return

    # Define modelo padrão como o mais moderno disponível
    default_model = "phi-3-mini-4k-instruct" if "phi-3-mini-4k-instruct" in found_models else list(found_models.keys())[0]

    config = {
        "models": found_models,
        "default": default_model
    }

    output_path = os.path.join(project_dir, REM_OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"REM Configuração gerada com sucesso: {output_path}")
    print(f"REM Modelos detectados: {list(found_models.keys())}")
    print(f"REM Modelo padrão: {default_model}")

if __name__ == "__main__":
    main()
