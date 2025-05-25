"""
Model Finder - Busca recursiva por modelos GGUF e geração de configuração
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("model_finder")

def find_gguf_files(start_paths):
    """
    Procura recursivamente por arquivos .gguf
    
    Args:
        start_paths: Lista de caminhos onde iniciar a busca
        
    Returns:
        Lista de caminhos de arquivos .gguf encontrados
    """
    model_files = []
    
    for base_path in start_paths:
        base = Path(base_path).expanduser()
        if not base.exists():
            logger.warning(f"Caminho não encontrado: {base}")
            continue
            
        logger.info(f"Procurando em: {base}")
        
        # Contador para feedback visual
        count = 0
        spinner = "|/-\\"
        
        # Se é um arquivo e termina com .gguf, adiciona à lista
        if base.is_file() and base.name.lower().endswith('.gguf'):
            model_files.append(base)
            logger.info(f"Encontrado: {base}")
            continue
            
        # Senão, percorre o diretório recursivamente
        for root, dirs, files in os.walk(base):
            count += 1
            if count % 10 == 0:  # Atualiza o spinner a cada 10 diretórios
                sys.stdout.write(f"\rProcurando... {spinner[count % 4]} ({count} diretórios verificados)")
                sys.stdout.flush()
                
            for file in files:
                if file.lower().endswith('.gguf'):
                    full_path = Path(root) / file
                    model_files.append(full_path)
                    logger.info(f"\rEncontrado: {full_path}")
                    sys.stdout.write("\r" + " " * 50 + "\r")  # Limpa a linha do spinner
    
    sys.stdout.write("\r" + " " * 50 + "\r")  # Limpa a linha do spinner
    return model_files

def create_config_from_models(model_files):
    """
    Cria uma configuração a partir dos modelos encontrados
    
    Args:
        model_files: Lista de caminhos para arquivos de modelo
        
    Returns:
        Dicionário de configuração
    """
    models = []
    default_model = None
    
    for model_path in model_files:
        # Determina um nome razoável baseado no arquivo
        model_name = model_path.stem.lower().replace('-', '_')
        
        # Simplifica nomes muito longos
        if len(model_name) > 30:
            # Extrai partes-chave do nome
            parts = model_name.split('_')
            if len(parts) > 2:
                model_name = f"{parts[0]}_{parts[-1]}"
        
        # Identifica características com base no nome
        capabilities = ["general"]
        context_size = 4096
        
        if "phi" in model_name:
            capabilities = ["fast", "general"]
            context_size = 2048
            # Prioriza phi por ser pequeno e rápido
            default_model = model_name
        elif "mistral" in model_name:
            capabilities = ["precise", "general", "creative"]
            context_size = 8192
            if not default_model:
                default_model = model_name
        elif "llama" in model_name and "chat" in model_name:
            capabilities = ["general", "creative", "chat"]
            context_size = 4096
            if not default_model:
                default_model = model_name
        elif "codellama" in model_name or "code" in model_name:
            capabilities = ["code", "technical"]
            context_size = 4096
        elif "tiny" in model_name:
            capabilities = ["fast", "general"]
            context_size = 2048
            # Prioriza modelos pequenos
            if not default_model:
                default_model = model_name
        
        # Adiciona o modelo à configuração
        models.append({
            "name": model_name,
            "path": str(model_path),
            "model_type": "gguf",
            "capabilities": capabilities,
            "min_prompt_tokens": 0,
            "max_prompt_tokens": context_size,
            "params": {
                "n_threads": 4,  # Ajuste conforme necessário
                "n_ctx": context_size
            }
        })
    
    # Define um modelo padrão
    if not default_model and models:
        default_model = models[0]["name"]
    
    return {
        "models": models,
        "default_model": default_model
    }

def main():
    """Função principal"""
    # Locais prováveis para buscar (ajustar conforme necessário)
    search_paths = [
        Path.cwd(),  # Diretório atual
        Path.cwd().parent,  # Diretório pai
        Path.cwd() / "modelos_hf",  # Subdiretório modelos_hf
        Path.cwd().parent / "modelos_hf",  # modelos_hf no diretório pai
        "C:/Users/anapa/SuperIA/EzioFilhoUnified/modelos_hf",  # Caminho absoluto
        "C:/Users/anapa/SuperIA/modelos_hf",  # Outro caminho possível
        "D:/modelos_hf",  # Outra unidade
        "D:/models",  # Outra unidade
    ]
    
    # Instrução inicial
    logger.info("Buscando arquivos de modelo GGUF em diretórios comuns...")
    logger.info("Isso pode levar alguns minutos dependendo do tamanho dos diretórios.")
    
    # Busca modelos
    start_time = time.time()
    model_files = find_gguf_files(search_paths)
    search_time = time.time() - start_time
    
    # Resultados da busca
    if not model_files:
        logger.error("\nNenhum arquivo GGUF encontrado nos diretórios verificados.")
        logger.info("\nOpções:")
        logger.info("1. Especifique manualmente o caminho para os modelos:")
        logger.info("   python config_generator.py --phi2 \"CAMINHO_COMPLETO_PARA_MODELO.gguf\"")
        logger.info("2. Ou execute o sistema em modo de simulação (adicionando parâmetro):")
        logger.info("   python add_simulate_mode.py")
        logger.info("   python main.py --simulate --prompt \"Sua pergunta aqui\"")
        return 1
    
    # Gera configuração
    logger.info(f"\nEncontrados {len(model_files)} arquivos GGUF em {search_time:.1f} segundos.")
    config = create_config_from_models(model_files)
    
    # Salva o arquivo de configuração
    output_path = Path("models_found.json")
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\n✅ Configuração gerada: {output_path}")
    logger.info(f"Modelos configurados: {', '.join(m['name'] for m in config['models'])}")
    logger.info(f"Modelo padrão: {config['default_model']}")
    logger.info(f"\nPara usar esta configuração:")
    logger.info(f"python main.py --config {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())