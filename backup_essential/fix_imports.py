"""
Script de correção de importações para o projeto EzioFilhoUnified
"""

import os
import re
from pathlib import Path

def fix_imports(file_path, dry_run=False):
    """
    Corrige problemas de importação em um arquivo Python
    
    Args:
        file_path: Caminho para o arquivo a ser corrigido
        dry_run: Se True, apenas simula as mudanças sem alterar o arquivo
        
    Returns:
        (bool, str): (Sucesso, Mensagem)
    """
    path = Path(file_path)
    if not path.exists():
        return False, f"Arquivo não encontrado: {file_path}"
    
    # Lê o conteúdo do arquivo
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Padrões de substituição
    replacements = [
        # Importação normal
        (r'^import\s+local_model_wrapper', 'import core.local_model_wrapper'),
        # Importação from-import
        (r'^from\s+local_model_wrapper\s+import', 'from core.local_model_wrapper import'),
        # Referências no código
        (r'(?<!\.)local_model_wrapper\.', 'core.local_model_wrapper.'),
    ]
    
    # Conta substituições
    original = content
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Se nada mudou, retorna
    if content == original:
        return False, "Nenhuma mudança necessária"
    
    # Aplica as mudanças se não for dry run
    if not dry_run:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, f"Arquivo corrigido: {file_path}"
    else:
        return True, f"Mudanças identificadas (simulação): {file_path}"

def create_init_file(directory):
    """
    Cria ou atualiza arquivo __init__.py com código de suporte a importações
    
    Args:
        directory: Diretório onde criar o arquivo
    """
    init_path = Path(directory) / "__init__.py"
    
    # Conteúdo do arquivo __init__.py
    init_content = """# Inicialização do pacote core
# Este arquivo facilita importações relativas

# Torna os módulos disponíveis diretamente
from . import local_model_wrapper
from . import model_router

# Exporta funções e classes principais para facilitar o uso
try:
    from .local_model_wrapper import create_model_wrapper
    from .model_router import create_model_router, ModelRouter
except ImportError as e:
    print(f"Aviso: Erro ao importar módulos no __init__.py: {e}")
"""
    
    # Escreve o arquivo
    with open(init_path, 'w', encoding='utf-8') as f:
        f.write(init_content)
    
    return init_path

def main():
    """Função principal"""
    # Encontra o diretório do projeto
    project_root = Path(__file__).parent.absolute()
    core_dir = project_root / "core"
    
    print(f"Projeto: {project_root}")
    print(f"Diretório core: {core_dir}")
    
    if not core_dir.exists():
        print(f"Erro: Diretório core/ não encontrado em {project_root}")
        return 1
    
    # Cria/__init__.py aprimorado
    init_path = create_init_file(core_dir)
    print(f"✅ Criado/atualizado: {init_path}")
    
    # Corrige model_router.py
    model_router_path = core_dir / "model_router.py"
    if model_router_path.exists():
        success, msg = fix_imports(model_router_path)
        if success:
            print(f"✅ {msg}")
        else:
            print(f"ℹ️ {msg}")
    else:
        print(f"❌ Arquivo não encontrado: {model_router_path}")
    
    # Verifica importações
    print("\nVerificando importações...")
    try:
        import sys
        sys.path.insert(0, str(project_root))
        
        # Tenta importar como pacote
        from core import local_model_wrapper, model_router
        print("✅ Importações funcionando corretamente!")
        
        # Verifica funções específicas
        from core import create_model_wrapper, create_model_router, ModelRouter
        print("✅ Funções exportadas corretamente!")
        
        return 0
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("\nSugestão: Execute este script novamente após corrigir os erros.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())