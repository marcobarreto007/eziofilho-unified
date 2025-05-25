"""Ferramenta de diagnóstico para o sistema AutoGen"""

import sys
import importlib.util
from pathlib import Path

# Configuração de cores para terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def check(condition, message, success_msg, fail_msg):
    """Verifica uma condição e exibe mensagem apropriada"""
    print(f"{Colors.GREEN if condition else Colors.RED}{'✔' if condition else '✘'}{Colors.RESET} {message}", end=" ")
    if condition:
        print(success_msg)
    else:
        print(fail_msg)
    return condition

def print_header(msg):
    """Exibe cabeçalho"""
    print(f"{Colors.BLUE}▶{Colors.RESET} {msg}")

def main():
    """Função principal"""
    # Verifica diretório do projeto
    project_root = Path(__file__).parent.absolute()
    print_header(f"Projeto em: {project_root}")
    
    # Verifica estrutura de diretórios
    core_dir = project_root / "core"
    if not core_dir.exists():
        print(f"{Colors.RED}✘{Colors.RESET} Diretório 'core/' não encontrado!")
        return 1
    
    # Lista arquivos em core/
    core_files = list(core_dir.glob("*.*"))
    print_header(f"Conteúdo de core/: {core_files}")
    
    # Verifica __init__.py
    init_file = core_dir / "__init__.py"
    check(init_file.exists(), 
          "core/__init__.py", 
          "encontrado", 
          "NÃO encontrado (necessário para importação como pacote)")
    
    # Verifica instalação do autogen
    try:
        import autogen
        version = getattr(autogen, "__version__", "desconhecida")
        check(True, f"autogen: {version}", "", "")
    except ImportError:
        check(False, "autogen", "", "NÃO instalado")
    
    # Tenta importar módulos do core
    sys.path.insert(0, str(project_root))
    
    # Tenta importar core.local_model_wrapper
    try:
        import core.local_model_wrapper
        check(True, "core.local_model_wrapper", "importado com sucesso", "")
    except ImportError as e:
        check(False, "core.local_model_wrapper", "", f"NÃO encontrável ({e})")
    
    # Tenta importar core.model_router
    try:
        import core.model_router
        check(True, "core.model_router", "importado com sucesso", "")
    except ImportError as e:
        check(False, "core.model_router", "", f"NÃO encontrável ({e})")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())