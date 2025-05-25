# Inicialização do pacote core
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
