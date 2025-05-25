"""
Pacote de especialistas unificados para o sistema EzioFilho
Fornece uma implementação consolidada de classes de especialistas
"""

__version__ = '3.0.0'
__author__ = 'EzioFilho Team'

# Importar para disponibilizar diretamente no pacote
from core.unified_base_expert import EzioBaseExpert
from core.unified_sentiment_expert import SentimentExpert

# Registrar classes disponíveis
__all__ = [
    'EzioBaseExpert',
    'SentimentExpert'
]

# Configuração dos especialistas disponíveis
AVAILABLE_EXPERTS = {
    "sentiment": {
        "class": "SentimentExpert",
        "description": "Especialista unificado em análise de sentimento financeiro",
        "module": "core.unified_sentiment_expert"
    }
    # Outros especialistas serão adicionados conforme implementados
}

def get_expert(expert_type, **kwargs):
    """
    Factory function para criar instância do especialista solicitado
    
    Args:
        expert_type: Tipo de especialista (sentiment, technical, etc.)
        **kwargs: Argumentos adicionais para o construtor do especialista
            - config_path: Caminho para arquivo de configuração
            - gpu_id: ID específico da GPU (opcional)
            - gpu_ids: Lista de IDs das GPUs disponíveis (opcional)
            - gpu_monitor: Instância do GPUMonitor (opcional)
        
    Returns:
        Instância do especialista solicitado
    
    Raises:
        ValueError: Se tipo de especialista não é suportado
    """
    if expert_type not in AVAILABLE_EXPERTS:
        available = ", ".join(AVAILABLE_EXPERTS.keys())
        raise ValueError(f"Tipo de especialista '{expert_type}' não disponível. Disponíveis: {available}")
    
    expert_info = AVAILABLE_EXPERTS[expert_type]
    
    # Importar dinamicamente
    import importlib
    module = importlib.import_module(expert_info["module"])
    
    # Obter e instanciar a classe
    expert_class = getattr(module, expert_info["class"])
    
    # Adicionar expert_type aos kwargs
    kwargs["expert_type"] = expert_type
    
    # Instanciar o especialista com todos os parâmetros
    expert_instance = expert_class(**kwargs)
    
    return expert_instance

def get_available_experts():
    """
    Retorna informações sobre especialistas disponíveis
    
    Returns:
        Dicionário com informações de especialistas disponíveis
    """
    return AVAILABLE_EXPERTS
