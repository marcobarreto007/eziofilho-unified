#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EzioFilho_LLMGraph - Especialistas Phi-2
----------------------------------------
Fornece especialistas financeiros baseados em Phi-2, otimizados para
análises em domínios específicos de mercado, risco e quantitativos.

Autor: EzioFilho LLMGraph Team
Data: Maio/2025
"""

from .phi2_base_expert import Phi2Expert

# Especialistas de Mercado
from .market.sentiment_expert import SentimentAnalystExpert as SentimentExpert
from .market.technical_expert import TechnicalAnalystExpert as TechnicalExpert
from .market.fundamental_expert import FundamentalAnalystExpert as FundamentalExpert
from .market.macro_expert import MacroAnalystExpert as MacroExpert

# Especialistas de Risco
from .risk.risk_manager_expert import RiskManagerExpert
from .risk.volatility_expert import VolatilityExpert
from .risk.credit_expert import CreditExpert
from .risk.liquidity_expert import LiquidityExpert

# Especialistas Quantitativos
from .quant.algorithmic_expert import AlgorithmicExpert
from .quant.options_expert import OptionsExpert
from .quant.fixed_income_expert import FixedIncomeExpert
from .quant.crypto_expert import CryptoExpert

# Dicionário de especialistas disponíveis
AVAILABLE_PHI2_EXPERTS = {
    # Mercado
    "sentiment_analyst": SentimentExpert,
    "technical_analyst": TechnicalExpert,
    "fundamental_analyst": FundamentalExpert,
    "macro_economist": MacroExpert,
    
    # Risco
    "risk_manager": RiskManagerExpert,
    "volatility_expert": VolatilityExpert,
    "credit_analyst": CreditExpert,
    "liquidity_specialist": LiquidityExpert,
    
    # Quantitativo
    "algorithmic_trader": AlgorithmicExpert,
    "options_specialist": OptionsExpert,
    "fixed_income": FixedIncomeExpert,
    "crypto_analyst": CryptoExpert
}

def get_phi2_expert(expert_type: str, **kwargs):
    """
    Obtém uma instância de especialista Phi-2
    
    Args:
        expert_type: Tipo de especialista (ex: "sentiment_analyst", "risk_manager")
        **kwargs: Parâmetros adicionais para inicialização do especialista
    
    Returns:
        Instância do especialista
    
    Raises:
        ValueError: Se o tipo de especialista for inválido
    """
    if expert_type not in AVAILABLE_PHI2_EXPERTS:
        valid_types = list(AVAILABLE_PHI2_EXPERTS.keys())
        raise ValueError(f"Tipo de especialista inválido: {expert_type}. Tipos válidos: {valid_types}")
    
    return AVAILABLE_PHI2_EXPERTS[expert_type](**kwargs)

def get_available_phi2_experts():
    """
    Retorna lista de tipos de especialistas Phi-2 disponíveis
    
    Returns:
        Lista de tipos de especialistas
    """
    return list(AVAILABLE_PHI2_EXPERTS.keys())

__all__ = [
    # Classe base
    "Phi2Expert",
    
    # Especialistas de Mercado
    "SentimentExpert",
    "TechnicalExpert",
    "FundamentalExpert",
    "MacroExpert",
    
    # Especialistas de Risco
    "RiskManagerExpert",
    "VolatilityExpert",
    "CreditExpert",
    "LiquidityExpert",
    
    # Especialistas Quantitativos
    "AlgorithmicExpert",
    "OptionsExpert",
    "FixedIncomeExpert",
    "CryptoExpert",
    
    # Funções auxiliares
    "get_phi2_expert",
    "get_available_phi2_experts"
]
