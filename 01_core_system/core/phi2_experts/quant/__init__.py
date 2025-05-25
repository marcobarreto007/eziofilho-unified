"""
Especialistas Quantitativos baseados em Phi-2
"""

from core.phi2_experts.quant.algorithmic_expert import AlgorithmicExpert
from core.phi2_experts.quant.options_expert import OptionsExpert
from core.phi2_experts.quant.fixed_income_expert import FixedIncomeExpert
from core.phi2_experts.quant.crypto_expert import CryptoExpert

__all__ = [
    "AlgorithmicExpert",
    "OptionsExpert",
    "FixedIncomeExpert", 
    "CryptoExpert"
]
