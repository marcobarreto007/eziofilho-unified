#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OptionsExpert - Especialista em Análise de Opções
------------------------------------------------
Analisa mercados de opções, estratégias de derivativos e superfícies de volatilidade.

Autor: EzioFilho LLMGraph Team
Data: Maio/2025
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Adicionar diretório pai ao path para importações relativas
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Importar módulos do projeto
from core.phi2_experts.phi2_base_expert import Phi2Expert

class OptionsExpert(Phi2Expert):
    """
    Especialista em análise de opções baseado em Phi-2.
    
    Capacidades:
    - Análise de superfícies de volatilidade
    - Estratégias de trading com opções
    - Avaliação de preços de opções
    - Análise de skew e estrutura a termo
    - Cálculo e interpretação de gregas
    - Operações de estrutura e spreads
    """
    
    def __init__(self, **kwargs):
        """
        Inicializa o especialista de opções
        
        Args:
            **kwargs: Parâmetros adicionais para a classe base
        """
        # Definir mensagem de sistema especializada
        system_message = """
        Você é um especialista em análise de opções com ampla experiência em mercados financeiros.
        Sua tarefa é analisar mercados de opções, desenvolver estratégias de derivativos, 
        avaliar superfícies de volatilidade, calcular e interpretar gregas, analisar skew e estrutura a termo,
        e criar estratégias estruturadas com opções.
        
        Regras:
        1. Sempre retorne uma análise estruturada com campos claros
        2. Explique detalhadamente os aspectos teóricos e práticos relevantes
        3. Considere o impacto das gregas e sensibilidades nas estratégias
        4. Analise as implicações de volatilidade implícita para as expectativas do mercado
        5. Seja específico sobre as estruturas de opções e suas características
        6. Forneça recomendações contextualizadas com base em objetivos de risco/retorno
        """
        
        super().__init__(
            expert_type="options_expert",
            domain="quant",
            specialization="options_analysis",
            system_message=system_message,
            **kwargs
        )
        
        # Configurações específicas para análise de opções
        self.option_strategies = [
            "Call Covered", "Put Protected", "Straddle", "Strangle",
            "Bull Spread", "Bear Spread", "Butterfly", "Iron Condor",
            "Calendar Spread", "Diagonal Spread", "Collar", "Risk Reversal",
            "Ratio Spread", "Box Spread", "Jade Lizard", "Iron Fly"
        ]
        
        self.greeks = [
            "Delta", "Gamma", "Theta", "Vega", "Rho", "Charm", 
            "Color", "Speed", "Vanna", "Volga", "Vomma", "DvegaDtime"
        ]
        
        self.volatility_structures = [
            "Flat", "Smiling", "Skewed", "Steep", "Inverted", "Forward",
            "Backward", "Kinked", "Twisted", "Convex", "Concave"
        ]
    
    def analyze(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analisa mercados de opções e fornece insights
        
        Args:
            input_data: Descrição textual ou dicionário com dados de opções
            
        Returns:
            Resultado da análise de opções
        """
        start_time = time.time()
        
        # Extrair dados se input for dicionário
        if isinstance(input_data, dict):
            if "options_data" in input_data:
                data_type = "structured"
                description = self._format_structured_data(input_data)
                underlying = input_data.get("underlying", "desconhecido")
                expiration = input_data.get("expiration", "desconhecido")
            else:
                data_type = "text"
                description = input_data.get("description", "")
                underlying = input_data.get("underlying", "desconhecido")
                expiration = input_data.get("expiration", "desconhecido")
        else:
            # Input é texto descritivo
            data_type = "text"
            description = input_data
            underlying = "desconhecido"
            expiration = "desconhecido"
        
        # Verificar se temos dados suficientes
        if not description or len(description.strip()) < 20:
            return {
                "error": "Dados insuficientes para análise de opções",
                "market_sentiment": "indefinido",
                "confidence": 0,
                "processing_time": time.time() - start_time
            }
        
        # Preparar prompt para o modelo
        prompt = f"""
        Realize uma análise detalhada do mercado de opções com base nos seguintes dados:
        
        ATIVO SUBJACENTE: {underlying}
        VENCIMENTO: {expiration}
        
        DADOS: {description}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "market_sentiment": {{
                "directional_bias": "viés direcional do mercado",
                "volatility_expectation": "expectativa de volatilidade",
                "term_structure": "estrutura a termo da volatilidade"
            }},
            "volatility_surface": {{
                "skew": "análise do skew de volatilidade",
                "term_structure": "análise da estrutura a termo",
                "notable_strikes": [strikes notáveis com distorções],
                "surface_shape": "forma da superfície de volatilidade"
            }},
            "options_pricing": {{
                "value_areas": [áreas de valor identificadas],
                "overpriced_options": [opções potencialmente sobrevalorizadas],
                "underpriced_options": [opções potencialmente subvalorizadas]
            }},
            "greek_analysis": {{
                "key_exposures": [principais exposições de gregas no mercado],
                "gamma_profile": "perfil de gamma por strike",
                "theta_characteristics": "características de theta",
                "vega_distribution": "distribuição de vega"
            }},
            "trading_opportunities": [
                {{
                    "strategy": "estratégia recomendada",
                    "strikes": "strikes envolvidos",
                    "rationale": "justificativa",
                    "risk_reward": "perfil de risco/recompensa",
                    "key_risks": [principais riscos]
                }}
            ],
            "catalysts": [catalisadores potenciais para o mercado de opções],
            "confidence": (porcentagem de 0 a 100)
        }}
        """
        
        # Gerar resposta
        try:
            response = self.generate_response(prompt)
            
            # Extrair JSON da resposta
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                
                # Adicionar metadados
                result["underlying"] = underlying
                result["expiration"] = expiration
                result["data_type"] = data_type
                result["processing_time"] = time.time() - start_time
                
                return result
            else:
                # Falha ao extrair JSON, tentar parsear manualmente
                return self._parse_non_json_response(response, underlying, expiration, start_time)
                
        except Exception as e:
            return {
                "error": f"Erro na análise de opções: {str(e)}",
                "market_sentiment": "indefinido",
                "confidence": 0,
                "underlying": underlying,
                "expiration": expiration,
                "processing_time": time.time() - start_time
            }
    
    def _format_structured_data(self, data: Dict[str, Any]) -> str:
        """
        Formata dados estruturados em descrição textual para o modelo
        
        Args:
            data: Dicionário com dados estruturados
            
        Returns:
            Descrição textual formatada
        """
        options_data = data.get("options_data", {})
        
        # Formatar chain de opções
        options_chain = options_data.get("chain", {})
        chain_str = "CHAIN DE OPÇÕES:\n"
        
        # Formatar calls
        calls = options_chain.get("calls", [])
        chain_str += "CALLS:\n"
        for call in calls:
            chain_str += f"- Strike: {call.get('strike')}, Prêmio: {call.get('premium')}, IV: {call.get('iv')}, "
            chain_str += f"Delta: {call.get('delta')}, Volume: {call.get('volume')}, OI: {call.get('open_interest')}\n"
        
        # Formatar puts
        puts = options_chain.get("puts", [])
        chain_str += "PUTS:\n"
        for put in puts:
            chain_str += f"- Strike: {put.get('strike')}, Prêmio: {put.get('premium')}, IV: {put.get('iv')}, "
            chain_str += f"Delta: {put.get('delta')}, Volume: {put.get('volume')}, OI: {put.get('open_interest')}\n"
        
        # Formatar superfície de volatilidade
        vol_surface = options_data.get("volatility_surface", {})
        vol_str = "SUPERFÍCIE DE VOLATILIDADE:\n"
        for key, value in vol_surface.items():
            vol_str += f"- {key}: {value}\n"
        
        # Formatar dados de mercado
        market_data = options_data.get("market_data", {})
        market_str = "DADOS DE MERCADO:\n"
        for key, value in market_data.items():
            market_str += f"- {key}: {value}\n"
        
        # Formatar conjunto
        formatted_data = f"""
        {data.get('description', 'Sem descrição disponível')}
        
        {chain_str}
        
        {vol_str}
        
        {market_str}
        
        INFORMAÇÕES ADICIONAIS:
        {data.get('additional_info', '')}
        """
        
        return formatted_data
    
    def _parse_non_json_response(self, response: str, underlying: str, expiration: str, start_time: float) -> Dict[str, Any]:
        """
        Tenta extrair informações relevantes de uma resposta não-JSON
        
        Args:
            response: Texto da resposta
            underlying: Ativo subjacente
            expiration: Data de vencimento
            start_time: Hora de início do processamento
            
        Returns:
            Dicionário com informações extraídas
        """
        result = {
            "underlying": underlying,
            "expiration": expiration,
            "processing_time": time.time() - start_time,
            "format_error": "Resposta não veio no formato JSON esperado"
        }
        
        # Tentar extrair sentimento de mercado
        lower_resp = response.lower()
        
        # Extrair viés direcional
        if "alta" in lower_resp or "bullish" in lower_resp:
            directional_bias = "alta"
        elif "baixa" in lower_resp or "bearish" in lower_resp:
            directional_bias = "baixa"
        elif "neutro" in lower_resp or "neutral" in lower_resp:
            directional_bias = "neutro"
        else:
            directional_bias = "não identificado"
        
        # Extrair expectativa de volatilidade
        if "alta volatilidade" in lower_resp or "volatilidade alta" in lower_resp:
            vol_expectation = "alta"
        elif "baixa volatilidade" in lower_resp or "volatilidade baixa" in lower_resp:
            vol_expectation = "baixa"
        elif "volatilidade estável" in lower_resp or "estável" in lower_resp:
            vol_expectation = "estável"
        else:
            vol_expectation = "não identificada"
        
        # Estruturar sentimento
        result["market_sentiment"] = {
            "directional_bias": directional_bias,
            "volatility_expectation": vol_expectation
        }
        
        # Extrair estratégias mencionadas
        mentioned_strategies = []
        for strategy in self.option_strategies:
            if strategy.lower() in lower_resp:
                mentioned_strategies.append(strategy)
        
        if mentioned_strategies:
            result["trading_opportunities"] = [{"strategy": strategy} for strategy in mentioned_strategies]
        else:
            result["trading_opportunities"] = [{"strategy": "não identificada"}]
        
        # Extrair confiança
        confidence = 50  # valor padrão
        result["confidence"] = confidence
        
        # Texto original
        result["original_response"] = response
        
        return result
    
    def analyze_volatility_surface(self, surface_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa superfície de volatilidade em detalhes
        
        Args:
            surface_data: Dados da superfície de volatilidade
            
        Returns:
            Análise detalhada da superfície
        """
        description = f"""
        Analise a seguinte superfície de volatilidade:
        
        DADOS DA SUPERFÍCIE:
        """
        
        for key, value in surface_data.items():
            if isinstance(value, dict):
                description += f"- {key}:\n"
                for sub_key, sub_value in value.items():
                    description += f"  * {sub_key}: {sub_value}\n"
            else:
                description += f"- {key}: {value}\n"
        
        prompt = f"""
        {description}
        
        Forneça sua análise da superfície de volatilidade no seguinte formato JSON:
        {{
            "surface_shape": "forma geral da superfície",
            "skew_analysis": {{
                "pattern": "padrão do skew",
                "steepness": "inclinação do skew",
                "market_implications": "implicações para o mercado"
            }},
            "term_structure": {{
                "shape": "forma da estrutura a termo",
                "key_tenors": [tenores chave com características notáveis],
                "market_implications": "implicações para o mercado"
            }},
            "volatility_regimes": [regimes de volatilidade identificados],
            "volatility_arbitrage": [oportunidades de arbitragem identificadas],
            "catalysts": [catalisadores que poderiam impactar a superfície],
            "tactical_recommendations": [recomendações táticas baseadas na superfície]
        }}
        """
        
        try:
            response = self.generate_response(prompt)
            
            # Processar resposta JSON
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                return {"error": "Não foi possível extrair JSON da resposta"}
        except Exception as e:
            return {"error": f"Erro na análise da superfície de volatilidade: {str(e)}"}
    
    def design_options_strategy(self, market_view: Dict[str, Any], risk_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Desenvolve uma estratégia de opções baseada em visão de mercado e parâmetros de risco
        
        Args:
            market_view: Visão de mercado do cliente
            risk_parameters: Parâmetros de risco e objetivos
            
        Returns:
            Estratégia de opções recomendada
        """
        # Formatar visão de mercado
        view_str = "VISÃO DE MERCADO:\n"
        for key, value in market_view.items():
            if isinstance(value, dict):
                view_str += f"- {key}:\n"
                for sub_key, sub_value in value.items():
                    view_str += f"  * {sub_key}: {sub_value}\n"
            else:
                view_str += f"- {key}: {value}\n"
        
        # Formatar parâmetros de risco
        risk_str = "PARÂMETROS DE RISCO:\n"
        for key, value in risk_parameters.items():
            if isinstance(value, dict):
                risk_str += f"- {key}:\n"
                for sub_key, sub_value in value.items():
                    risk_str += f"  * {sub_key}: {sub_value}\n"
            else:
                risk_str += f"- {key}: {value}\n"
        
        prompt = f"""
        Projete uma estratégia de opções com base nas seguintes informações:
        
        {view_str}
        
        {risk_str}
        
        Forneça sua recomendação de estratégia no seguinte formato JSON:
        {{
            "strategy_overview": {{
                "name": "nome da estratégia",
                "category": "categoria da estratégia",
                "alignment": "alinhamento com a visão de mercado"
            }},
            "construction": {{
                "legs": [
                    {{
                        "type": "tipo da opção (call/put)",
                        "action": "comprar/vender",
                        "strike_selection": "critério de seleção do strike",
                        "expiration": "critério de seleção do vencimento",
                        "quantity": "quantidade relativa"
                    }}
                ],
                "ratios": "relações entre as pernas (se aplicável)"
            }},
            "risk_profile": {{
                "max_loss": "perda máxima",
                "max_gain": "ganho máximo",
                "breakeven_points": [pontos de equilíbrio],
                "key_greeks": {{
                    "net_delta": "delta líquido",
                    "net_gamma": "gamma líquido",
                    "net_theta": "theta líquido",
                    "net_vega": "vega líquido"
                }}
            }},
            "scenario_analysis": [análise de cenários],
            "trade_management": {{
                "entry_criteria": "critérios de entrada",
                "exit_criteria": "critérios de saída",
                "adjustment_triggers": "gatilhos para ajustes",
                "hedging_considerations": "considerações de hedge"
            }},
            "key_risks": [principais riscos da estratégia],
            "alternative_strategies": [estratégias alternativas a considerar]
        }}
        """
        
        try:
            response = self.generate_response(prompt)
            
            # Processar resposta JSON
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                return {"error": "Não foi possível extrair JSON da resposta"}
        except Exception as e:
            return {"error": f"Erro no desenvolvimento da estratégia de opções: {str(e)}"}
    
    def calculate_greeks(self, option_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula e interpreta as gregas para uma opção ou estratégia
        
        Args:
            option_data: Dados da opção ou estratégia
            
        Returns:
            Análise de gregas e sensibilidades
        """
        description = f"""
        Calcule e interprete as gregas para a seguinte opção ou estratégia:
        
        DADOS:
        """
        
        for key, value in option_data.items():
            if isinstance(value, dict):
                description += f"- {key}:\n"
                for sub_key, sub_value in value.items():
                    description += f"  * {sub_key}: {sub_value}\n"
            else:
                description += f"- {key}: {value}\n"
        
        prompt = f"""
        {description}
        
        Forneça sua análise de gregas no seguinte formato JSON:
        {{
            "primary_greeks": {{
                "delta": "valor e interpretação",
                "gamma": "valor e interpretação",
                "theta": "valor e interpretação",
                "vega": "valor e interpretação",
                "rho": "valor e interpretação"
            }},
            "secondary_greeks": {{
                "vanna": "valor e interpretação",
                "volga": "valor e interpretação",
                "charm": "valor e interpretação",
                "color": "valor e interpretação"
            }},
            "risk_exposures": [principais exposições a risco],
            "greek_dynamics": {{
                "time_effect": "como as gregas mudam com o tempo",
                "underlying_moves": "como as gregas mudam com movimentos do ativo",
                "volatility_changes": "como as gregas mudam com mudanças na volatilidade"
            }},
            "hedging_implications": [implicações para estratégias de hedge],
            "key_sensitivities": [sensibilidades críticas a monitorar]
        }}
        """
        
        try:
            response = self.generate_response(prompt)
            
            # Processar resposta JSON
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                return {"error": "Não foi possível extrair JSON da resposta"}
        except Exception as e:
            return {"error": f"Erro no cálculo e interpretação de gregas: {str(e)}"}
