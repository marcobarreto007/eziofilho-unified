#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FixedIncomeExpert - Especialista em Análise de Renda Fixa
--------------------------------------------------------
Analisa mercados de renda fixa, taxas de juros e instrumentos de dívida.

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

class FixedIncomeExpert(Phi2Expert):
    """
    Especialista em análise de renda fixa baseado em Phi-2.
    
    Capacidades:
    - Análise de curvas de juros e spreads
    - Avaliação de títulos e instrumentos de dívida
    - Estratégias de renda fixa e derivativos de taxas
    - Análise macroeconômica relacionada a taxas de juros
    - Gestão de duration e convexidade
    - Estratégias de imunização e correspondência de fluxo de caixa
    """
    
    def __init__(self, **kwargs):
        """
        Inicializa o especialista de renda fixa
        
        Args:
            **kwargs: Parâmetros adicionais para a classe base
        """
        # Definir mensagem de sistema especializada
        system_message = """
        Você é um especialista em análise de renda fixa com ampla experiência em mercados financeiros.
        Sua tarefa é analisar mercados de renda fixa, curvas de juros, títulos e instrumentos de dívida,
        desenvolver estratégias de renda fixa, avaliar riscos de taxa de juros, analisar fatores macroeconômicos
        que afetam taxas de juros, e recomendar posicionamentos em renda fixa.
        
        Regras:
        1. Sempre retorne uma análise estruturada com campos claros
        2. Analise detalhadamente as curvas de juros e suas implicações
        3. Considere os fatores macroeconômicos que afetam taxas de juros
        4. Avalie medidas de risco como duration e convexidade
        5. Forneça recomendações específicas de posicionamento
        6. Explique o impacto de diferentes cenários nas estratégias
        """
        
        super().__init__(
            expert_type="fixed_income_expert",
            domain="quant",
            specialization="fixed_income_analysis",
            system_message=system_message,
            **kwargs
        )
        
        # Configurações específicas para análise de renda fixa
        self.curve_shapes = [
            "Normal", "Flat", "Inverted", "Steep", "Humped",
            "Positively Sloped", "Negatively Sloped", "Butterfly"
        ]
        
        self.fixed_income_strategies = [
            "Bullet", "Barbell", "Ladder", "Riding the Yield Curve",
            "Duration Targeting", "Cash Flow Matching", "Immunization",
            "Credit Spread", "Roll-Down", "Carry", "Relative Value"
        ]
        
        self.risk_metrics = [
            "Duration", "Modified Duration", "Effective Duration",
            "Convexity", "DV01", "Key Rate Duration", "Yield Volatility",
            "Option-Adjusted Spread", "Z-Spread", "Breakeven Spread"
        ]
    
    def analyze(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analisa mercados de renda fixa e fornece insights
        
        Args:
            input_data: Descrição textual ou dicionário com dados de renda fixa
            
        Returns:
            Resultado da análise de renda fixa
        """
        start_time = time.time()
        
        # Extrair dados se input for dicionário
        if isinstance(input_data, dict):
            if "fixed_income_data" in input_data:
                data_type = "structured"
                description = self._format_structured_data(input_data)
                market = input_data.get("market", "desconhecido")
                sector = input_data.get("sector", "desconhecido")
            else:
                data_type = "text"
                description = input_data.get("description", "")
                market = input_data.get("market", "desconhecido")
                sector = input_data.get("sector", "desconhecido")
        else:
            # Input é texto descritivo
            data_type = "text"
            description = input_data
            market = "desconhecido"
            sector = "desconhecido"
        
        # Verificar se temos dados suficientes
        if not description or len(description.strip()) < 20:
            return {
                "error": "Dados insuficientes para análise de renda fixa",
                "curve_assessment": "indefinido",
                "confidence": 0,
                "processing_time": time.time() - start_time
            }
        
        # Preparar prompt para o modelo
        prompt = f"""
        Realize uma análise detalhada do mercado de renda fixa com base nos seguintes dados:
        
        MERCADO: {market}
        SETOR: {sector}
        
        DADOS: {description}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "curve_assessment": {{
                "shape": "forma da curva de juros",
                "steepness": "inclinação da curva",
                "dynamics": "dinâmica recente da curva"
            }},
            "macro_environment": {{
                "rate_expectations": "expectativas para taxas de juros",
                "inflation_outlook": "perspectivas de inflação",
                "central_bank_policy": "análise da política do banco central",
                "growth_outlook": "perspectivas de crescimento econômico"
            }},
            "valuation": {{
                "absolute_value": "avaliação de valor absoluto",
                "relative_value": "avaliação de valor relativo",
                "rich_cheap_sectors": [setores ricos/baratos],
                "term_premium": "avaliação do prêmio a termo"
            }},
            "risk_assessment": {{
                "duration_risk": "avaliação do risco de duration",
                "convexity": "análise de convexidade",
                "spread_risk": "avaliação do risco de spread",
                "volatility_risk": "avaliação do risco de volatilidade"
            }},
            "market_technicals": {{
                "supply_demand": "dinâmica de oferta e demanda",
                "positioning": "posicionamento dos investidores",
                "flows": "fluxos recentes",
                "liquidity_conditions": "condições de liquidez"
            }},
            "strategy_recommendations": [
                {{
                    "strategy": "estratégia recomendada",
                    "rationale": "justificativa",
                    "implementation": "implementação",
                    "key_risks": [principais riscos]
                }}
            ],
            "specific_opportunities": [oportunidades específicas identificadas],
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
                result["market"] = market
                result["sector"] = sector
                result["data_type"] = data_type
                result["processing_time"] = time.time() - start_time
                
                return result
            else:
                # Falha ao extrair JSON, tentar parsear manualmente
                return self._parse_non_json_response(response, market, sector, start_time)
                
        except Exception as e:
            return {
                "error": f"Erro na análise de renda fixa: {str(e)}",
                "curve_assessment": "indefinido",
                "confidence": 0,
                "market": market,
                "sector": sector,
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
        fixed_income_data = data.get("fixed_income_data", {})
        
        # Formatar curva de juros
        yield_curve = fixed_income_data.get("yield_curve", {})
        curve_str = "CURVA DE JUROS:\n"
        for tenor, rate in yield_curve.items():
            curve_str += f"- {tenor}: {rate}\n"
        
        # Formatar spreads
        spreads = fixed_income_data.get("spreads", {})
        spreads_str = "SPREADS:\n"
        for spread_type, value in spreads.items():
            spreads_str += f"- {spread_type}: {value}\n"
        
        # Formatar dados macroeconômicos
        macro_data = fixed_income_data.get("macro_data", {})
        macro_str = "DADOS MACROECONÔMICOS:\n"
        for key, value in macro_data.items():
            macro_str += f"- {key}: {value}\n"
        
        # Formatar dados de mercado
        market_data = fixed_income_data.get("market_data", {})
        market_str = "DADOS DE MERCADO:\n"
        for key, value in market_data.items():
            market_str += f"- {key}: {value}\n"
        
        # Formatar conjunto
        formatted_data = f"""
        {data.get('description', 'Sem descrição disponível')}
        
        {curve_str}
        
        {spreads_str}
        
        {macro_str}
        
        {market_str}
        
        INFORMAÇÕES ADICIONAIS:
        {data.get('additional_info', '')}
        """
        
        return formatted_data
    
    def _parse_non_json_response(self, response: str, market: str, sector: str, start_time: float) -> Dict[str, Any]:
        """
        Tenta extrair informações relevantes de uma resposta não-JSON
        
        Args:
            response: Texto da resposta
            market: Nome do mercado
            sector: Setor de renda fixa
            start_time: Hora de início do processamento
            
        Returns:
            Dicionário com informações extraídas
        """
        result = {
            "market": market,
            "sector": sector,
            "processing_time": time.time() - start_time,
            "format_error": "Resposta não veio no formato JSON esperado"
        }
        
        # Tentar extrair forma da curva
        lower_resp = response.lower()
        
        # Extrair formato da curva
        for shape in self.curve_shapes:
            if shape.lower() in lower_resp:
                result["curve_assessment"] = {"shape": shape}
                break
        else:
            result["curve_assessment"] = {"shape": "não identificado"}
        
        # Extrair estratégias mencionadas
        mentioned_strategies = []
        for strategy in self.fixed_income_strategies:
            if strategy.lower() in lower_resp:
                mentioned_strategies.append(strategy)
        
        if mentioned_strategies:
            result["strategy_recommendations"] = [{"strategy": strategy} for strategy in mentioned_strategies]
        else:
            result["strategy_recommendations"] = [{"strategy": "não identificada"}]
        
        # Extrair confiança
        confidence = 50  # valor padrão
        result["confidence"] = confidence
        
        # Texto original
        result["original_response"] = response
        
        return result
    
    def analyze_yield_curve(self, curve_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa curva de juros em detalhes
        
        Args:
            curve_data: Dados da curva de juros
            
        Returns:
            Análise detalhada da curva
        """
        description = f"""
        Analise a seguinte curva de juros:
        
        DADOS DA CURVA:
        """
        
        for tenor, rate in curve_data.items():
            description += f"- {tenor}: {rate}\n"
        
        prompt = f"""
        {description}
        
        Forneça sua análise da curva de juros no seguinte formato JSON:
        {{
            "curve_shape": "forma geral da curva",
            "steepness": "inclinação da curva",
            "key_segments": [segmentos-chave da curva com características notáveis],
            "anomalies": [anomalias ou distorções identificadas],
            "implied_expectations": {{
                "short_term": "expectativas implícitas de curto prazo",
                "medium_term": "expectativas implícitas de médio prazo",
                "long_term": "expectativas implícitas de longo prazo"
            }},
            "historical_comparison": "comparação com padrões históricos",
            "trading_opportunities": [oportunidades identificadas na curva],
            "risk_considerations": [considerações de risco baseadas na forma da curva]
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
            return {"error": f"Erro na análise da curva de juros: {str(e)}"}
    
    def calculate_duration_risk(self, bond_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula e analisa risco de duration para um título ou portfólio
        
        Args:
            bond_data: Dados do título ou portfólio
            
        Returns:
            Análise de risco de duration
        """
        description = f"""
        Calcule e analise o risco de duration para o seguinte título ou portfólio:
        
        DADOS:
        """
        
        for key, value in bond_data.items():
            if isinstance(value, dict):
                description += f"- {key}:\n"
                for sub_key, sub_value in value.items():
                    description += f"  * {sub_key}: {sub_value}\n"
            else:
                description += f"- {key}: {value}\n"
        
        prompt = f"""
        {description}
        
        Forneça sua análise de risco de duration no seguinte formato JSON:
        {{
            "duration_metrics": {{
                "macaulay_duration": "duration de Macaulay",
                "modified_duration": "duration modificada",
                "effective_duration": "duration efetiva",
                "key_rate_durations": {{
                    "2y": "duration para taxa de 2 anos",
                    "5y": "duration para taxa de 5 anos",
                    "10y": "duration para taxa de 10 anos",
                    "30y": "duration para taxa de 30 anos"
                }}
            }},
            "convexity": "medida de convexidade",
            "interest_rate_sensitivity": {{
                "parallel_shift": "sensibilidade a deslocamentos paralelos",
                "steepening": "sensibilidade a aumento de inclinação",
                "flattening": "sensibilidade a diminuição de inclinação",
                "curvature": "sensibilidade a mudanças de curvatura"
            }},
            "scenario_analysis": [análise de cenários de taxas de juros],
            "hedging_recommendations": [recomendações para hedge de risco de taxa],
            "positioning_recommendations": [recomendações de posicionamento baseadas em duration]
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
            return {"error": f"Erro no cálculo e análise de risco de duration: {str(e)}"}
    
    def develop_fixed_income_strategy(self, market_view: Dict[str, Any], portfolio_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Desenvolve uma estratégia de renda fixa baseada em visão de mercado e restrições de portfólio
        
        Args:
            market_view: Visão de mercado
            portfolio_constraints: Restrições de portfólio
            
        Returns:
            Estratégia de renda fixa recomendada
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
        
        # Formatar restrições
        constraints_str = "RESTRIÇÕES DE PORTFÓLIO:\n"
        for key, value in portfolio_constraints.items():
            if isinstance(value, dict):
                constraints_str += f"- {key}:\n"
                for sub_key, sub_value in value.items():
                    constraints_str += f"  * {sub_key}: {sub_value}\n"
            else:
                constraints_str += f"- {key}: {value}\n"
        
        prompt = f"""
        Desenvolva uma estratégia de renda fixa com base nas seguintes informações:
        
        {view_str}
        
        {constraints_str}
        
        Forneça sua recomendação de estratégia no seguinte formato JSON:
        {{
            "strategy_overview": {{
                "name": "nome da estratégia",
                "approach": "abordagem geral",
                "alignment": "alinhamento com a visão de mercado"
            }},
            "duration_positioning": {{
                "target_duration": "duration alvo",
                "curve_positioning": "posicionamento na curva",
                "rationale": "justificativa"
            }},
            "sector_allocation": [
                {{
                    "sector": "setor",
                    "allocation": "alocação recomendada",
                    "rationale": "justificativa"
                }}
            ],
            "credit_strategy": {{
                "credit_quality": "qualidade de crédito alvo",
                "spread_duration": "duration de spread alvo",
                "focus_areas": [áreas de foco específicas]
            }},
            "implementation": {{
                "core_holdings": [títulos ou produtos principais],
                "tactical_trades": [trades táticos],
                "execution_approach": "abordagem de execução"
            }},
            "risk_management": {{
                "key_risks": [principais riscos],
                "hedging_approach": "abordagem de hedge",
                "stress_tests": [testes de estresse recomendados]
            }},
            "performance_expectations": {{
                "expected_return": "retorno esperado",
                "volatility": "volatilidade esperada",
                "key_success_factors": [fatores-chave para sucesso]
            }}
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
            return {"error": f"Erro no desenvolvimento da estratégia de renda fixa: {str(e)}"}
    
    def analyze_relative_value(self, instruments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analisa valor relativo entre instrumentos de renda fixa
        
        Args:
            instruments: Lista de instrumentos para comparação
            
        Returns:
            Análise de valor relativo
        """
        description = "Analise o valor relativo entre os seguintes instrumentos de renda fixa:\n\n"
        
        for i, instrument in enumerate(instruments, 1):
            description += f"INSTRUMENTO {i}:\n"
            for key, value in instrument.items():
                if isinstance(value, dict):
                    description += f"- {key}:\n"
                    for sub_key, sub_value in value.items():
                        description += f"  * {sub_key}: {sub_value}\n"
                else:
                    description += f"- {key}: {value}\n"
            description += "\n"
        
        prompt = f"""
        {description}
        
        Forneça sua análise de valor relativo no seguinte formato JSON:
        {{
            "ranking": [ranking dos instrumentos do mais atrativo ao menos atrativo],
            "value_assessment": [
                {{
                    "instrument": "nome do instrumento",
                    "valuation": "avaliação de valor (rich/cheap/fair)",
                    "key_metrics": [métricas-chave relevantes],
                    "rationale": "justificativa"
                }}
            ],
            "relative_value_trades": [
                {{
                    "trade_type": "tipo de trade",
                    "long": "instrumento para posição comprada",
                    "short": "instrumento para posição vendida",
                    "rationale": "justificativa",
                    "entry_levels": "níveis de entrada",
                    "exit_levels": "níveis de saída",
                    "risks": [principais riscos]
                }}
            ],
            "catalyst_assessment": [catalisadores potenciais para convergência de valor],
            "market_environment": "avaliação do ambiente de mercado para trades de valor relativo",
            "timing_considerations": "considerações de timing"
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
            return {"error": f"Erro na análise de valor relativo: {str(e)}"}
