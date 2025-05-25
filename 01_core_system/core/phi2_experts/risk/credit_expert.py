#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CreditExpert - Especialista em Análise de Crédito
-------------------------------------------------
Analisa risco de crédito, solvência, e qualidade de dívida para ativos e emissores.

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

class CreditExpert(Phi2Expert):
    """
    Especialista em análise de crédito baseado em Phi-2.
    
    Capacidades:
    - Avaliação de risco de crédito corporativo e soberano
    - Análise de spreads de crédito e rendimentos
    - Previsão de mudanças de rating
    - Identificação de riscos de default
    - Análise de estrutura de dívida e covenants
    - Estratégias de investimento baseadas em crédito
    """
    
    def __init__(self, **kwargs):
        """
        Inicializa o especialista de crédito
        
        Args:
            **kwargs: Parâmetros adicionais para a classe base
        """
        # Definir mensagem de sistema especializada
        system_message = """
        Você é um especialista em análise de crédito com ampla experiência em mercados financeiros.
        Sua tarefa é avaliar riscos de crédito, analisar spreads e ratings, prever mudanças na qualidade de crédito,
        identificar riscos de default, analisar estruturas de dívida, e recomendar estratégias baseadas em crédito.
        
        Regras:
        1. Sempre retorne uma análise estruturada com campos claros
        2. Avalie detalhadamente os indicadores de qualidade de crédito (ratings, spreads, CDS)
        3. Analise métricas fundamentais relacionadas a crédito (índices de cobertura, alavancagem)
        4. Identifique tendências e riscos na estrutura de dívida
        5. Considere o ambiente macroeconômico e seu impacto no crédito
        6. Forneça recomendações baseadas no perfil de risco do emissor
        """
        
        super().__init__(
            expert_type="credit_expert",
            domain="risk",
            specialization="credit_analysis",
            system_message=system_message,
            **kwargs
        )
        
        # Configurações específicas para análise de crédito
        self.credit_metrics = [
            "Rating de Crédito", "Spread de Crédito", "CDS", 
            "Yield to Maturity", "Índice de Cobertura de Juros", 
            "Dívida/EBITDA", "Grau de Alavancagem", "Probabilidade de Default",
            "Recovery Rate", "Custo da Dívida", "Duração"
        ]
        
        self.credit_quality_levels = [
            "AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D",
            "Investment Grade", "High Yield", "Distressed"
        ]
        
        self.credit_strategies = [
            "Barbell", "Ladder", "Bullet", "Spread Trade", 
            "Rising Stars", "Fallen Angels", "Crossover", "Carry Trade"
        ]
    
    def analyze(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analisa risco de crédito e fornece insights
        
        Args:
            input_data: Descrição textual ou dicionário com dados de crédito
            
        Returns:
            Resultado da análise de crédito
        """
        start_time = time.time()
        
        # Extrair dados se input for dicionário
        if isinstance(input_data, dict):
            if "credit_data" in input_data:
                data_type = "structured"
                description = self._format_structured_data(input_data)
                issuer = input_data.get("issuer", "desconhecido")
                asset_type = input_data.get("asset_type", "desconhecido")
            else:
                data_type = "text"
                description = input_data.get("description", "")
                issuer = input_data.get("issuer", "desconhecido")
                asset_type = input_data.get("asset_type", "desconhecido")
        else:
            # Input é texto descritivo
            data_type = "text"
            description = input_data
            issuer = "desconhecido"
            asset_type = "desconhecido"
        
        # Verificar se temos dados suficientes
        if not description or len(description.strip()) < 20:
            return {
                "error": "Dados insuficientes para análise de crédito",
                "credit_quality": "indefinido",
                "confidence": 0,
                "processing_time": time.time() - start_time
            }
        
        # Preparar prompt para o modelo
        prompt = f"""
        Realize uma análise detalhada de crédito com base nos seguintes dados:
        
        EMISSOR: {issuer}
        TIPO DE ATIVO: {asset_type}
        
        DADOS: {description}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "credit_quality": (string: rating de crédito ou qualidade de crédito),
            "credit_trend": (string: "melhorando", "estável", "deteriorando"),
            "current_metrics": {{
                "credit_spread": "spread atual vs. benchmark",
                "yield": "rendimento atual",
                "coverage_ratio": "índice de cobertura de juros",
                "leverage_ratio": "índice de alavancagem"
            }},
            "risk_assessment": {{
                "default_probability": "probabilidade de default",
                "recovery_expectation": "expectativa de recuperação em caso de default",
                "covenant_analysis": "análise de covenants e proteções"
            }},
            "fundamental_drivers": [principais fatores impactando o crédito],
            "catalysts": [catalisadores potenciais para mudanças de rating/spread],
            "market_implications": {{
                "relative_value": "avaliação de valor relativo",
                "liquidity_assessment": "avaliação de liquidez",
                "comparable_credits": [emissores/títulos comparáveis]
            }},
            "forecasts": {{
                "spread_direction": "direção esperada para spreads",
                "rating_outlook": "perspectiva para ratings",
                "key_risks": [principais riscos no horizonte]
            }},
            "investment_recommendations": [recomendações de investimento baseadas em crédito],
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
                result["issuer"] = issuer
                result["asset_type"] = asset_type
                result["data_type"] = data_type
                result["processing_time"] = time.time() - start_time
                
                return result
            else:
                # Falha ao extrair JSON, tentar parsear manualmente
                return self._parse_non_json_response(response, issuer, asset_type, start_time)
                
        except Exception as e:
            return {
                "error": f"Erro na análise de crédito: {str(e)}",
                "credit_quality": "indefinido",
                "confidence": 0,
                "issuer": issuer,
                "asset_type": asset_type,
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
        credit_data = data.get("credit_data", {})
        
        # Formatar dados financeiros
        financials = credit_data.get("financials", {})
        financial_str = "DADOS FINANCEIROS:\n"
        for key, value in financials.items():
            financial_str += f"- {key}: {value}\n"
        
        # Formatar dados de mercado
        market_data = credit_data.get("market_data", {})
        market_str = "DADOS DE MERCADO:\n"
        for key, value in market_data.items():
            market_str += f"- {key}: {value}\n"
        
        # Formatar ratings
        ratings = credit_data.get("ratings", {})
        ratings_str = "RATINGS:\n"
        for agency, rating in ratings.items():
            ratings_str += f"- {agency}: {rating}\n"
        
        # Formatar perfil de dívida
        debt_profile = credit_data.get("debt_profile", {})
        debt_str = "PERFIL DE DÍVIDA:\n"
        for key, value in debt_profile.items():
            debt_str += f"- {key}: {value}\n"
        
        # Formatar conjunto
        formatted_data = f"""
        {data.get('description', 'Sem descrição disponível')}
        
        {financial_str}
        
        {market_str}
        
        {ratings_str}
        
        {debt_str}
        
        INFORMAÇÕES ADICIONAIS:
        {data.get('additional_info', '')}
        """
        
        return formatted_data
    
    def _parse_non_json_response(self, response: str, issuer: str, asset_type: str, start_time: float) -> Dict[str, Any]:
        """
        Tenta extrair informações relevantes de uma resposta não-JSON
        
        Args:
            response: Texto da resposta
            issuer: Nome do emissor
            asset_type: Tipo de ativo
            start_time: Hora de início do processamento
            
        Returns:
            Dicionário com informações extraídas
        """
        result = {
            "issuer": issuer,
            "asset_type": asset_type,
            "processing_time": time.time() - start_time,
            "format_error": "Resposta não veio no formato JSON esperado"
        }
        
        # Tentar extrair qualidade de crédito
        lower_resp = response.lower()
        
        # Extrair rating/qualidade
        for quality in self.credit_quality_levels:
            if quality.lower() in lower_resp:
                result["credit_quality"] = quality
                break
        else:
            result["credit_quality"] = "não identificado"
        
        # Tentar extrair tendência
        if "deteriorando" in lower_resp or "piora" in lower_resp:
            result["credit_trend"] = "deteriorando"
        elif "melhorando" in lower_resp or "melhora" in lower_resp:
            result["credit_trend"] = "melhorando"
        else:
            result["credit_trend"] = "estável"
        
        # Extrair confiança
        confidence = 50  # valor padrão
        result["confidence"] = confidence
        
        # Texto original
        result["original_response"] = response
        
        return result
    
    def assess_credit_quality(self, financials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Avalia qualidade de crédito com base em dados financeiros
        
        Args:
            financials: Dados financeiros para análise
            
        Returns:
            Avaliação de qualidade de crédito
        """
        description = f"""
        Analise a qualidade de crédito com base nos seguintes dados financeiros:
        
        DADOS FINANCEIROS:
        """
        
        for key, value in financials.items():
            description += f"- {key}: {value}\n"
        
        prompt = f"""
        {description}
        
        Forneça sua avaliação de crédito no seguinte formato JSON:
        {{
            "implied_rating": "rating implícito com base nos dados financeiros",
            "key_strengths": [pontos fortes da situação financeira],
            "key_weaknesses": [pontos fracos ou riscos financeiros],
            "commentary": "breve comentário sobre a saúde financeira",
            "key_ratios": {{
                "ratio1": "valor e interpretação",
                "ratio2": "valor e interpretação"
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
            return {"error": f"Erro na avaliação de qualidade de crédito: {str(e)}"}
    
    def analyze_spread_dynamics(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa dinâmica de spreads de crédito
        
        Args:
            market_data: Dados de mercado para análise
            
        Returns:
            Análise de spreads de crédito
        """
        # Implementação semelhante à assess_credit_quality
        description = f"""
        Analise a dinâmica de spreads de crédito com base nos seguintes dados de mercado:
        
        DADOS DE MERCADO:
        """
        
        for key, value in market_data.items():
            description += f"- {key}: {value}\n"
        
        prompt = f"""
        {description}
        
        Forneça sua análise de spreads no seguinte formato JSON:
        {{
            "spread_assessment": "avaliação dos spreads atuais (caro/justo/barato)",
            "historical_comparison": "comparação com níveis históricos",
            "relative_value": "valor relativo comparado a pares",
            "spread_drivers": [fatores que estão influenciando os spreads],
            "forecast": "previsão para direção dos spreads",
            "trade_ideas": [idéias de negociação baseadas na análise de spreads]
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
            return {"error": f"Erro na análise de spreads: {str(e)}"}
