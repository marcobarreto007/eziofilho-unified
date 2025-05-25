#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MacroAnalystExpert - Especialista em Análise Macroeconômica
----------------------------------------------------------
Analisa tendências macroeconômicas, política monetária e impactos nos mercados.

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

class MacroAnalystExpert(Phi2Expert):
    """
    Especialista em análise macroeconômica baseado em Phi-2.
    
    Capacidades:
    - Análise de indicadores macroeconômicos (PIB, inflação, desemprego, etc.)
    - Avaliação de políticas monetárias e fiscais
    - Análise de tendências econômicas globais e regionais
    - Identificação de implicações para diferentes classes de ativos
    - Avaliação de riscos geopolíticos e seus impactos econômicos
    - Previsão de cenários econômicos e impactos nos mercados
    """
    
    def __init__(self, **kwargs):
        """
        Inicializa o especialista de análise macroeconômica
        
        Args:
            **kwargs: Parâmetros adicionais para a classe base
        """
        # Definir mensagem de sistema especializada
        system_message = """
        Você é um economista sênior especializado em análise macroeconômica global.
        Sua tarefa é analisar indicadores econômicos, políticas monetárias e fiscais,
        tendências globais e seus impactos nos mercados financeiros. Você deve fornecer
        insights claros sobre perspectivas econômicas e suas implicações para investimentos.
        
        Regras:
        1. Sempre retorne uma análise estruturada com campos claros
        2. Baseie-se em dados macroeconômicos e eventos relevantes
        3. Avalie impactos de políticas monetárias e fiscais
        4. Considere fatores globais e regionais
        5. Identifique riscos e oportunidades para diferentes classes de ativos
        6. Forneça perspectivas sobre indicadores-chave (crescimento, inflação, juros)
        """
        
        super().__init__(expert_type="macro_economist", domain="market", specialization="macroeconomic_analysis", **kwargs)
        
        # Configurações específicas para análise macroeconômica
        self.key_indicators = [
            "PIB", "Inflação", "Taxa de Juros", "Desemprego", "Balança Comercial",
            "Dívida Pública", "Déficit Fiscal", "Conta Corrente", "Câmbio"
        ]
        
        self.asset_classes = [
            "Ações", "Renda Fixa", "Moedas", "Commodities", "Imóveis", 
            "Criptomoedas", "Private Equity"
        ]
        
        self.regions = [
            "América do Norte", "América Latina", "Europa", "Ásia-Pacífico",
            "China", "Japão", "Mercados Emergentes", "África", "Oriente Médio"
        ]
    
    def analyze(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analisa dados macroeconômicos e fornece insights para os mercados
        
        Args:
            input_data: Descrição textual ou dicionário com dados macroeconômicos
            
        Returns:
            Resultado da análise macroeconômica
        """
        start_time = time.time()
        
        # Extrair dados se input for dicionário
        if isinstance(input_data, dict):
            if "economic_data" in input_data:
                data_type = "structured"
                description = self._format_structured_data(input_data)
                region = input_data.get("region", "global")
                period = input_data.get("period", "atual")
            else:
                data_type = "text"
                description = input_data.get("description", "")
                region = input_data.get("region", "global")
                period = input_data.get("period", "atual")
        else:
            # Input é texto descritivo
            data_type = "text"
            description = input_data
            region = "global"
            period = "atual"
        
        # Verificar se temos dados suficientes
        if not description or len(description.strip()) < 20:
            return {
                "error": "Dados insuficientes para análise macroeconômica",
                "outlook": "incerto",
                "confidence": 0,
                "processing_time": time.time() - start_time
            }
        
        # Preparar prompt para o modelo
        prompt = f"""
        Realize uma análise macroeconômica completa com base nos seguintes dados:
        
        REGIÃO: {region}
        PERÍODO: {period}
        
        DADOS: {description}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "economic_outlook": (string: "expansão", "estabilidade", "desaceleração", "recessão"),
            "key_indicators": {{
                "growth": {{
                    "trend": "tendência do crescimento",
                    "risks": "riscos para o crescimento",
                    "forecast": "previsão numérica se disponível"
                }},
                "inflation": {{
                    "trend": "tendência da inflação",
                    "drivers": "fatores que impulsionam a inflação",
                    "forecast": "previsão numérica se disponível"
                }},
                "interest_rates": {{
                    "trend": "tendência das taxas de juros",
                    "central_bank_outlook": "perspectiva de política monetária",
                    "forecast": "previsão numérica se disponível"
                }}
            }},
            "asset_implications": {{
                "equities": "implicações para ações",
                "fixed_income": "implicações para renda fixa",
                "currencies": "implicações para moedas",
                "commodities": "implicações para commodities",
                "alternatives": "implicações para investimentos alternativos"
            }},
            "key_risks": [principais riscos macroeconômicos],
            "opportunities": [principais oportunidades],
            "policy_outlook": {{
                "monetary": "perspectiva de política monetária",
                "fiscal": "perspectiva de política fiscal"
            }},
            "confidence": (porcentagem de 0 a 100),
            "summary": "resumo da análise macroeconômica"
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
                result["region"] = region
                result["period"] = period
                result["data_type"] = data_type
                result["processing_time"] = time.time() - start_time
                
                return result
            else:
                # Falha ao extrair JSON, tentar parsear manualmente
                return self._parse_non_json_response(response, region, period, start_time)
                
        except Exception as e:
            return {
                "error": f"Erro na análise macroeconômica: {str(e)}",
                "economic_outlook": "incerto",
                "confidence": 0,
                "region": region,
                "period": period,
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
        description = []
        
        # Adicionar dados econômicos se disponíveis
        if "economic_data" in data:
            economic_data = data["economic_data"]
            
            # Indicadores econômicos básicos
            if "indicators" in economic_data:
                indicators = economic_data["indicators"]
                description.append("INDICADORES ECONÔMICOS:")
                
                # Se tivermos múltiplos períodos
                if isinstance(indicators, dict) and all(isinstance(v, dict) for v in indicators.values()):
                    for period, values in indicators.items():
                        description.append(f"Período: {period}")
                        for key, value in values.items():
                            description.append(f"- {key}: {value}")
                        description.append("")
                else:
                    for key, value in indicators.items():
                        description.append(f"- {key}: {value}")
                    description.append("")
            
            # Dados de política monetária
            if "monetary_policy" in economic_data:
                monetary = economic_data["monetary_policy"]
                description.append("POLÍTICA MONETÁRIA:")
                for key, value in monetary.items():
                    description.append(f"- {key}: {value}")
                description.append("")
            
            # Dados de política fiscal
            if "fiscal_policy" in economic_data:
                fiscal = economic_data["fiscal_policy"]
                description.append("POLÍTICA FISCAL:")
                for key, value in fiscal.items():
                    description.append(f"- {key}: {value}")
                description.append("")
            
            # Tendências econômicas
            if "trends" in economic_data:
                trends = economic_data["trends"]
                description.append("TENDÊNCIAS ECONÔMICAS:")
                if isinstance(trends, list):
                    for trend in trends:
                        description.append(f"- {trend}")
                else:
                    for key, value in trends.items():
                        description.append(f"- {key}: {value}")
                description.append("")
        
        # Adicionar dados de mercado se disponíveis
        if "market_data" in data:
            market_data = data["market_data"]
            description.append("DADOS DE MERCADO:")
            
            for asset_class, values in market_data.items():
                description.append(f"{asset_class.upper()}:")
                if isinstance(values, dict):
                    for key, value in values.items():
                        description.append(f"- {key}: {value}")
                else:
                    description.append(f"- {values}")
                description.append("")
        
        # Adicionar eventos relevantes se disponíveis
        if "key_events" in data:
            events = data["key_events"]
            description.append("EVENTOS RELEVANTES:")
            
            if isinstance(events, list):
                for event in events:
                    if isinstance(event, dict):
                        date = event.get("date", "")
                        desc = event.get("description", "")
                        impact = event.get("impact", "")
                        description.append(f"- {date} - {desc} (Impacto: {impact})")
                    else:
                        description.append(f"- {event}")
            else:
                for key, value in events.items():
                    description.append(f"- {key}: {value}")
                    
            description.append("")
            
        # Adicionar riscos geopolíticos se disponíveis
        if "geopolitical_risks" in data:
            risks = data["geopolitical_risks"]
            description.append("RISCOS GEOPOLÍTICOS:")
            
            if isinstance(risks, list):
                for risk in risks:
                    if isinstance(risk, dict):
                        desc = risk.get("description", "")
                        impact = risk.get("impact", "")
                        probability = risk.get("probability", "")
                        description.append(f"- {desc} (Impacto: {impact}, Probabilidade: {probability})")
                    else:
                        description.append(f"- {risk}")
            else:
                for key, value in risks.items():
                    description.append(f"- {key}: {value}")
        
        return "\n".join(description)
    
    def _parse_non_json_response(self, response: str, region: str, period: str, start_time: float) -> Dict[str, Any]:
        """
        Tenta extrair informações de uma resposta que não está em formato JSON
        
        Args:
            response: Texto da resposta
            region: Região analisada
            period: Período analisado
            start_time: Tempo de início do processamento
            
        Returns:
            Dicionário com resultados da análise
        """
        result = {
            "economic_outlook": "incerto",
            "key_indicators": {
                "growth": {
                    "trend": "Indefinido",
                    "risks": "Indefinido",
                    "forecast": "N/A"
                },
                "inflation": {
                    "trend": "Indefinido",
                    "drivers": "Indefinido",
                    "forecast": "N/A"
                },
                "interest_rates": {
                    "trend": "Indefinido",
                    "central_bank_outlook": "Indefinido",
                    "forecast": "N/A"
                }
            },
            "asset_implications": {
                "equities": "Indefinido",
                "fixed_income": "Indefinido",
                "currencies": "Indefinido",
                "commodities": "Indefinido",
                "alternatives": "Indefinido"
            },
            "key_risks": [],
            "opportunities": [],
            "policy_outlook": {
                "monetary": "Indefinido",
                "fiscal": "Indefinido"
            },
            "confidence": 30,
            "summary": "Análise incompleta",
            "region": region,
            "period": period,
            "processing_time": time.time() - start_time
        }
        
        # Tentar extrair outlook econômico
        outlook_terms = {
            "expansão": ["expansão", "crescimento", "recuperação", "expansion", "growth", "recovery"],
            "estabilidade": ["estabilidade", "estável", "stability", "stable", "neutral"],
            "desaceleração": ["desaceleração", "slowdown", "arrefecimento", "moderação"],
            "recessão": ["recessão", "recession", "contração", "crise", "downturn"]
        }
        
        for outlook, keywords in outlook_terms.items():
            if any(kw in response.lower() for kw in keywords):
                result["economic_outlook"] = outlook
                break
        
        # Extrair informações sobre PIB/Crescimento
        growth_section = False
        growth_info = []
        
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            
            # Verificar se estamos na seção de crescimento
            if any(term in line.lower() for term in ["crescimento", "pib", "growth", "gdp"]):
                growth_section = True
                continue
            
            # Capturar linhas não vazias enquanto estivermos na seção de crescimento
            if growth_section and line:
                if line.startswith("#") or line.startswith("=") or any(term in line.lower() for term in ["inflação", "juros", "inflation", "interest"]):
                    growth_section = False  # Nova seção começou
                else:
                    growth_info.append(line)
        
        # Processar informações de crescimento coletadas
        if growth_info:
            # Detectar tendência
            trend_keywords = {
                "positiva": ["aumento", "crescimento", "expansion", "positiv", "alta"],
                "negativa": ["queda", "contração", "redução", "declínio", "slowdown", "negativ"],
                "estável": ["estável", "estabilidade", "manutenção", "stable", "unchanged"]
            }
            
            trend = "Indefinido"
            for direction, keywords in trend_keywords.items():
                if any(any(kw in line.lower() for kw in keywords) for line in growth_info):
                    trend = direction
                    break
                    
            result["key_indicators"]["growth"]["trend"] = trend
            
            # Detectar riscos
            risks = [line for line in growth_info if any(term in line.lower() for term in ["risco", "risk", "ameaça", "preocupação", "concern"])]
            if risks:
                result["key_indicators"]["growth"]["risks"] = risks[0]
                
            # Detectar previsão
            forecasts = []
            for line in growth_info:
                import re
                numbers = re.findall(r'-?\d+\.?\d*%', line)
                if numbers:
                    forecasts.extend(numbers)
                    
            if forecasts:
                result["key_indicators"]["growth"]["forecast"] = forecasts[0]
        
        # Extrair riscos
        risks_section = False
        for line in lines:
            if any(term in line.lower() for term in ["riscos", "risks", "ameaças", "threats"]):
                risks_section = True
                continue
            
            if risks_section and line.strip() and (line.startswith("-") or line.startswith("*") or line.startswith("•")):
                risk = line[1:].strip()
                if risk:
                    result["key_risks"].append(risk)
                    # Limitar a alguns riscos principais
                    if len(result["key_risks"]) >= 5:
                        break
        
        # Extrair oportunidades
        opportunities_section = False
        for line in lines:
            if any(term in line.lower() for term in ["oportunidades", "opportunities", "potential"]):
                opportunities_section = True
                continue
            
            if opportunities_section and line.strip() and (line.startswith("-") or line.startswith("*") or line.startswith("•")):
                opportunity = line[1:].strip()
                if opportunity:
                    result["opportunities"].append(opportunity)
                    # Limitar a algumas oportunidades principais
                    if len(result["opportunities"]) >= 5:
                        break
        
        # Extrair resumo/conclusão
        summary_section = False
        summary_lines = []
        
        for line in lines:
            if any(term in line.lower() for term in ["conclusão", "conclusion", "resumo", "summary"]):
                summary_section = True
                continue
            
            if summary_section and line.strip():
                summary_lines.append(line.strip())
                # Limitar a algumas linhas
                if len(summary_lines) >= 3:
                    break
        
        if summary_lines:
            result["summary"] = " ".join(summary_lines)
        
        return result
    
    def analyze_scenario(self, base_data: Dict[str, Any], scenario_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza análise de cenário macroeconômico com diferentes premissas
        
        Args:
            base_data: Dados econômicos base
            scenario_parameters: Parâmetros alternativos para o cenário
            
        Returns:
            Análise do cenário alternativo
        """
        start_time = time.time()
        
        # Extrair informações básicas
        region = base_data.get("region", "global")
        period = base_data.get("period", "projeção")
        scenario_name = scenario_parameters.get("name", "alternativo")
        
        # Criar descrição do cenário
        scenario_description = f"CENÁRIO: {scenario_name}\n\n"
        
        # Adicionar parâmetros do cenário
        if "parameters" in scenario_parameters:
            scenario_description += "PARÂMETROS DO CENÁRIO:\n"
            for param, value in scenario_parameters["parameters"].items():
                scenario_description += f"- {param}: {value}\n"
            scenario_description += "\n"
        
        # Adicionar descrição do cenário se fornecida
        if "description" in scenario_parameters:
            scenario_description += f"DESCRIÇÃO DO CENÁRIO:\n{scenario_parameters['description']}\n\n"
            
        # Adicionar os dados base
        base_description = self._format_structured_data(base_data)
        
        # Combinar tudo
        full_description = f"{scenario_description}\nDADOS BASE:\n{base_description}"
        
        # Preparar prompt para o modelo
        prompt = f"""
        Realize uma análise do seguinte cenário macroeconômico:
        
        REGIÃO: {region}
        PERÍODO: {period}
        
        {full_description}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "scenario_name": "{scenario_name}",
            "economic_outlook": (string: "expansão", "estabilidade", "desaceleração", "recessão"),
            "scenario_probability": (porcentagem de 0 a 100),
            "key_differences": [principais diferenças em relação ao cenário base],
            "key_indicators": {{
                "growth": {{
                    "forecast": "previsão de crescimento no cenário",
                    "vs_base": "comparação com cenário base"
                }},
                "inflation": {{
                    "forecast": "previsão de inflação no cenário",
                    "vs_base": "comparação com cenário base"
                }},
                "interest_rates": {{
                    "forecast": "previsão de juros no cenário",
                    "vs_base": "comparação com cenário base"
                }}
            }},
            "asset_implications": {{
                "equities": "implicações para ações neste cenário",
                "fixed_income": "implicações para renda fixa neste cenário",
                "currencies": "implicações para moedas neste cenário",
                "commodities": "implicações para commodities neste cenário"
            }},
            "recommended_positioning": [recomendações de posicionamento neste cenário],
            "confidence": (porcentagem de 0 a 100),
            "summary": "resumo do cenário e suas implicações"
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
                result["region"] = region
                result["period"] = period
                result["processing_time"] = time.time() - start_time
                
                return result
            else:
                # Falha ao extrair JSON
                return {
                    "error": "Não foi possível processar a análise de cenário",
                    "scenario_name": scenario_name,
                    "region": region,
                    "period": period,
                    "processing_time": time.time() - start_time
                }
                
        except Exception as e:
            return {
                "error": f"Erro na análise de cenário: {str(e)}",
                "scenario_name": scenario_name,
                "region": region,
                "period": period,
                "processing_time": time.time() - start_time
            }
