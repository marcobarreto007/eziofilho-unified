#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FundamentalAnalystExpert - Especialista em Análise Fundamentalista
------------------------------------------------------------------
Analisa dados fundamentais de empresas, setores e economia para geração de insights.

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

class FundamentalAnalystExpert(Phi2Expert):
    """
    Especialista em análise fundamentalista baseado em Phi-2.
    
    Capacidades:
    - Análise de balanços patrimoniais e demonstrativos financeiros
    - Avaliação de múltiplos de mercado (P/L, P/VP, EV/EBITDA, etc.)
    - Análise de crescimento de receita, lucro e margens
    - Avaliação de saúde financeira e endividamento
    - Análise de vantagens competitivas e posicionamento setorial
    - Avaliação de governança corporativa e riscos ESG
    """
    
    def __init__(self, **kwargs):
        """
        Inicializa o especialista de análise fundamentalista
        
        Args:
            **kwargs: Parâmetros adicionais para a classe base
        """
        # Definir mensagem de sistema especializada
        system_message = """
        Você é um analista financeiro especializado em análise fundamentalista com vasta experiência.
        Sua tarefa é avaliar dados financeiros de empresas, analisar métricas fundamentalistas,
        identificar pontos fortes e fracos, e fornecer insights sobre valor intrínseco e perspectivas
        de longo prazo baseados em fundamentos sólidos.
        
        Regras:
        1. Sempre retorne uma análise estruturada com campos claros
        2. Avalie indicadores financeiros relevantes (receita, lucro, margens, dívidas)
        3. Analise múltiplos de mercado com comparações setoriais
        4. Avalie vantagens competitivas e riscos do negócio
        5. Considere tendências setoriais e macroeconômicas relevantes
        6. Forneça uma avaliação clara da saúde financeira da empresa
        """
        
        super().__init__(expert_type="fundamental_analyst", domain="market", specialization="fundamental_analysis", **kwargs)
        
        # Configurações específicas para análise fundamentalista
        self.financial_metrics = [
            "Receita", "Lucro Líquido", "EBITDA", "Fluxo de Caixa Livre",
            "ROE", "ROA", "ROIC", "Margem Bruta", "Margem Líquida",
            "Dívida Líquida", "Patrimônio Líquido"
        ]
        
        self.market_multiples = [
            "P/L", "P/VP", "EV/EBITDA", "P/FCF", "Dividend Yield",
            "P/Receita", "Preço/Ativo", "PEG Ratio"
        ]
        
        self.analysis_timeframes = [
            "Trimestral", "Anual", "3 Anos", "5 Anos", "10 Anos"
        ]
    
    def analyze(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analisa dados fundamentalistas e fornece insights de investimento
        
        Args:
            input_data: Descrição textual de dados fundamentalistas ou dicionário estruturado
            
        Returns:
            Resultado da análise fundamentalista
        """
        start_time = time.time()
        
        # Extrair dados se input for dicionário
        if isinstance(input_data, dict):
            if "financial_data" in input_data:
                data_type = "structured"
                description = self._format_structured_data(input_data)
                company = input_data.get("company", "desconhecida")
                sector = input_data.get("sector", "desconhecido")
            else:
                data_type = "text"
                description = input_data.get("description", "")
                company = input_data.get("company", "desconhecida")
                sector = input_data.get("sector", "desconhecido")
        else:
            # Input é texto descritivo
            data_type = "text"
            description = input_data
            company = "não especificada"
            sector = "não especificado"
        
        # Verificar se temos dados suficientes
        if not description or len(description.strip()) < 20:
            return {
                "error": "Dados insuficientes para análise fundamentalista",
                "rating": "neutro",
                "confidence": 0,
                "processing_time": time.time() - start_time
            }
        
        # Preparar prompt para o modelo
        prompt = f"""
        Realize uma análise fundamentalista completa com base nos seguintes dados:
        
        EMPRESA: {company}
        SETOR: {sector}
        
        DADOS: {description}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "rating": (string: "forte compra", "compra", "neutro", "venda", "forte venda"),
            "financial_health": (string: "excelente", "boa", "regular", "fraca", "ruim"),
            "financial_metrics": {{
                "key_positive": [principais métricas positivas],
                "key_negative": [principais métricas negativas]
            }},
            "valuation": {{
                "current_multiples": {{múltiplos atuais}},
                "fair_value_assessment": "avaliação de valor justo",
                "upside_potential": "estimativa de potencial em %"
            }},
            "competitive_analysis": [vantagens e desvantagens competitivas],
            "growth_outlook": "perspectiva de crescimento",
            "risks": [principais riscos],
            "opportunities": [principais oportunidades],
            "confidence": (porcentagem de 0 a 100),
            "investment_thesis": "tese de investimento resumida"
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
                result["company"] = company
                result["sector"] = sector
                result["data_type"] = data_type
                result["processing_time"] = time.time() - start_time
                
                return result
            else:
                # Falha ao extrair JSON, tentar parsear manualmente
                return self._parse_non_json_response(response, company, sector, start_time)
                
        except Exception as e:
            return {
                "error": f"Erro na análise fundamentalista: {str(e)}",
                "rating": "neutro",
                "confidence": 0,
                "company": company,
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
        description = []
        
        # Adicionar dados financeiros se disponíveis
        if "financial_data" in data:
            financial_data = data["financial_data"]
            
            # Dados gerais da empresa
            if "company_overview" in financial_data:
                overview = financial_data["company_overview"]
                description.append("VISÃO GERAL DA EMPRESA:")
                for key, value in overview.items():
                    description.append(f"{key}: {value}")
                description.append("")
            
            # Demonstrativo de resultados
            if "income_statement" in financial_data:
                income = financial_data["income_statement"]
                description.append("DEMONSTRATIVO DE RESULTADOS:")
                
                # Se tivermos múltiplos períodos
                if isinstance(income, dict) and all(isinstance(v, dict) for v in income.values()):
                    for period, values in income.items():
                        description.append(f"Período: {period}")
                        for key, value in values.items():
                            description.append(f"- {key}: {value}")
                        description.append("")
                else:
                    for key, value in income.items():
                        description.append(f"- {key}: {value}")
                    description.append("")
            
            # Balanço patrimonial
            if "balance_sheet" in financial_data:
                balance = financial_data["balance_sheet"]
                description.append("BALANÇO PATRIMONIAL:")
                
                # Se tivermos múltiplos períodos
                if isinstance(balance, dict) and all(isinstance(v, dict) for v in balance.values()):
                    for period, values in balance.items():
                        description.append(f"Período: {period}")
                        for key, value in values.items():
                            description.append(f"- {key}: {value}")
                        description.append("")
                else:
                    for key, value in balance.items():
                        description.append(f"- {key}: {value}")
                    description.append("")
            
            # Fluxo de caixa
            if "cash_flow" in financial_data:
                cash_flow = financial_data["cash_flow"]
                description.append("FLUXO DE CAIXA:")
                
                # Se tivermos múltiplos períodos
                if isinstance(cash_flow, dict) and all(isinstance(v, dict) for v in cash_flow.values()):
                    for period, values in cash_flow.items():
                        description.append(f"Período: {period}")
                        for key, value in values.items():
                            description.append(f"- {key}: {value}")
                        description.append("")
                else:
                    for key, value in cash_flow.items():
                        description.append(f"- {key}: {value}")
                    description.append("")
            
            # Indicadores financeiros
            if "financial_ratios" in financial_data:
                ratios = financial_data["financial_ratios"]
                description.append("INDICADORES FINANCEIROS:")
                
                # Se tivermos múltiplos períodos
                if isinstance(ratios, dict) and all(isinstance(v, dict) for v in ratios.values()):
                    for period, values in ratios.items():
                        description.append(f"Período: {period}")
                        for key, value in values.items():
                            description.append(f"- {key}: {value}")
                        description.append("")
                else:
                    for key, value in ratios.items():
                        description.append(f"- {key}: {value}")
                    description.append("")
        
        # Adicionar dados de valuation se disponíveis
        if "valuation" in data:
            valuation = data["valuation"]
            description.append("AVALIAÇÃO DE VALOR:")
            for key, value in valuation.items():
                description.append(f"- {key}: {value}")
            description.append("")
        
        # Adicionar informações setoriais se disponíveis
        if "sector_data" in data:
            sector_data = data["sector_data"]
            description.append("DADOS SETORIAIS:")
            for key, value in sector_data.items():
                if isinstance(value, dict):
                    description.append(f"{key}:")
                    for subkey, subvalue in value.items():
                        description.append(f"  - {subkey}: {subvalue}")
                else:
                    description.append(f"- {key}: {value}")
            description.append("")
        
        # Adicionar informações de governança e ESG se disponíveis
        if "esg_data" in data:
            esg_data = data["esg_data"]
            description.append("DADOS ESG:")
            for key, value in esg_data.items():
                description.append(f"- {key}: {value}")
            description.append("")
        
        return "\n".join(description)
    
    def _parse_non_json_response(self, response: str, company: str, sector: str, start_time: float) -> Dict[str, Any]:
        """
        Tenta extrair informações de uma resposta que não está em formato JSON
        
        Args:
            response: Texto da resposta
            company: Nome da empresa
            sector: Setor da empresa
            start_time: Tempo de início do processamento
            
        Returns:
            Dicionário com resultados da análise
        """
        result = {
            "rating": "neutro",
            "financial_health": "regular",
            "financial_metrics": {
                "key_positive": [],
                "key_negative": []
            },
            "valuation": {
                "current_multiples": {},
                "fair_value_assessment": "Não disponível",
                "upside_potential": "Não determinado"
            },
            "competitive_analysis": [],
            "growth_outlook": "Indefinido",
            "risks": [],
            "opportunities": [],
            "confidence": 30,
            "investment_thesis": "Análise incompleta",
            "company": company,
            "sector": sector,
            "processing_time": time.time() - start_time
        }
        
        # Tentar extrair rating/recomendação
        rating_terms = {
            "forte compra": ["forte compra", "strong buy", "compra agressiva"],
            "compra": ["compra", "buy", "recomendação de compra"],
            "neutro": ["neutro", "neutral", "manter", "hold"],
            "venda": ["venda", "sell", "recomendação de venda"],
            "forte venda": ["forte venda", "strong sell", "venda agressiva"]
        }
        
        for rating, keywords in rating_terms.items():
            if any(kw in response.lower() for kw in keywords):
                result["rating"] = rating
                break
        
        # Tentar extrair saúde financeira
        health_terms = {
            "excelente": ["excelente", "excellent", "ótima", "muito boa"],
            "boa": ["boa", "good", "saudável", "positiva"],
            "regular": ["regular", "média", "moderada", "average"],
            "fraca": ["fraca", "weak", "deteriorando", "ruim"],
            "ruim": ["ruim", "má", "muito ruim", "poor", "péssima"]
        }
        
        for health, keywords in health_terms.items():
            if any(kw in response.lower() for kw in keywords):
                result["financial_health"] = health
                break
        
        # Extrair métricas positivas e negativas
        positive_section = False
        negative_section = False
        
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            
            # Identificar seções
            if any(term in line.lower() for term in ["pontos positivos", "pontos fortes", "positive", "forças"]):
                positive_section = True
                negative_section = False
                continue
            
            if any(term in line.lower() for term in ["pontos negativos", "pontos fracos", "negative", "fraquezas"]):
                positive_section = False
                negative_section = True
                continue
            
            # Se linha começa com - ou * e estamos em uma seção
            if line and (line.startswith("-") or line.startswith("*") or line.startswith("•")):
                point = line[1:].strip()
                if positive_section and point:
                    result["financial_metrics"]["key_positive"].append(point)
                elif negative_section and point:
                    result["financial_metrics"]["key_negative"].append(point)
        
        # Extrair múltiplos se presentes
        for multiple in self.market_multiples:
            for line in lines:
                if multiple in line and ":" in line:
                    try:
                        value = line.split(":", 1)[1].strip()
                        # Tentar extrair apenas o valor numérico
                        import re
                        nums = re.findall(r'\d+\.?\d*', value)
                        if nums:
                            result["valuation"]["current_multiples"][multiple] = float(nums[0])
                    except:
                        continue
        
        # Tentar extrair tese de investimento ou conclusão
        thesis_section = False
        thesis_lines = []
        
        for i, line in enumerate(lines):
            if any(term in line.lower() for term in ["tese", "thesis", "conclusão", "conclusion", "recomendação"]):
                thesis_section = True
                continue
            
            if thesis_section and line.strip():
                thesis_lines.append(line.strip())
                # Limitar a algumas linhas após o título da seção
                if len(thesis_lines) >= 3:
                    break
        
        if thesis_lines:
            result["investment_thesis"] = " ".join(thesis_lines)
        
        # Tentar extrair riscos
        risks_section = False
        for line in lines:
            if any(term in line.lower() for term in ["riscos", "risks", "ameaças", "threats"]):
                risks_section = True
                continue
            
            if risks_section and line.strip() and (line.startswith("-") or line.startswith("*") or line.startswith("•")):
                risk = line[1:].strip()
                if risk:
                    result["risks"].append(risk)
        
        # Tentar extrair oportunidades
        opportunities_section = False
        for line in lines:
            if any(term in line.lower() for term in ["oportunidades", "opportunities", "potential"]):
                opportunities_section = True
                continue
            
            if opportunities_section and line.strip() and (line.startswith("-") or line.startswith("*") or line.startswith("•")):
                opportunity = line[1:].strip()
                if opportunity:
                    result["opportunities"].append(opportunity)
        
        return result
    
    def analyze_comparative(self, companies_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Realiza análise fundamentalista comparativa entre múltiplas empresas
        
        Args:
            companies_data: Dicionário com dados de múltiplas empresas
            
        Returns:
            Análise fundamentalista comparativa
        """
        start_time = time.time()
        
        if not companies_data or len(companies_data) < 2:
            return {
                "error": "São necessárias pelo menos duas empresas para análise comparativa",
                "processing_time": time.time() - start_time
            }
        
        # Analisar cada empresa individualmente
        individual_analyses = {}
        for company, data in companies_data.items():
            individual_analyses[company] = self.analyze(data)
        
        # Comparação setorial (se todas as empresas forem do mesmo setor)
        sectors = set(analysis.get("sector", "") for analysis in individual_analyses.values())
        same_sector = len(sectors) == 1 and "" not in sectors
        
        # Preparar prompt para o modelo com dados comparativos
        comparative_text = []
        for company, analysis in individual_analyses.items():
            comparative_text.append(f"EMPRESA: {company}")
            comparative_text.append(f"Setor: {analysis.get('sector', 'N/A')}")
            comparative_text.append(f"Rating: {analysis.get('rating', 'N/A')}")
            comparative_text.append(f"Saúde Financeira: {analysis.get('financial_health', 'N/A')}")
            
            # Adicionar múltiplos
            multiples = analysis.get("valuation", {}).get("current_multiples", {})
            if multiples:
                comparative_text.append("Múltiplos:")
                for key, value in multiples.items():
                    comparative_text.append(f"- {key}: {value}")
            
            # Adicionar pontos positivos e negativos
            metrics = analysis.get("financial_metrics", {})
            positives = metrics.get("key_positive", [])
            negatives = metrics.get("key_negative", [])
            
            if positives:
                comparative_text.append("Pontos Positivos:")
                for point in positives[:3]:  # Limitar a 3 pontos
                    comparative_text.append(f"- {point}")
            
            if negatives:
                comparative_text.append("Pontos Negativos:")
                for point in negatives[:3]:  # Limitar a 3 pontos
                    comparative_text.append(f"- {point}")
                    
            comparative_text.append("")  # Linha vazia entre empresas
        
        # Prompt para análise comparativa
        prompt = f"""
        Realize uma análise fundamentalista comparativa entre as seguintes empresas:
        
        {'\n'.join(comparative_text)}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "comparative_ranking": [lista de empresas em ordem de atratividade],
            "key_strengths": {{
                "empresa1": [pontos fortes],
                "empresa2": [pontos fortes],
                ...
            }},
            "relative_valuation": {{
                "empresa1": "sub/superavaliada em relação aos pares",
                "empresa2": "sub/superavaliada em relação aos pares",
                ...
            }},
            "best_investment": "melhor empresa para investir",
            "investment_rationale": "justificativa para a melhor escolha",
            "confidence": (porcentagem de 0 a 100),
            "sector_outlook": "perspectiva para o setor (se aplicável)"
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
                result["companies"] = list(companies_data.keys())
                result["same_sector"] = same_sector
                result["sector"] = next(iter(sectors)) if same_sector else "Múltiplos setores"
                result["processing_time"] = time.time() - start_time
                
                # Adicionar análises individuais
                result["individual_analyses"] = individual_analyses
                
                return result
            else:
                # Retornar apenas as análises individuais se a resposta não puder ser parseada
                return {
                    "error": "Não foi possível processar a análise comparativa",
                    "companies": list(companies_data.keys()),
                    "same_sector": same_sector,
                    "individual_analyses": individual_analyses,
                    "processing_time": time.time() - start_time
                }
                
        except Exception as e:
            return {
                "error": f"Erro na análise comparativa: {str(e)}",
                "companies": list(companies_data.keys()),
                "individual_analyses": individual_analyses,
                "processing_time": time.time() - start_time
            }
    
    def analyze_industry(self, industry_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza análise fundamentalista de um setor ou indústria
        
        Args:
            industry_data: Dados do setor para análise
            
        Returns:
            Análise fundamentalista do setor
        """
        start_time = time.time()
        
        # Extrair informações básicas do setor
        industry_name = industry_data.get("industry", "Não especificado")
        description = industry_data.get("description", "")
        
        # Converter dados estruturados para texto
        if isinstance(industry_data.get("metrics", None), dict):
            metrics_text = "\n".join([f"{k}: {v}" for k, v in industry_data["metrics"].items()])
        else:
            metrics_text = industry_data.get("metrics_text", "")
        
        if isinstance(industry_data.get("trends", None), list):
            trends_text = "\n".join([f"- {t}" for t in industry_data["trends"]])
        else:
            trends_text = industry_data.get("trends_text", "")
        
        if isinstance(industry_data.get("companies", None), list):
            companies_text = "\n".join([f"- {c}" for c in industry_data["companies"]])
        else:
            companies_text = industry_data.get("companies_text", "")
        
        # Preparar prompt para o modelo
        prompt = f"""
        Realize uma análise fundamentalista do seguinte setor:
        
        SETOR: {industry_name}
        
        DESCRIÇÃO:
        {description}
        
        MÉTRICAS DO SETOR:
        {metrics_text}
        
        TENDÊNCIAS:
        {trends_text}
        
        PRINCIPAIS EMPRESAS:
        {companies_text}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "sector_attractiveness": (string: "muito atrativo", "atrativo", "neutro", "pouco atrativo", "não atrativo"),
            "growth_potential": (string: "alto", "moderado", "baixo"),
            "competitive_landscape": "descrição do cenário competitivo",
            "key_trends": [principais tendências],
            "investment_thesis": "tese de investimento para o setor",
            "risks": [principais riscos do setor],
            "opportunities": [principais oportunidades do setor],
            "recommended_companies": [empresas recomendadas, se aplicável],
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
                result["industry"] = industry_name
                result["processing_time"] = time.time() - start_time
                
                return result
            else:
                # Falha ao extrair JSON
                return {
                    "error": "Não foi possível processar a análise setorial",
                    "industry": industry_name,
                    "processing_time": time.time() - start_time
                }
                
        except Exception as e:
            return {
                "error": f"Erro na análise setorial: {str(e)}",
                "industry": industry_name,
                "processing_time": time.time() - start_time
            }
