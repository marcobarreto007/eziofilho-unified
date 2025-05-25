#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CryptoExpert - Especialista em Análise de Criptomoedas
-----------------------------------------------------
Analisa mercados de criptomoedas, tecnologia blockchain e DeFi.

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

class CryptoExpert(Phi2Expert):
    """
    Especialista em análise de criptomoedas baseado em Phi-2.
    
    Capacidades:
    - Análise técnica e fundamental de criptomoedas
    - Avaliação de protocolos blockchain e projetos DeFi
    - Análise de tokenomics e modelos de valor
    - Monitoramento de métricas on-chain e de sentimento
    - Avaliação de tendências de mercado e adoção
    - Análise de riscos e oportunidades no ecossistema cripto
    """
    
    def __init__(self, **kwargs):
        """
        Inicializa o especialista de criptomoedas
        
        Args:
            **kwargs: Parâmetros adicionais para a classe base
        """
        # Definir mensagem de sistema especializada
        system_message = """
        Você é um especialista em análise de criptomoedas com ampla experiência em mercados blockchain.
        Sua tarefa é analisar mercados de criptomoedas, avaliar protocolos blockchain e projetos DeFi,
        analisar tokenomics e modelos de valor, monitorar métricas on-chain e de sentimento,
        identificar tendências de mercado e adoção, e avaliar riscos e oportunidades no ecossistema cripto.
        
        Regras:
        1. Sempre retorne uma análise estruturada com campos claros
        2. Considere tanto análise técnica quanto fundamentalista em suas avaliações
        3. Avalie métricas específicas de cripto, como dados on-chain e tokenomics
        4. Considere o contexto macroeconômico e regulatório para o mercado cripto
        5. Analise modelos de valoração específicos para ativos digitais
        6. Identifique riscos específicos e questões de segurança para projetos cripto
        """
        
        super().__init__(
            expert_type="crypto_expert",
            domain="quant",
            specialization="crypto_analysis",
            system_message=system_message,
            **kwargs
        )
        
        # Configurações específicas para análise de criptomoedas
        self.crypto_sectors = [
            "Layer 1", "Layer 2", "DeFi", "CeFi", "NFT", "GameFi", 
            "Metaverse", "Privacy Coins", "Stablecoins", "Exchange Tokens",
            "Oracle Networks", "Infrastructure", "Web3"
        ]
        
        self.chain_metrics = [
            "Hash Rate", "Active Addresses", "Transaction Volume", 
            "Network Value", "Staking Ratio", "Transaction Fee", 
            "NVTV Ratio", "TVL", "MVRV Ratio", "Coin Days Destroyed",
            "Exchange Flows", "Supply Distribution"
        ]
        
        self.market_cycles = [
            "Early Bull", "Mid Bull", "Late Bull", "Bull Market Top",
            "Early Bear", "Mid Bear", "Late Bear", "Bear Market Bottom",
            "Accumulation", "Distribution", "Uncertainty"
        ]
    
    def analyze(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analisa mercados de criptomoedas e fornece insights
        
        Args:
            input_data: Descrição textual ou dicionário com dados de criptomoedas
            
        Returns:
            Resultado da análise de criptomoedas
        """
        start_time = time.time()
        
        # Extrair dados se input for dicionário
        if isinstance(input_data, dict):
            if "crypto_data" in input_data:
                data_type = "structured"
                description = self._format_structured_data(input_data)
                asset = input_data.get("asset", "desconhecido")
                protocol = input_data.get("protocol", "desconhecido")
            else:
                data_type = "text"
                description = input_data.get("description", "")
                asset = input_data.get("asset", "desconhecido")
                protocol = input_data.get("protocol", "desconhecido")
        else:
            # Input é texto descritivo
            data_type = "text"
            description = input_data
            asset = "desconhecido"
            protocol = "desconhecido"
        
        # Verificar se temos dados suficientes
        if not description or len(description.strip()) < 20:
            return {
                "error": "Dados insuficientes para análise de criptomoedas",
                "market_assessment": "indefinido",
                "confidence": 0,
                "processing_time": time.time() - start_time
            }
        
        # Preparar prompt para o modelo
        prompt = f"""
        Realize uma análise detalhada de criptomoedas com base nos seguintes dados:
        
        ATIVO: {asset}
        PROTOCOLO: {protocol}
        
        DADOS: {description}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "market_assessment": {{
                "cycle_position": "posição no ciclo de mercado",
                "sentiment": "sentimento atual do mercado",
                "momentum": "momentum de curto prazo"
            }},
            "fundamental_analysis": {{
                "use_case": "avaliação do caso de uso",
                "technology": "avaliação da tecnologia/infraestrutura",
                "adoption_metrics": "métricas de adoção",
                "competitive_position": "posição competitiva"
            }},
            "tokenomics": {{
                "supply_dynamics": "dinâmica de oferta e inflação",
                "value_accrual": "mecanismos de captura de valor",
                "token_utility": "utilidade do token",
                "distribution": "distribuição do token"
            }},
            "on_chain_metrics": {{
                "network_activity": "atividade da rede",
                "holder_behavior": "comportamento dos detentores",
                "exchange_flows": "fluxos de/para exchanges",
                "key_indicators": [indicadores on-chain relevantes]
            }},
            "risk_assessment": {{
                "technical_risks": [riscos técnicos],
                "regulatory_risks": [riscos regulatórios],
                "market_risks": [riscos de mercado],
                "security_considerations": "considerações de segurança"
            }},
            "technical_outlook": {{
                "key_levels": [níveis técnicos importantes],
                "patterns": [padrões identificados],
                "indicators": [indicadores técnicos relevantes]
            }},
            "investment_recommendation": {{
                "outlook": "perspectiva geral (bullish/bearish/neutral)",
                "time_horizon": "horizonte de tempo recomendado",
                "entry_strategy": "estratégia de entrada",
                "position_sizing": "sugestão de dimensionamento de posição"
            }},
            "catalysts": [catalisadores potenciais a monitorar],
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
                result["asset"] = asset
                result["protocol"] = protocol
                result["data_type"] = data_type
                result["processing_time"] = time.time() - start_time
                
                return result
            else:
                # Falha ao extrair JSON, tentar parsear manualmente
                return self._parse_non_json_response(response, asset, protocol, start_time)
                
        except Exception as e:
            return {
                "error": f"Erro na análise de criptomoedas: {str(e)}",
                "market_assessment": "indefinido",
                "confidence": 0,
                "asset": asset,
                "protocol": protocol,
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
        crypto_data = data.get("crypto_data", {})
        
        # Formatar dados de mercado
        market_data = crypto_data.get("market_data", {})
        market_str = "DADOS DE MERCADO:\n"
        for key, value in market_data.items():
            market_str += f"- {key}: {value}\n"
        
        # Formatar dados on-chain
        onchain_data = crypto_data.get("onchain_data", {})
        onchain_str = "DADOS ON-CHAIN:\n"
        for key, value in onchain_data.items():
            onchain_str += f"- {key}: {value}\n"
        
        # Formatar tokenomics
        tokenomics = crypto_data.get("tokenomics", {})
        token_str = "TOKENOMICS:\n"
        for key, value in tokenomics.items():
            token_str += f"- {key}: {value}\n"
        
        # Formatar dados fundamentais
        fundamentals = crypto_data.get("fundamentals", {})
        fund_str = "DADOS FUNDAMENTAIS:\n"
        for key, value in fundamentals.items():
            fund_str += f"- {key}: {value}\n"
        
        # Formatar conjunto
        formatted_data = f"""
        {data.get('description', 'Sem descrição disponível')}
        
        {market_str}
        
        {onchain_str}
        
        {token_str}
        
        {fund_str}
        
        INFORMAÇÕES ADICIONAIS:
        {data.get('additional_info', '')}
        """
        
        return formatted_data
    
    def _parse_non_json_response(self, response: str, asset: str, protocol: str, start_time: float) -> Dict[str, Any]:
        """
        Tenta extrair informações relevantes de uma resposta não-JSON
        
        Args:
            response: Texto da resposta
            asset: Nome do ativo
            protocol: Nome do protocolo
            start_time: Hora de início do processamento
            
        Returns:
            Dicionário com informações extraídas
        """
        result = {
            "asset": asset,
            "protocol": protocol,
            "processing_time": time.time() - start_time,
            "format_error": "Resposta não veio no formato JSON esperado"
        }
        
        # Tentar extrair posição no ciclo
        lower_resp = response.lower()
        
        # Extrair ciclo de mercado
        for cycle in self.market_cycles:
            if cycle.lower() in lower_resp:
                result["market_assessment"] = {"cycle_position": cycle}
                break
        else:
            result["market_assessment"] = {"cycle_position": "não identificado"}
        
        # Extrair perspectiva
        if "bullish" in lower_resp or "alta" in lower_resp:
            outlook = "bullish"
        elif "bearish" in lower_resp or "baixa" in lower_resp:
            outlook = "bearish"
        else:
            outlook = "neutral"
        
        result["investment_recommendation"] = {"outlook": outlook}
        
        # Extrair confiança
        confidence = 50  # valor padrão
        result["confidence"] = confidence
        
        # Texto original
        result["original_response"] = response
        
        return result
    
    def analyze_tokenomics(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa tokenomics e modelo econômico de um projeto
        
        Args:
            token_data: Dados de tokenomics para análise
            
        Returns:
            Análise detalhada de tokenomics
        """
        description = f"""
        Analise os tokenomics do seguinte projeto:
        
        DADOS DE TOKEN:
        """
        
        for key, value in token_data.items():
            if isinstance(value, dict):
                description += f"- {key}:\n"
                for sub_key, sub_value in value.items():
                    description += f"  * {sub_key}: {sub_value}\n"
            else:
                description += f"- {key}: {value}\n"
        
        prompt = f"""
        {description}
        
        Forneça sua análise de tokenomics no seguinte formato JSON:
        {{
            "supply_analysis": {{
                "current_supply": "análise da oferta atual",
                "max_supply": "análise da oferta máxima",
                "circulating_supply": "análise da oferta circulante",
                "inflation_schedule": "avaliação do cronograma de inflação"
            }},
            "value_accrual": {{
                "mechanisms": [mecanismos de captura de valor],
                "effectiveness": "efetividade dos mecanismos",
                "sustainability": "sustentabilidade do modelo"
            }},
            "distribution": {{
                "initial_allocation": "avaliação da alocação inicial",
                "current_distribution": "avaliação da distribuição atual",
                "concentration_risks": "riscos de concentração"
            }},
            "token_utility": [utilidades do token e sua avaliação],
            "economic_alignment": "alinhamento econômico entre stakeholders",
            "comparative_analysis": "comparação com tokenomics de projetos semelhantes",
            "key_risks": [principais riscos relacionados ao modelo econômico],
            "improvement_opportunities": [oportunidades de melhoria no modelo]
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
            return {"error": f"Erro na análise de tokenomics: {str(e)}"}
    
    def analyze_on_chain_metrics(self, chain_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa métricas on-chain e comportamento de rede
        
        Args:
            chain_data: Dados on-chain para análise
            
        Returns:
            Análise detalhada de métricas on-chain
        """
        description = f"""
        Analise as seguintes métricas on-chain:
        
        DADOS ON-CHAIN:
        """
        
        for key, value in chain_data.items():
            if isinstance(value, dict):
                description += f"- {key}:\n"
                for sub_key, sub_value in value.items():
                    description += f"  * {sub_key}: {sub_value}\n"
            else:
                description += f"- {key}: {value}\n"
        
        prompt = f"""
        {description}
        
        Forneça sua análise de métricas on-chain no seguinte formato JSON:
        {{
            "network_health": {{
                "activity_level": "nível de atividade da rede",
                "growth_metrics": "métricas de crescimento",
                "comparison_to_history": "comparação com dados históricos"
            }},
            "address_analysis": {{
                "active_addresses": "análise de endereços ativos",
                "new_addresses": "análise de novos endereços",
                "whale_behavior": "comportamento das baleias",
                "distribution_trends": "tendências de distribuição"
            }},
            "transaction_analysis": {{
                "volume_assessment": "avaliação de volume de transações",
                "fee_trends": "tendências de taxas",
                "transaction_types": "análise dos tipos de transação"
            }},
            "exchange_flows": {{
                "inflows": "análise de entradas em exchanges",
                "outflows": "análise de saídas de exchanges",
                "net_flow": "fluxo líquido",
                "implications": "implicações para o preço"
            }},
            "staking_and_defi": {{
                "staking_ratio": "análise da taxa de staking",
                "liquidity_provisions": "análise de provisão de liquidez",
                "yields": "análise de rendimentos"
            }},
            "bullish_signals": [sinais on-chain bullish],
            "bearish_signals": [sinais on-chain bearish],
            "notable_trends": [tendências notáveis identificadas]
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
            return {"error": f"Erro na análise de métricas on-chain: {str(e)}"}
    
    def evaluate_protocol(self, protocol_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Avalia protocolo blockchain ou projeto DeFi
        
        Args:
            protocol_data: Dados do protocolo para avaliação
            
        Returns:
            Avaliação detalhada do protocolo
        """
        description = f"""
        Avalie o seguinte protocolo blockchain ou projeto DeFi:
        
        DADOS DO PROTOCOLO:
        """
        
        for key, value in protocol_data.items():
            if isinstance(value, dict):
                description += f"- {key}:\n"
                for sub_key, sub_value in value.items():
                    description += f"  * {sub_key}: {sub_value}\n"
            else:
                description += f"- {key}: {value}\n"
        
        prompt = f"""
        {description}
        
        Forneça sua avaliação do protocolo no seguinte formato JSON:
        {{
            "technology_assessment": {{
                "architecture": "avaliação da arquitetura",
                "scalability": "avaliação de escalabilidade",
                "security": "avaliação de segurança",
                "decentralization": "nível de descentralização",
                "interoperability": "capacidade de interoperabilidade"
            }},
            "product_market_fit": {{
                "use_case_validity": "validade do caso de uso",
                "addressable_market": "mercado endereçável",
                "competitive_advantage": "vantagem competitiva",
                "user_experience": "experiência do usuário"
            }},
            "adoption_metrics": {{
                "user_growth": "crescimento de usuários",
                "transaction_volume": "volume de transações",
                "tvl_analysis": "análise de TVL (se aplicável)",
                "ecosystem_development": "desenvolvimento do ecossistema"
            }},
            "team_and_community": {{
                "team_assessment": "avaliação da equipe",
                "development_activity": "atividade de desenvolvimento",
                "community_engagement": "engajamento da comunidade",
                "governance": "estrutura de governança"
            }},
            "risk_assessment": {{
                "technical_risks": [riscos técnicos],
                "economic_risks": [riscos econômicos],
                "competitive_risks": [riscos competitivos],
                "regulatory_risks": [riscos regulatórios]
            }},
            "development_roadmap": "avaliação do roadmap de desenvolvimento",
            "investment_thesis": "tese de investimento",
            "valuation_assessment": "avaliação de valoração"
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
            return {"error": f"Erro na avaliação do protocolo: {str(e)}"}
    
    def assess_crypto_market_cycle(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Avalia ciclo atual do mercado de criptomoedas
        
        Args:
            market_data: Dados de mercado para análise
            
        Returns:
            Avaliação do ciclo de mercado
        """
        description = f"""
        Avalie o ciclo atual do mercado de criptomoedas com base nos seguintes dados:
        
        DADOS DE MERCADO:
        """
        
        for key, value in market_data.items():
            if isinstance(value, dict):
                description += f"- {key}:\n"
                for sub_key, sub_value in value.items():
                    description += f"  * {sub_key}: {sub_value}\n"
            else:
                description += f"- {key}: {value}\n"
        
        prompt = f"""
        {description}
        
        Forneça sua avaliação do ciclo de mercado no seguinte formato JSON:
        {{
            "cycle_position": {{
                "current_phase": "fase atual do ciclo",
                "cycle_maturity": "maturidade do ciclo",
                "historical_comparison": "comparação com ciclos históricos"
            }},
            "market_indicators": {{
                "price_action": "análise da ação do preço",
                "volume_trends": "tendências de volume",
                "market_dominance": "análise de dominância",
                "sector_rotation": "rotação entre setores"
            }},
            "sentiment_analysis": {{
                "retail_sentiment": "sentimento de varejo",
                "institutional_sentiment": "sentimento institucional",
                "social_metrics": "métricas sociais",
                "fear_and_greed": "índice de medo e ganância"
            }},
            "liquidity_conditions": {{
                "exchange_liquidity": "liquidez em exchanges",
                "stablecoin_flows": "fluxos de stablecoins",
                "funding_rates": "taxas de funding",
                "market_depth": "profundidade de mercado"
            }},
            "macro_correlation": {{
                "correlation_with_equities": "correlação com ações",
                "correlation_with_gold": "correlação com ouro",
                "correlation_with_dollar": "correlação com dólar",
                "narrative_shifts": "mudanças de narrativa"
            }},
            "cycle_forecast": {{
                "short_term": "previsão de curto prazo",
                "medium_term": "previsão de médio prazo",
                "potential_catalysts": [catalisadores potenciais],
                "warning_signals": [sinais de alerta]
            }},
            "positioning_strategy": [estratégias de posicionamento recomendadas para o ciclo atual]
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
            return {"error": f"Erro na avaliação do ciclo de mercado: {str(e)}"}
