#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SentimentAnalystExpert - Especialista em Análise de Sentimento de Mercado
-------------------------------------------------------------------------
Analisa o sentimento em conteúdo financeiro de múltiplas fontes.

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

class SentimentAnalystExpert(Phi2Expert):
    """
    Especialista em análise de sentimento de mercado baseado em Phi-2.
    
    Capacidades:
    - Análise de sentimento em textos financeiros (notícias, tweets, relatórios)
    - Detecção de viés bullish/bearish
    - Classificação multilíngue (47 idiomas)
    - Extração de fatores de sentimento
    - Quantificação da confiança da análise
    """
    
    def __init__(self, **kwargs):
        """
        Inicializa o especialista de análise de sentimento
        
        Args:
            **kwargs: Parâmetros adicionais para a classe base
        """
        # Definir mensagem de sistema especializada
        system_message = """
        Você é um especialista em análise de sentimento de mercado financeiro, capaz de analisar conteúdo em 47 idiomas.
        Sua tarefa é avaliar o sentimento geral (bullish/bearish/neutro) em textos financeiros, identificar fatores
        que influenciam esse sentimento, e avaliar a confiança e a polaridade da análise.
        
        Regras:
        1. Sempre retorne uma análise estruturada com campos claros
        2. Classifique o sentimento na escala de -5 (extremamente bearish) a +5 (extremamente bullish)
        3. Identifique fatores específicos que contribuem para o sentimento
        4. Avalie sua confiança na análise (0-100%)
        5. Inclua entidades relevantes mencionadas (empresas, índices, commodities, etc.)
        6. Seja objetivo e baseie-se nos fatos apresentados
        """
        
        super().__init__(
            expert_type="sentiment_analyst",
            domain="market",
            specialization="sentiment_analysis",
            **kwargs
        )
        
        # Configurações específicas para análise de sentimento
        self.sentiment_scales = {
            "bullish": range(1, 6),     # 1 a 5
            "neutral": range(-1, 2),    # -1, 0, 1
            "bearish": range(-5, 0)     # -5 a -1
        }
        
        # Idiomas suportados
        self.supported_languages = [
            "inglês", "espanhol", "francês", "alemão", "italiano", "português", "russo", 
            "chinês", "japonês", "coreano", "árabe", "hindi", "turco", "holandês", 
            # ... (outros idiomas)
        ]
    
    def analyze(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analisa o sentimento em um texto financeiro
        
        Args:
            input_data: Texto para analisar ou dicionário com chave 'text'
            
        Returns:
            Resultado da análise de sentimento
        """
        start_time = time.time()
        
        # Extrair texto se input for dicionário
        if isinstance(input_data, dict):
            text = input_data.get("text", "")
            source = input_data.get("source", "desconhecido")
            date = input_data.get("date", "")
        else:
            text = input_data
            source = "entrada direta"
            date = time.strftime("%Y-%m-%d")
        
        # Verificar se temos texto para analisar
        if not text or len(text.strip()) < 10:
            return {
                "error": "Texto muito curto ou vazio para análise",
                "sentiment_score": 0,
                "sentiment": "neutral",
                "confidence": 0,
                "processing_time": time.time() - start_time
            }
            
        # Preparar prompt para o modelo
        prompt = f"""
        Analise o seguinte texto financeiro e determine seu sentimento:
        
        TEXTO: {text}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "sentiment_score": (número de -5 a +5, onde -5 é extremamente bearish e +5 é extremamente bullish),
            "sentiment": (string: "bearish", "neutral" ou "bullish"),
            "confidence": (porcentagem de 0 a 100),
            "factors": [lista de fatores que influenciaram sua análise],
            "entities": [lista de entidades relevantes mencionadas],
            "summary": "breve resumo de uma frase sobre o sentimento"
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
                result["source"] = source
                result["date"] = date
                result["processing_time"] = time.time() - start_time
                
                # Validar sentimento
                score = result.get("sentiment_score", 0)
                if score > 0:
                    result["sentiment"] = "bullish"
                elif score < 0:
                    result["sentiment"] = "bearish"
                else:
                    result["sentiment"] = "neutral"
                    
                return result
            else:
                # Falha ao extrair JSON, tentar parsear manualmente
                return self._parse_non_json_response(response, start_time)
                
        except Exception as e:
            return {
                "error": f"Erro na análise: {str(e)}",
                "sentiment_score": 0,
                "sentiment": "neutral",
                "confidence": 0,
                "processing_time": time.time() - start_time
            }
    
    def _parse_non_json_response(self, response: str, start_time: float) -> Dict[str, Any]:
        """
        Tenta extrair informações de uma resposta que não está em formato JSON
        
        Args:
            response: Texto da resposta
            start_time: Tempo de início do processamento
            
        Returns:
            Dicionário com resultados da análise
        """
        result = {
            "sentiment_score": 0,
            "sentiment": "neutral",
            "confidence": 50,
            "factors": [],
            "entities": [],
            "summary": "Não foi possível processar completamente",
            "processing_time": time.time() - start_time
        }
        
        # Tentar extrair pontuação de sentimento
        if "score" in response.lower():
            lines = response.split("\n")
            for line in lines:
                if "score" in line.lower() and ":" in line:
                    try:
                        score_str = line.split(":", 1)[1].strip()
                        # Remover caracteres não numéricos exceto - e .
                        score_str = ''.join(c for c in score_str if c.isdigit() or c in '.-')
                        score = float(score_str)
                        result["sentiment_score"] = score
                        
                        if score > 0:
                            result["sentiment"] = "bullish"
                        elif score < 0:
                            result["sentiment"] = "bearish"
                        else:
                            result["sentiment"] = "neutral"
                            
                        break
                    except:
                        pass
        
        # Tentar extrair confiança
        if "confian" in response.lower() or "confidence" in response.lower():
            lines = response.split("\n")
            for line in lines:
                if ("confian" in line.lower() or "confidence" in line.lower()) and ":" in line:
                    try:
                        conf_str = line.split(":", 1)[1].strip()
                        # Extrair número
                        conf_str = ''.join(c for c in conf_str if c.isdigit() or c == '.')
                        conf = float(conf_str)
                        if conf > 1 and conf <= 100:
                            result["confidence"] = conf
                        elif conf <= 1:
                            result["confidence"] = conf * 100
                        break
                    except:
                        pass
        
        # Extrair fatores se houver
        if "fator" in response.lower() or "factor" in response.lower():
            sections = response.split("\n\n")
            for section in sections:
                if "fator" in section.lower() or "factor" in section.lower():
                    factors = []
                    lines = [l.strip() for l in section.split("\n") if l.strip()]
                    for line in lines[1:]:  # Pular o título
                        if line.startswith("-") or line.startswith("*"):
                            factors.append(line[1:].strip())
                    if factors:
                        result["factors"] = factors
                    break
        
        return result
    
    def analyze_multiple(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analisa o sentimento em múltiplos textos
        
        Args:
            texts: Lista de textos para analisar
            
        Returns:
            Lista de resultados de análise de sentimento
        """
        results = []
        for text in texts:
            result = self.analyze(text)
            results.append(result)
        return results
    
    def analyze_with_context(self, text: str, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa o sentimento considerando o contexto de mercado atual
        
        Args:
            text: Texto para analisar
            market_context: Contexto do mercado (tendências, indicadores, etc.)
            
        Returns:
            Resultado da análise de sentimento com contexto
        """
        # Formatação do contexto de mercado
        context_str = "CONTEXTO DE MERCADO:\n"
        for key, value in market_context.items():
            context_str += f"- {key}: {value}\n"
        
        # Preparar prompt com contexto
        prompt = f"""
        Analise o seguinte texto financeiro considerando o contexto de mercado atual:
        
        {context_str}
        
        TEXTO: {text}
        
        Forneça sua análise no formato JSON (sem explicações adicionais):
        {{
            "sentiment_score": (número de -5 a +5),
            "sentiment": (string: "bearish", "neutral" ou "bullish"),
            "confidence": (porcentagem de 0 a 100),
            "factors": [lista de fatores],
            "context_relevance": (alta, média ou baixa - quão relevante o contexto é para o texto),
            "entities": [lista de entidades relevantes],
            "summary": "breve resumo de uma frase"
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
                result["with_market_context"] = True
                result["processing_time"] = time.time()
                
                return result
            else:
                # Tentar análise sem contexto como fallback
                basic_result = self.analyze(text)
                basic_result["with_market_context"] = False
                return basic_result
                
        except Exception as e:
            return {
                "error": f"Erro na análise com contexto: {str(e)}",
                "sentiment_score": 0,
                "sentiment": "neutral",
                "confidence": 0
            }
