"""
Especialista de Dados de Fallback para o Sistema EzioFilhoUnified.
Este especialista fornece dados alternativos quando as fontes primárias falham.
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# Configuração de logging
logger = logging.getLogger("eziofilho.fallback_data")

class FallbackDataExpert:
    """
    Especialista que fornece dados de fallback quando fontes primárias não estão disponíveis.
    Implementa mecanismos de armazenamento local, caching e recuperação de dados históricos.
    """
    
    def __init__(
        self,
        data_dir: str = "./data/fallback",
        cache_retention_days: int = 7,
        sources_config: str = "./config/fallback_sources.json",
        max_retries: int = 3
    ):
        """
        Inicializa o especialista de fallback de dados.
        
        Args:
            data_dir: Diretório para armazenar dados de fallback
            cache_retention_days: Número de dias para manter dados em cache
            sources_config: Caminho para o arquivo de configuração de fontes alternativas
            max_retries: Número máximo de tentativas para recuperar dados
        """
        self.data_dir = data_dir
        self.cache_retention_days = cache_retention_days
        self.sources_config = sources_config
        self.max_retries = max_retries
        self.sources = {}
        self.initialize()
        
    def initialize(self) -> None:
        """Inicializa o especialista carregando configurações e preparando diretórios"""
        logger.info("Inicializando FallbackDataExpert")
        
        # Criar diretório de dados se não existir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            logger.info(f"Criado diretório de dados: {self.data_dir}")
        
        # Carregar configuração de fontes alternativas
        try:
            if os.path.exists(self.sources_config):
                with open(self.sources_config, 'r') as f:
                    self.sources = json.load(f)
                logger.info(f"Carregadas {len(self.sources)} fontes alternativas de dados")
            else:
                logger.warning(f"Arquivo de configuração não encontrado: {self.sources_config}")
                self.sources = self._get_default_sources()
        except Exception as e:
            logger.error(f"Erro ao carregar configurações: {str(e)}")
            self.sources = self._get_default_sources()
    
    def _get_default_sources(self) -> Dict[str, Dict[str, str]]:
        """Define fontes padrão quando a configuração não está disponível"""
        return {
            "financial_data": {
                "primary": "yahoo_finance_api",
                "secondary": "alpha_vantage_api",
                "fallback": "local_historical_data",
                "last_resort": "synthetic_data_generator"
            },
            "market_news": {
                "primary": "bloomberg_api",
                "secondary": "reuters_api",
                "fallback": "financial_times_api",
                "last_resort": "local_news_archive"
            },
            "economic_indicators": {
                "primary": "fed_reserve_api",
                "secondary": "world_bank_api",
                "fallback": "local_economic_data",
                "last_resort": "estimated_indicators"
            }
        }
    
    def get_data(
        self, 
        data_type: str, 
        query_params: Dict[str, Any],
        force_fallback: bool = False
    ) -> Dict[str, Any]:
        """
        Obtém dados do tipo especificado, recorrendo a fontes alternativas se necessário.
        
        Args:
            data_type: Tipo de dados solicitados (ex: "financial_data", "market_news")
            query_params: Parâmetros para consulta de dados
            force_fallback: Se True, ignora fonte primária e vai direto para fallback
            
        Returns:
            Dicionário contendo os dados solicitados e metadados sobre a fonte
        """
        if data_type not in self.sources:
            logger.error(f"Tipo de dados não configurado: {data_type}")
            return self._generate_error_response(f"Tipo de dados não suportado: {data_type}")
        
        # Verificar cache primeiro
        cached_data = self._check_cache(data_type, query_params)
        if cached_data:
            logger.info(f"Dados encontrados em cache para {data_type}")
            return {
                "data": cached_data,
                "source": "cache",
                "timestamp": datetime.now().isoformat(),
                "is_fallback": True
            }
        
        # Se forçar fallback, pular fontes primárias
        if not force_fallback:
            # Tentar fonte primária
            try:
                primary_source = self.sources[data_type]["primary"]
                logger.info(f"Tentando fonte primária: {primary_source}")
                # Código para consultar fonte primária seria implementado aqui
                # ...
                
                # Simulando falha na fonte primária para este exemplo
                raise ConnectionError("Simulação de falha na fonte primária")
            except Exception as e:
                logger.warning(f"Falha na fonte primária: {str(e)}")
        
        # Tentar fontes alternativas em ordem
        for source_type in ["secondary", "fallback", "last_resort"]:
            if source_type in self.sources[data_type]:
                try:
                    source_name = self.sources[data_type][source_type]
                    logger.info(f"Tentando fonte alternativa: {source_name}")
                    
                    # Aqui implementaríamos a lógica real para cada tipo de fonte
                    # Para este exemplo, simularemos dados para a fonte "local_historical_data"
                    if source_name == "local_historical_data":
                        data = self._get_local_historical_data(data_type, query_params)
                        if data:
                            # Armazenar em cache para uso futuro
                            self._store_in_cache(data_type, query_params, data)
                            return {
                                "data": data,
                                "source": source_name,
                                "timestamp": datetime.now().isoformat(),
                                "is_fallback": True
                            }
                    
                    # Simulando dados sintéticos como último recurso
                    elif source_name == "synthetic_data_generator":
                        data = self._generate_synthetic_data(data_type, query_params)
                        return {
                            "data": data,
                            "source": source_name,
                            "timestamp": datetime.now().isoformat(),
                            "is_fallback": True,
                            "is_synthetic": True
                        }
                
                except Exception as e:
                    logger.warning(f"Falha na fonte {source_name}: {str(e)}")
        
        # Se chegou aqui, nenhuma fonte funcionou
        logger.error(f"Todas as fontes falharam para {data_type}")
        return self._generate_error_response("Dados indisponíveis em todas as fontes")
    
    def _check_cache(self, data_type: str, query_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Verifica se existem dados em cache para os parâmetros fornecidos"""
        cache_key = self._generate_cache_key(data_type, query_params)
        cache_file = os.path.join(self.data_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Verificar se cache ainda é válido
                cache_date = datetime.fromisoformat(cached_data.get("timestamp", "2000-01-01"))
                expiry_date = datetime.now() - timedelta(days=self.cache_retention_days)
                
                if cache_date > expiry_date:
                    return cached_data.get("data")
                else:
                    logger.info(f"Cache expirado para {cache_key}")
                    return None
            except Exception as e:
                logger.warning(f"Erro ao ler cache: {str(e)}")
                return None
        return None
    
    def _store_in_cache(self, data_type: str, query_params: Dict[str, Any], data: Dict[str, Any]) -> None:
        """Armazena dados em cache para uso futuro"""
        try:
            cache_key = self._generate_cache_key(data_type, query_params)
            cache_file = os.path.join(self.data_dir, f"{cache_key}.json")
            
            cache_data = {
                "data": data,
                "query_params": query_params,
                "timestamp": datetime.now().isoformat(),
                "data_type": data_type
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Dados armazenados em cache: {cache_key}")
        except Exception as e:
            logger.error(f"Erro ao armazenar em cache: {str(e)}")
    
    def _generate_cache_key(self, data_type: str, query_params: Dict[str, Any]) -> str:
        """Gera uma chave única para o cache baseada no tipo de dados e parâmetros"""
        # Criar uma versão ordenada e estável dos parâmetros
        param_str = json.dumps(query_params, sort_keys=True)
        import hashlib
        return f"{data_type}_{hashlib.md5(param_str.encode()).hexdigest()}"
    
    def _get_local_historical_data(self, data_type: str, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Recupera dados históricos armazenados localmente"""
        # Implementação real recuperaria dados de arquivos locais
        # Para este exemplo, retornamos dados simulados
        if data_type == "financial_data":
            return {
                "ticker": query_params.get("ticker", "UNKNOWN"),
                "historical_prices": [
                    {"date": "2025-05-19", "open": 150.23, "close": 152.45, "high": 153.20, "low": 149.90, "volume": 1234567},
                    {"date": "2025-05-18", "open": 149.56, "close": 150.23, "high": 151.78, "low": 148.33, "volume": 1122334},
                    {"date": "2025-05-17", "open": 148.89, "close": 149.56, "high": 150.11, "low": 147.67, "volume": 1345678}
                ],
                "metadata": {
                    "source": "local_historical_data",
                    "last_updated": "2025-05-19T23:59:59",
                    "is_complete": True
                }
            }
        elif data_type == "market_news":
            return {
                "headlines": [
                    {"title": "Mercados fecham em alta após anúncio do Fed", "date": "2025-05-19", "source": "cached_news"},
                    {"title": "Empresas de tecnologia lideram ganhos", "date": "2025-05-18", "source": "cached_news"},
                ],
                "metadata": {
                    "source": "local_news_archive",
                    "last_updated": "2025-05-19T18:30:00",
                    "coverage": "partial"
                }
            }
        return {}
    
    def _generate_synthetic_data(self, data_type: str, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Gera dados sintéticos como último recurso quando nenhuma fonte está disponível"""
        import random
        
        if data_type == "financial_data":
            ticker = query_params.get("ticker", "UNKNOWN")
            base_price = random.uniform(50, 500)
            
            # Gerar preços sintéticos com tendência aleatória
            trend = random.uniform(-0.02, 0.02)
            volatility = random.uniform(0.005, 0.02)
            
            prices = []
            current_price = base_price
            
            for i in range(10):
                date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                daily_change = random.normalvariate(trend, volatility)
                open_price = current_price
                close_price = open_price * (1 + daily_change)
                high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
                low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
                volume = int(random.uniform(100000, 10000000))
                
                prices.append({
                    "date": date,
                    "open": round(open_price, 2),
                    "close": round(close_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "volume": volume
                })
                
                current_price = close_price
            
            return {
                "ticker": ticker,
                "historical_prices": prices,
                "metadata": {
                    "source": "synthetic_data_generator",
                    "generation_date": datetime.now().isoformat(),
                    "warning": "ESTES DADOS SÃO SINTÉTICOS E NÃO DEVEM SER USADOS PARA DECISÕES REAIS"
                }
            }
        
        return {
            "warning": "Dados sintéticos gerados como último recurso",
            "generation_date": datetime.now().isoformat(),
            "data_type": data_type,
            "is_synthetic": True
        }
    
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Gera resposta padrão para erros"""
        return {
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }
    
    def clean_expired_cache(self) -> int:
        """
        Remove entradas de cache expiradas.
        
        Returns:
            Número de arquivos de cache removidos
        """
        try:
            count = 0
            expiry_date = datetime.now() - timedelta(days=self.cache_retention_days)
            
            for filename in os.listdir(self.data_dir):
                if not filename.endswith('.json'):
                    continue
                    
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        cache_data = json.load(f)
                    
                    cache_date = datetime.fromisoformat(cache_data.get("timestamp", "2000-01-01"))
                    if cache_date < expiry_date:
                        os.remove(filepath)
                        count += 1
                except Exception as e:
                    logger.warning(f"Erro ao processar arquivo de cache {filename}: {str(e)}")
            
            logger.info(f"Limpeza de cache: {count} arquivos removidos")
            return count
        except Exception as e:
            logger.error(f"Erro durante limpeza de cache: {str(e)}")
            return 0
    
    def rebuild_cache_index(self) -> bool:
        """
        Reconstrói o índice de cache para otimizar consultas.
        
        Returns:
            True se o índice foi reconstruído com sucesso, False caso contrário
        """
        try:
            index = {}
            
            for filename in os.listdir(self.data_dir):
                if not filename.endswith('.json'):
                    continue
                    
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        cache_data = json.load(f)
                    
                    data_type = cache_data.get("data_type")
                    if data_type:
                        if data_type not in index:
                            index[data_type] = []
                        
                        index[data_type].append({
                            "filename": filename,
                            "query_params": cache_data.get("query_params", {}),
                            "timestamp": cache_data.get("timestamp")
                        })
                except Exception as e:
                    logger.warning(f"Erro ao indexar arquivo {filename}: {str(e)}")
            
            # Salvar índice
            index_path = os.path.join(self.data_dir, "_cache_index.json")
            with open(index_path, 'w') as f:
                json.dump(index, f, indent=2)
            
            logger.info(f"Índice de cache reconstruído: {sum(len(items) for items in index.values())} entradas")
            return True
        except Exception as e:
            logger.error(f"Erro ao reconstruir índice de cache: {str(e)}")
            return False