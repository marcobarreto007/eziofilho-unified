#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EzioFinisher - Versão Simplificada
Agente Autônomo para Detecção e Correção de Problemas de Código
"""

import os
import sys
import time
import json
import logging
import argparse
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Configuração de logging
LOG_FILE = "C:\\Users\\anapa\\SuperIA\\EzioFilhoUnified\\eziofinisher.log"
OUTPUT_DIR = "C:\\Users\\anapa\\SuperIA\\EzioFilhoUnified\\autogen_generated"
PROJECT_DIR = "C:\\Users\\anapa\\SuperIA\\EzioFilhoUnified"

# Configurar logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("EzioFinisher")

# Verifica se as pastas necessárias existem e cria se não existirem
os.makedirs(OUTPUT_DIR, exist_ok=True)

class EzioFinisher:
    """Agente autônomo para análise e otimização de código."""
    
    def __init__(self, use_api=False, api_key=None):
        """
        Inicializa o agente EzioFinisher.
        
        Args:
            use_api: Se True, usa a API do OpenAI. Se False, usa modelos locais.
            api_key: Chave de API (se use_api=True)
        """
        self.use_api = use_api
        self.api_key = api_key
        
        # Se usar API, configurar o cliente
        if self.use_api:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info("Cliente OpenAI inicializado com sucesso")
            except ImportError:
                logger.error("Biblioteca OpenAI não encontrada. Instale com 'pip install openai'")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Erro ao inicializar cliente OpenAI: {e}")
                sys.exit(1)
        # Se não usar API, configurar o modelo local
        else:
            try:
                from transformers import pipeline
                
                logger.info("Inicializando modelo local. Isso pode levar alguns minutos...")
                self.pipe = pipeline(
                    "text-generation",
                    model="Salesforce/codegen-350M-mono",  # Modelo pequeno para código
                    max_length=2048
                )
                logger.info("Modelo local inicializado com sucesso")
            except ImportError:
                logger.error("Biblioteca transformers não encontrada. Instale com 'pip install transformers'")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Erro ao inicializar modelo local: {e}")
                sys.exit(1)
        
        # Diretórios importantes
        self.project_dir = PROJECT_DIR
        self.output_dir = OUTPUT_DIR
        
        # Cache de arquivos já analisados
        self.analyzed_files = set()
        
        # Extensões de arquivo que vamos analisar
        self.target_extensions = {".py", ".js", ".html", ".css", ".md", ".txt", ".json", ".yaml", ".yml"}
        
        # Inicializa contador de melhorias
        self.improvements_count = 0
        
        logger.info("EzioFinisher inicializado com sucesso!")
        logger.info(f"Diretório do projeto: {self.project_dir}")
        logger.info(f"Diretório de saída: {self.output_dir}")

    def get_project_structure(self) -> Dict:
        """
        Mapeia a estrutura do projeto.
        
        Returns:
            Dict: Estrutura de diretórios e arquivos do projeto.
        """
        structure = {"dirs": [], "files": []}
        
        for root, dirs, files in os.walk(self.project_dir):
            # Ignorar diretórios gerados e venv
            if "autogen_generated" in root or "venv" in root or ".git" in root or "__pycache__" in root:
                continue
            
            relative_path = os.path.relpath(root, self.project_dir)
            if relative_path == ".":
                relative_path = ""
                
            # Adicionar diretório à estrutura
            if relative_path:
                structure["dirs"].append(relative_path)
            
            # Adicionar arquivos relevantes
            for file in files:
                _, ext = os.path.splitext(file)
                if ext in self.target_extensions:
                    file_path = os.path.join(relative_path, file) if relative_path else file
                    structure["files"].append(file_path)
        
        logger.info(f"Encontrado {len(structure['files'])} arquivos e {len(structure['dirs'])} diretórios relevantes.")
        return structure

    def read_file(self, file_path: str) -> str:
        """
        Lê o conteúdo de um arquivo.
        
        Args:
            file_path: Caminho do arquivo.
            
        Returns:
            str: Conteúdo do arquivo.
        """
        full_path = os.path.join(self.project_dir, file_path)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Erro ao ler arquivo {file_path}: {e}")
            return ""

    def write_file(self, file_path: str, content: str) -> bool:
        """
        Escreve conteúdo em um arquivo no diretório de saída.
        
        Args:
            file_path: Caminho relativo do arquivo.
            content: Conteúdo a ser escrito.
            
        Returns:
            bool: True se o arquivo foi escrito com sucesso, False caso contrário.
        """
        full_path = os.path.join(self.output_dir, file_path)
        try:
            # Garantir que o diretório exista
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Arquivo gerado com sucesso: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao escrever arquivo {file_path}: {e}")
            return False

    def extract_json_from_text(self, text: str) -> Dict:
        """
        Extrai um objeto JSON de um texto.
        
        Args:
            text: Texto que contém JSON.
            
        Returns:
            Dict: Objeto JSON extraído, ou dicionário vazio em caso de falha.
        """
        try:
            # Procura por padrões de JSON
            json_pattern = r'```json\s*([\s\S]*?)\s*```|^\s*(\{[\s\S]*\})\s*$'
            match = re.search(json_pattern, text, re.MULTILINE)
            
            if match:
                # Usa o grupo que capturou o JSON
                json_str = match.group(1) if match.group(1) else match.group(2)
                return json.loads(json_str)
            
            # Se o regex não encontrou, tenta encontrar { } no texto
            start = text.find('{')
            end = text.rfind('}')
            
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start:end+1])
                except:
                    pass
                    
            # Falha na extração
            logger.warning("Não foi possível extrair JSON da resposta.")
            return {}
                
        except Exception as e:
            logger.error(f"Erro ao processar resposta JSON: {e}")
            return {}

    def analyze_code(self, file_content: str, file_path: str) -> Dict:
        """
        Analisa o código usando modelo local ou API.
        
        Args:
            file_content: Conteúdo do arquivo.
            file_path: Caminho do arquivo (para contexto).
            
        Returns:
            Dict: Resultado da análise com problemas e sugestões de melhoria.
        """
        prompt = f"""
        Analise o seguinte código de um arquivo chamado {file_path}:
        
        ```
        {file_content}
        ```
        
        Identifique:
        1. Problemas, bugs ou partes incompletas
        2. Possíveis melhorias de performance, legibilidade ou segurança
        3. Sugestões de implementação para partes que parecem faltar
        
        Responda no seguinte formato JSON:
        {{
            "problemas": [
                {{
                    "tipo": "bug|incompleto|performance|legibilidade|segurança",
                    "linha_aproximada": "número ou range (ex: 10 ou 10-15)",
                    "descrição": "descrição do problema",
                    "severidade": "baixa|média|alta"
                }}
            ],
            "código_corrigido": "código completo com todas as correções e melhorias aplicadas",
            "explicação": "explicação detalhada das mudanças feitas"
        }}
        
        Retorne APENAS o JSON, sem texto adicional.
        """

        if self.use_api:
            try:
                # Usar API do OpenAI para análise
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Você é um engenheiro de software sênior especializado em análise de código. Responda em formato JSON válido."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=4000
                )
                
                response_text = response.choices[0].message.content
                analysis = self.extract_json_from_text(response_text)
                
                if not analysis or "problemas" not in analysis:
                    logger.warning(f"Resposta inválida para {file_path}. Tentando formatação alternativa...")
                    return self.create_fallback_analysis(response_text, file_content)
                
                return analysis
                
            except Exception as e:
                logger.error(f"Erro na chamada da API: {e}")
                return {
                    "problemas": [],
                    "código_corrigido": file_content,
                    "explicação": f"Erro na API: {str(e)}"
                }
        else:
            try:
                # Usar modelo local para análise
                result = self.pipe(prompt, max_length=len(prompt) + 4096)
                
                # Extrair resposta
                response_text = result[0]['generated_text'][len(prompt):]
                
                # Processar e extrair o JSON da resposta
                analysis = self.extract_json_from_text(response_text)
                
                if not analysis or "problemas" not in analysis:
                    logger.warning(f"Resposta inválida para {file_path}. Tentando formatação alternativa...")
                    return self.create_fallback_analysis(response_text, file_content)
                    
                return analysis
                    
            except Exception as e:
                logger.error(f"Erro na geração de análise: {e}")
                return {
                    "problemas": [],
                    "código_corrigido": file_content,
                    "explicação": f"Erro na análise: {str(e)}"
                }
    
    def create_fallback_analysis(self, response_text: str, original_code: str) -> Dict:
        """
        Cria uma análise de fallback quando a resposta JSON falha.
        
        Args:
            response_text: Texto da resposta.
            original_code: Código original.
            
        Returns:
            Dict: Análise simplificada.
        """
        # Tentar extrair o código corrigido entre blocos de código
        code_pattern = r'```(?:python|javascript|html|css)?\s*([\s\S]*?)\s*```'
        code_matches = re.findall(code_pattern, response_text)
        
        if code_matches and len(code_matches[-1]) > len(original_code) * 0.5:
            # Use o último bloco de código como código corrigido
            corrected_code = code_matches[-1]
        else:
            corrected_code = original_code
        
        # Tentar extrair problemas do texto
        problemas = []
        problem_patterns = [
            r'(?:problema|bug|erro|issue).*?(?:linha|line).*?(\d+)[^\d]*(.*?)(?:\.|$)',
            r'(?:incompleto|incomplete).*?(?:linha|line).*?(\d+)[^\d]*(.*?)(?:\.|$)',
            r'(?:segurança|security).*?(?:linha|line).*?(\d+)[^\d]*(.*?)(?:\.|$)'
        ]
        
        for pattern in problem_patterns:
            for match in re.finditer(pattern, response_text, re.IGNORECASE):
                line = match.group(1)
                description = match.group(2).strip()
                if description:
                    problemas.append({
                        "tipo": "bug",
                        "linha_aproximada": line,
                        "descrição": description,
                        "severidade": "média"
                    })
        
        # Extrair explicação
        explanation_match = re.search(r'(?:explicação|explanation).*?:(.*?)(?:$|\n\n)', 
                                     response_text, re.IGNORECASE | re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else "Análise parcial devido a erro de formato."
        
        return {
            "problemas": problemas,
            "código_corrigido": corrected_code,
            "explicação": explanation
        }

    def analyze_and_fix_file(self, file_path: str) -> bool:
        """
        Analisa e corrige um arquivo específico.
        
        Args:
            file_path: Caminho relativo do arquivo.
            
        Returns:
            bool: True se o arquivo foi analisado e corrigido com sucesso.
        """
        logger.info(f"Analisando arquivo: {file_path}")
        
        # Ler conteúdo do arquivo
        file_content = self.read_file(file_path)
        if not file_content:
            return False
            
        # Adicionar ao conjunto de arquivos analisados
        self.analyzed_files.add(file_path)
        
        # Executar análise
        analysis = self.analyze_code(file_content, file_path)
        
        if not analysis or "problemas" not in analysis:
            logger.warning(f"Análise falhou para {file_path}")
            return False
            
        # Verificar se há problemas identificados
        if not analysis["problemas"]:
            logger.info(f"Nenhum problema identificado em {file_path}")
            return True
            
        # Registrar problemas encontrados
        for problema in analysis["problemas"]:
            logger.info(f"Problema encontrado em {file_path}, linha {problema.get('linha_aproximada', 'N/A')}: {problema.get('descrição', 'Sem descrição')} (Severidade: {problema.get('severidade', 'N/A')})")
        
        # Salvar versão corrigida
        if "código_corrigido" in analysis and analysis["código_corrigido"]:
            output_path = file_path
            result = self.write_file(output_path, analysis["código_corrigido"])
            if result:
                self.improvements_count += 1
                logger.info(f"Arquivo corrigido salvo: {output_path}")
                logger.info(f"Explicação das correções: {analysis.get('explicação', 'Sem explicação')}")
            return result
        
        return False

    def identify_missing_files(self, structure: Dict) -> List[Dict]:
        """
        Identifica arquivos que parecem estar faltando no projeto.
        
        Args:
            structure: Estrutura do projeto.
            
        Returns:
            List[Dict]: Lista de sugestões de arquivos a serem criados.
        """
        prompt = f"""
        Analise a estrutura deste projeto Python:
        
        Diretórios:
        {json.dumps(structure['dirs'], indent=2)}
        
        Arquivos:
        {json.dumps(structure['files'], indent=2)}
        
        Identifique arquivos importantes que parecem estar faltando. Por exemplo:
        - Arquivos de configuração
        - Arquivos de documentação
        - Módulos de teste
        - Arquivos essenciais para a funcionalidade do projeto
        
        Responda no seguinte formato JSON:
        {{
            "arquivos_faltantes": [
                {{
                    "nome_arquivo": "caminho/para/arquivo.py",
                    "propósito": "descrição do propósito do arquivo",
                    "conteúdo_sugerido": "código completo sugerido para o arquivo",
                    "prioridade": "baixa|média|alta"
                }}
            ]
        }}
        
        Retorne APENAS o JSON, sem texto adicional.
        """
        
        if self.use_api:
            try:
                # Usar API do OpenAI para análise
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Você é um arquiteto de software experiente. Responda em formato JSON válido."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=4000
                )
                
                response_text = response.choices[0].message.content
                analysis = self.extract_json_from_text(response_text)
                
                return analysis.get("arquivos_faltantes", [])
                
            except Exception as e:
                logger.error(f"Erro na chamada da API: {e}")
                return []
        else:
            try:
                # Usar modelo local para análise
                result = self.pipe(prompt, max_length=len(prompt) + 4096)
                
                # Extrair resposta
                response_text = result[0]['generated_text'][len(prompt):]
                
                # Processar e extrair o JSON da resposta
                analysis = self.extract_json_from_text(response_text)
                
                return analysis.get("arquivos_faltantes", [])
                    
            except Exception as e:
                logger.error(f"Erro na identificação de arquivos faltantes: {e}")
                return []

    def create_missing_files(self, missing_files: List[Dict]) -> int:
        """
        Cria arquivos identificados como faltantes.
        
        Args:
            missing_files: Lista de arquivos a serem criados.
            
        Returns:
            int: Número de arquivos criados.
        """
        count = 0
        for file_info in missing_files:
            file_path = file_info.get("nome_arquivo")
            if not file_path:
                continue
                
            content = file_info.get("conteúdo_sugerido", "")
            purpose = file_info.get("propósito", "Arquivo faltante detectado")
            priority = file_info.get("prioridade", "média")
            
            logger.info(f"Criando arquivo faltante: {file_path} (Prioridade: {priority})")
            logger.info(f"Propósito: {purpose}")
            
            if self.write_file(file_path, content):
                count += 1
                self.improvements_count += 1
        
        return count

    def run(self, max_iterations: int = 3) -> Dict:
        """
        Executa o processo de análise e correção.
        
        Args:
            max_iterations: Número máximo de iterações.
            
        Returns:
            Dict: Resumo dos resultados.
        """
        start_time = time.time()
        logger.info(f"Iniciando EzioFinisher com {max_iterations} iterações máximas")
        
        resultados = {
            "arquivos_analisados": 0,
            "problemas_corrigidos": 0,
            "arquivos_criados": 0,
            "erros": 0
        }
        
        for iteration in range(max_iterations):
            logger.info(f"Iniciando iteração {iteration + 1}/{max_iterations}")
            
            # Obter estrutura do projeto
            structure = self.get_project_structure()
            
            # Analisar e corrigir arquivos existentes
            for file_path in structure["files"]:
                if file_path in self.analyzed_files:
                    continue
                    
                success = self.analyze_and_fix_file(file_path)
                resultados["arquivos_analisados"] += 1
                if not success:
                    resultados["erros"] += 1
            
            # Identificar e criar arquivos faltantes
            missing_files = self.identify_missing_files(structure)
            created_count = self.create_missing_files(missing_files)
            resultados["arquivos_criados"] += created_count
            
            # Atualizar contagem de problemas corrigidos
            resultados["problemas_corrigidos"] = self.improvements_count
            
            logger.info(f"Iteração {iteration + 1} concluída. Resumo parcial: {resultados}")
            
            # Se nenhuma melhoria foi feita nesta iteração, podemos parar
            if created_count == 0 and resultados["problemas_corrigidos"] == self.improvements_count:
                logger.info("Nenhuma nova melhoria identificada, finalizando ciclo de iterações")
                break
        
        elapsed_time = time.time() - start_time
        logger.info(f"EzioFinisher concluído em {elapsed_time:.2f} segundos")
        logger.info(f"Resumo final: {resultados}")
        
        return resultados

def main():
    """Função principal para executar o EzioFinisher como script."""
    parser = argparse.ArgumentParser(description="EzioFinisher - Agente autônomo para correção de código (Versão Simplificada)")
    parser.add_argument("--use-api", action="store_true", help="Usar API do OpenAI em vez de modelo local")
    parser.add_argument("--api-key", help="Chave de API do OpenAI (se --use-api for especificado)")
    parser.add_argument("--max-iterations", type=int, default=3, help="Número máximo de iterações")
    args = parser.parse_args()
    
    try:
        # Inicializar e executar o EzioFinisher
        ezio = EzioFinisher(use_api=args.use_api, api_key=args.api_key)
        resultados = ezio.run(max_iterations=args.max_iterations)
        
        # Imprimir resumo final
        print("\n" + "="*50)
        print("RESUMO DO EZIOFINISHER")
        print("="*50)
        print(f"Modo: {'API OpenAI' if args.use_api else 'Modelo Local'}")
        print(f"Arquivos analisados: {resultados['arquivos_analisados']}")
        print(f"Problemas corrigidos: {resultados['problemas_corrigidos']}")
        print(f"Arquivos criados: {resultados['arquivos_criados']}")
        print(f"Erros: {resultados['erros']}")
        print("="*50)
        print(f"Log completo disponível em: {LOG_FILE}")
        print(f"Arquivos gerados salvos em: {OUTPUT_DIR}")
        print("="*50)
        
    except Exception as e:
        logger.critical(f"Erro crítico: {e}")
        print(f"Erro ao executar EzioFinisher: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())