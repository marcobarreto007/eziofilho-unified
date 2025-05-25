#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EzioFinisher - Versão Melhorada
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
    
    def __init__(self, max_file_size=5000, file_limit=50):
        """
        Inicializa o agente EzioFinisher.
        
        Args:
            max_file_size: Tamanho máximo de arquivo em caracteres
            file_limit: Limite de arquivos a processar
        """
        self.max_file_size = max_file_size
        self.file_limit = file_limit
        
        # Configurar o modelo local
        try:
            from transformers import pipeline
            
            logger.info("Inicializando modelo local. Isso pode levar alguns minutos...")
            self.pipe = pipeline(
                "text-generation",
                model="Salesforce/codegen-350M-mono",  # Modelo pequeno para código
                max_length=1024,  # Reduzido para evitar erros de memória
                truncation=True   # Ativando truncamento explicitamente
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
        
        # Ignorar diretórios específicos
        self.ignore_dirs = {"venv", ".git", "__pycache__", "node_modules", "autogen_generated"}
        
        # Inicializa contador de melhorias
        self.improvements_count = 0
        
        logger.info("EzioFinisher inicializado com sucesso!")
        logger.info(f"Diretório do projeto: {self.project_dir}")
        logger.info(f"Diretório de saída: {self.output_dir}")
        logger.info(f"Limite de tamanho de arquivo: {self.max_file_size} caracteres")
        logger.info(f"Limite de arquivos a processar: {self.file_limit}")

    def get_project_structure(self) -> Dict:
        """
        Mapeia a estrutura do projeto.
        
        Returns:
            Dict: Estrutura de diretórios e arquivos do projeto.
        """
        structure = {"dirs": [], "files": []}
        
        for root, dirs, files in os.walk(self.project_dir):
            # Aplicar filtro de diretórios ignorados
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            
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

    def is_file_too_large(self, content: str) -> bool:
        """
        Verifica se um arquivo é grande demais para processamento.
        
        Args:
            content: Conteúdo do arquivo.
            
        Returns:
            bool: True se o arquivo for muito grande.
        """
        return len(content) > self.max_file_size

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

    def analyze_code(self, file_content: str, file_path: str) -> Dict:
        """
        Analisa o código usando modelo local.
        
        Args:
            file_content: Conteúdo do arquivo.
            file_path: Caminho do arquivo (para contexto).
            
        Returns:
            Dict: Resultado da análise com problemas e sugestões de melhoria.
        """
        # Verificar se o arquivo é muito grande
        if self.is_file_too_large(file_content):
            logger.warning(f"Arquivo {file_path} é muito grande ({len(file_content)} caracteres). Analisando apenas os primeiros {self.max_file_size} caracteres.")
            file_content = file_content[:self.max_file_size]
        
        prompt = f"""
        Analise o seguinte código de um arquivo chamado {file_path}:
        
        ```
        {file_content}
        ```
        
        Identifique:
        1. Problemas, bugs ou partes incompletas
        2. Possíveis melhorias de performance, legibilidade ou segurança
        3. Sugestões de implementação para partes que parecem faltar
        
        Responda no seguinte formato:
        PROBLEMAS:
        - [TIPO: bug/incompleto/performance] [LINHA: número] [SEVERIDADE: baixa/média/alta] Descrição do problema
        
        CÓDIGO CORRIGIDO:
        ```
        Insira aqui o código corrigido
        ```
        
        EXPLICAÇÃO:
        Explicação das mudanças feitas
        """

        try:
            # Usar modelo local para análise com tratamento de erros aprimorado
            result = self.pipe(prompt, max_length=1024, truncation=True, do_sample=False)
            
            # Extrair resposta
            response_text = result[0]['generated_text'][len(prompt):]
            
            # Processar a resposta em formato mais simples
            return self.parse_simple_format(response_text, file_content)
                
        except Exception as e:
            logger.error(f"Erro na geração de análise: {e}")
            return {
                "problemas": [],
                "código_corrigido": file_content,
                "explicação": f"Erro na análise: {str(e)}"
            }
    
    def parse_simple_format(self, response_text: str, original_code: str) -> Dict:
        """
        Analisa o formato de resposta mais simples.
        
        Args:
            response_text: Texto da resposta.
            original_code: Código original.
            
        Returns:
            Dict: Análise processada.
        """
        # Extractar problemas
        problemas = []
        problemas_pattern = r'- \[TIPO: (.*?)\] \[LINHA: (.*?)\] \[SEVERIDADE: (.*?)\] (.*?)(?:\n|$)'
        
        for match in re.finditer(problemas_pattern, response_text, re.MULTILINE):
            tipo = match.group(1).strip()
            linha = match.group(2).strip()
            severidade = match.group(3).strip()
            descricao = match.group(4).strip()
            
            problemas.append({
                "tipo": tipo,
                "linha_aproximada": linha,
                "descrição": descricao,
                "severidade": severidade
            })
        
        # Extrair código corrigido
        codigo_corrigido = original_code
        code_match = re.search(r'```(?:python|javascript|html|css)?\s*([\s\S]*?)\s*```', response_text)
        if code_match:
            codigo_corrigido = code_match.group(1)
        
        # Extrair explicação
        explicacao = ""
        explicacao_match = re.search(r'EXPLICAÇÃO:\s*([\s\S]*?)(?:\n\n|$)', response_text)
        if explicacao_match:
            explicacao = explicacao_match.group(1).strip()
        
        return {
            "problemas": problemas,
            "código_corrigido": codigo_corrigido,
            "explicação": explicacao
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
        
        # Verificar se o arquivo é muito grande
        if self.is_file_too_large(file_content):
            logger.warning(f"Arquivo {file_path} é muito grande ({len(file_content)} caracteres). Usando apenas os primeiros {self.max_file_size} caracteres para análise.")
        
        # Executar análise
        analysis = self.analyze_code(file_content, file_path)
        
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

    def create_missing_files(self) -> int:
        """
        Cria arquivos básicos que parecem estar faltando.
        
        Returns:
            int: Número de arquivos criados.
        """
        # Lista de arquivos essenciais para verificar
        essential_files = [
            {
                "nome_arquivo": "README.md",
                "verificar_existencia": os.path.exists(os.path.join(self.project_dir, "README.md")),
                "conteúdo": """# EzioFilhoUnified

Projeto unificado para automatização e otimização de código.

## Recursos

- Detecção automática de problemas
- Correção de código
- Geração de scripts

## Uso

...
"""
            },
            {
                "nome_arquivo": "requirements.txt",
                "verificar_existencia": os.path.exists(os.path.join(self.project_dir, "requirements.txt")),
                "conteúdo": """# Dependências principais
transformers>=4.30.0
torch>=2.0.0
"""
            },
            {
                "nome_arquivo": ".gitignore",
                "verificar_existencia": os.path.exists(os.path.join(self.project_dir, ".gitignore")),
                "conteúdo": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Logs
*.log
"""
            }
        ]
        
        count = 0
        for file_info in essential_files:
            if not file_info["verificar_existencia"]:
                logger.info(f"Criando arquivo essencial faltante: {file_info['nome_arquivo']}")
                
                if self.write_file(file_info["nome_arquivo"], file_info["conteúdo"]):
                    count += 1
                    self.improvements_count += 1
        
        return count

    def run(self, max_iterations: int = 1) -> Dict:
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
            
            # Limitar o número de arquivos
            limited_files = structure["files"][:self.file_limit]
            logger.info(f"Limitando análise a {len(limited_files)} de {len(structure['files'])} arquivos")
            
            # Analisar e corrigir arquivos existentes
            for file_path in limited_files:
                if file_path in self.analyzed_files:
                    continue
                    
                success = self.analyze_and_fix_file(file_path)
                resultados["arquivos_analisados"] += 1
                if not success:
                    resultados["erros"] += 1
            
            # Criar arquivos básicos faltantes
            created_count = self.create_missing_files()
            resultados["arquivos_criados"] += created_count
            
            # Atualizar contagem de problemas corrigidos
            resultados["problemas_corrigidos"] = self.improvements_count
            
            logger.info(f"Iteração {iteration + 1} concluída. Resumo parcial: {resultados}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"EzioFinisher concluído em {elapsed_time:.2f} segundos")
        logger.info(f"Resumo final: {resultados}")
        
        return resultados

def main():
    """Função principal para executar o EzioFinisher como script."""
    parser = argparse.ArgumentParser(description="EzioFinisher - Agente autônomo para correção de código (Versão Melhorada)")
    parser.add_argument("--max-iterations", type=int, default=1, help="Número máximo de iterações")
    parser.add_argument("--max-file-size", type=int, default=5000, help="Tamanho máximo de arquivo em caracteres")
    parser.add_argument("--file-limit", type=int, default=50, help="Limite de arquivos a processar")
    args = parser.parse_args()
    
    try:
        # Inicializar e executar o EzioFinisher
        ezio = EzioFinisher(max_file_size=args.max_file_size, file_limit=args.file_limit)
        resultados = ezio.run(max_iterations=args.max_iterations)
        
        # Imprimir resumo final
        print("\n" + "="*50)
        print("RESUMO DO EZIOFINISHER")
        print("="*50)
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