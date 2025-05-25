#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sistema de Chat Direto com Transformers
---------------------------------------
Este script carrega modelos Transformers diretamente e os usa com AutoGen
sem depender de nenhuma classe personalizada.
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s"
)
logger = logging.getLogger("direct_chat")

# Diretório atual
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(config_path):
    """Carrega a configuração de modelos."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Verifica estrutura
        if "models" in config:
            model_list = config["models"]
            logger.info(f"Configuração carregada: {len(model_list)} modelos")
        elif isinstance(config, list):
            model_list = config
            logger.info(f"Configuração (formato antigo): {len(model_list)} modelos")
        else:
            logger.warning("Formato desconhecido, usando vazio")
            model_list = []
        
        return model_list
    except Exception as e:
        logger.error(f"Erro ao carregar configuração: {e}")
        return []


def find_model(model_list, search_term=None):
    """Encontra um modelo adequado na lista."""
    if not model_list:
        return None
    
    # Se não há termo de busca, retorna o primeiro modelo Transformers
    if not search_term:
        for model in model_list:
            if model.get("type", "").lower() == "transformers":
                return model
        return model_list[0]  # Fallback para o primeiro
    
    # Procura pelo termo
    search_term = search_term.lower()
    for model in model_list:
        name = model.get("name", "").lower()
        if search_term in name:
            return model
    
    # Se não encontrou, usa o primeiro modelo Transformers
    logger.warning(f"Modelo '{search_term}' não encontrado, usando alternativa")
    for model in model_list:
        if model.get("type", "").lower() == "transformers":
            return model
    
    return model_list[0]  # Último recurso


def load_transformers_model(model_path):
    """Carrega modelo e tokenizer do Transformers."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Verifica se o modelo existe
        if not os.path.exists(model_path):
            logger.error(f"Caminho do modelo não existe: {model_path}")
            return None, None
        
        # Determina dispositivo (GPU/CPU)
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("GPU não disponível, usando CPU")
        
        # Suprime avisos não críticos
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Carrega tokenizer
            logger.info(f"Carregando tokenizer de: {os.path.basename(model_path)}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Define tokens especiais se necessário
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Carrega modelo
            logger.info(f"Carregando modelo de: {os.path.basename(model_path)}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        logger.info(f"✓ Modelo carregado com sucesso")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        return None, None


def format_prompt(prompt, model_path):
    """Formata o prompt de acordo com o modelo."""
    model_name = os.path.basename(model_path).lower()
    
    # Detecta o tipo de modelo e aplica o formato apropriado
    if "phi-3" in model_name:
        return f"<|user|>\n{prompt}\n<|assistant|>\n"
    elif "phi-2" in model_name:
        return f"Instruct: {prompt}\nOutput: "
    elif "qwen" in model_name:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    elif any(x in model_name for x in ["llama2", "llama-2", "mistral"]):
        return f"<s>[INST] {prompt} [/INST] "
    elif "falcon" in model_name:
        return f"User: {prompt}\nAssistant: "
    else:
        return f"Pergunta: {prompt}\nResposta: "


def generate_text(model, tokenizer, prompt, model_path, max_tokens=2048):
    """Gera texto com o modelo."""
    import torch
    
    # Formata o prompt adequadamente
    formatted_prompt = format_prompt(prompt, model_path)
    
    # Tokeniza o prompt
    input_ids = tokenizer.encode(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096-max_tokens
    )
    
    # Move para o mesmo dispositivo do modelo
    input_ids = input_ids.to(model.device)
    
    # Parâmetros de geração
    gen_params = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_new_tokens": max_tokens,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # Gera a resposta
    with torch.no_grad():
        output = model.generate(
            input_ids,
            **gen_params
        )
    
    # Decodifica a saída
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove o prompt da saída
    if decoded_output.startswith(formatted_prompt):
        response = decoded_output[len(formatted_prompt):]
    else:
        response = decoded_output
    
    # Remove tokens especiais
    for token in ["<|endoftext|>", "</s>", "<|im_end|>"]:
        if token in response:
            response = response.split(token)[0]
    
    return response.strip()


def create_autogen_completion(model, tokenizer, model_path):
    """Cria função de completion para AutoGen."""
    def completion_function(prompt, **kwargs):
        """Função de completion para AutoGen."""
        try:
            logger.info(f"Gerando resposta para: '{prompt[:30]}...'")
            start_time = time.time()
            
            # Gera o texto
            response = generate_text(model, tokenizer, prompt, model_path)
            
            elapsed = time.time() - start_time
            logger.info(f"Resposta gerada em {elapsed:.2f}s")
            
            return {
                "content": response,
                "model": os.path.basename(model_path)
            }
        except Exception as e:
            logger.error(f"Erro na geração: {e}")
            return {
                "content": f"Erro: {str(e)}",
                "model": "error"
            }
    
    return completion_function


def setup_autogen(completion_function):
    """Configura o sistema AutoGen."""
    try:
        import autogen
        logger.info("AutoGen importado com sucesso")
        
        # Registra a função de completion
        autogen.register_llm_provider(
            model_type="direct-transformers",
            completion_func=completion_function
        )
        
        # Configuração do LLM
        llm_config = {
            "config_list": [{"model": "direct-transformers", "api_key": "not-needed"}],
            "temperature": 0.7,
            "timeout": 300,
        }
        
        # Cria o agente assistente
        assistant = autogen.AssistantAgent(
            name="Assistente",
            system_message="""Você é um assistente IA útil e versátil.
Responda em português do Brasil de forma clara, precisa e concisa.
Quando a resposta for extensa, use formatação em Markdown para melhorar a legibilidade.""",
            llm_config=llm_config
        )
        
        # Cria o agente usuário
        user = autogen.UserProxyAgent(
            name="Usuario",
            human_input_mode="NEVER",
            is_termination_msg=lambda x: False,
            code_execution_config=False,
        )
        
        return assistant, user
    
    except Exception as e:
        logger.error(f"Erro ao configurar AutoGen: {e}")
        return None, None


def main():
    """Função principal."""
    # Configura argumentos de linha de comando
    parser = argparse.ArgumentParser(description="Chat direto com modelos Transformers")
    parser.add_argument("--config", default="models_config.json", help="Arquivo de configuração")
    parser.add_argument("--model", help="Nome do modelo (parcial)")
    parser.add_argument("--prompt", help="Prompt inicial")
    args = parser.parse_args()
    
    # Configura Python path
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    
    # Carrega configuração
    model_list = load_config(args.config)
    if not model_list:
        logger.error("Nenhum modelo encontrado na configuração")
        return 1
    
    # Encontra modelo adequado
    model_config = find_model(model_list, args.model)
    if not model_config:
        logger.error("Nenhum modelo utilizável encontrado")
        return 1
    
    model_path = model_config.get("path")
    model_name = model_config.get("name")
    
    logger.info(f"Usando modelo: {model_name} ({model_path})")
    
    # Carrega modelo e tokenizer
    model, tokenizer = load_transformers_model(model_path)
    if not model or not tokenizer:
        logger.error("Falha ao carregar modelo/tokenizer")
        return 1
    
    # Cria função de completion
    completion_function = create_autogen_completion(model, tokenizer, model_path)
    
    # Configura AutoGen
    assistant, user = setup_autogen(completion_function)
    if not assistant or not user:
        logger.error("Falha ao configurar AutoGen")
        return 1
    
    # Obtém prompt se não especificado
    prompt = args.prompt
    if not prompt:
        print("\n===== Chat Direto com Modelo Transformers =====")
        print(f"Modelo: {model_name}")
        print("==============================================\n")
        prompt = input("Digite seu prompt: ")
    
    # Executa conversa
    logger.info(f"Iniciando conversa com prompt: '{prompt[:50]}...'")
    try:
        user.initiate_chat(
            assistant,
            message=prompt
        )
        logger.info("Conversa concluída com sucesso")
    except Exception as e:
        logger.error(f"Erro durante conversa: {e}")
        print(f"\nErro: {str(e)}")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Operação interrompida pelo usuário")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erro não tratado: {e}")
        sys.exit(1)