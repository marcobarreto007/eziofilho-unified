# Caminho do arquivo: C:\Users\anapa\SuperIA\EzioFilhoUnified\tests_modelos\run_phi3_robusto.py

import os
import sys
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# CONFIGS GERAIS
MODEL_PATH = r"C:\Users\anapa\modelos_hf\microsoft--phi-3-mini-4k-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "Explique como utilizar modelos de linguagem locais para automação de tarefas de IA."
MAX_TOKENS = 256

# SETUP DE LOG
LOG_PATH = os.path.join(os.path.dirname(__file__), "log_phi3.txt")
logging.basicConfig(filename=LOG_PATH,
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def log_and_print(msg, level="info"):
    print(msg)
    if level == "info":
        logging.info(msg)
    elif level == "error":
        logging.error(msg)
    elif level == "warning":
        logging.warning(msg)

def load_model_and_tokenizer():
    log_and_print(f"[INFO] Carregando modelo do caminho: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
        model.to(DEVICE)
        log_and_print(f"[OK] Modelo carregado e movido para {DEVICE.upper()}")
        return tokenizer, model
    except Exception as e:
        log_and_print(f"[ERRO] Falha ao carregar modelo/tokenizer: {e}", level="error")
        sys.exit(1)

def infer(prompt, tokenizer, model):
    log_and_print(f"[INFO] Realizando inferência...")
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=True,
                top_p=0.95,
                top_k=40,
                temperature=0.8
            )
        resposta = tokenizer.decode(output[0], skip_special_tokens=True)
        log_and_print("[OK] Inferência realizada com sucesso!")
        return resposta
    except Exception as e:
        log_and_print(f"[ERRO] Falha durante a inferência: {e}", level="error")
        return "[ERRO DE INFERÊNCIA]"

def main():
    log_and_print("==== Teste ROBUSTO — Phi-3 Mini Local ====")
    log_and_print(f"Dispositivo detectado: {DEVICE.upper()}")
    tokenizer, model = load_model_and_tokenizer()
    resposta = infer(PROMPT, tokenizer, model)
    print("\n========== RESPOSTA ==========")
    print(resposta)
    print("=========== FIM ==============")
    log_and_print("Teste finalizado!\n")

if __name__ == "__main__":
    main()
