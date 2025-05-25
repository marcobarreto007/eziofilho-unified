# Caminho: C:\Users\anapa\SuperIA\EzioFilhoUnified\cli_phi3_gpu\cli_phi3_gpu.py

"""
CLI paralelizável para Phi-3 Mini: tenta usar ambas as GPUs se possível (device_map='auto').
Se não funcionar, rode B) abaixo em 2 prompts distintos, um para cada GPU.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_DIR = r"C:\Users\anapa\modelos_hf\microsoft--phi-3-mini-4k-instruct"

def get_device_map():
    # Se houver 2 GPUs, tenta usar ambas
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print(f"[INFO] {torch.cuda.device_count()} GPUs detectadas! Tentar paralelizar modelo...")
        return "auto"  # Pode ser "balanced" ou "auto"
    elif torch.cuda.is_available():
        print(f"[INFO] Apenas 1 GPU disponível: {torch.cuda.get_device_name(0)}")
        return {"": 0}
    else:
        print("[INFO] CUDA NÃO disponível, usando CPU")
        return "cpu"

def main():
    print("==== CLI Phi-3 Mini (Paralelo GPU) ====")
    device_map = get_device_map()
    print("[INFO] Carregando modelo e tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    print("[OK] Modelo carregado.")

    streamer = TextStreamer(tokenizer)

    print("\nDigite sua pergunta para o Phi-3 (ou 'exit' para sair):")
    while True:
        user_input = input(">>> ").strip()
        if user_input.lower() in ["exit", "sair", "quit"]:
            print("Saindo...")
            break

        prompt = f"{user_input}\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        # Não precisa mover para device manualmente, o transformers faz
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                streamer=streamer,
            )
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
