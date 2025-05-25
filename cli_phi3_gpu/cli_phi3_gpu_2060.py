# Caminho: C:\Users\anapa\SuperIA\EzioFilhoUnified\cli_phi3_gpu\cli_phi3_gpu_2060.py

"""
CLI Phi-3 Mini — Força uso da GPU 0 (RTX 2060)
Para sair, digite: exit
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # RTX 2060

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_DIR = r"C:\Users\anapa\modelos_hf\microsoft--phi-3-mini-4k-instruct"

def main():
    print("==== CLI Phi-3 Mini (GPU 0 — RTX 2060) ====")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando: {torch.cuda.get_device_name(0)}")

    print("[INFO] Carregando modelo e tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)
    print("[OK] Modelo carregado.")

    streamer = TextStreamer(tokenizer)

    print("\nDigite sua pergunta para o Phi-3 (ou 'exit' para sair):")
    while True:
        user_input = input(">>> ").strip()
        if user_input.lower() in ["exit", "sair", "quit"]:
            print("Saindo...")
            break

        prompt = f"{user_input}\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
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
