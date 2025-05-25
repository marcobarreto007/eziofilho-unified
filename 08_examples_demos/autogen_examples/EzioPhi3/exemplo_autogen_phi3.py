"""
exemplo_autogen_phi3.py  –  GTX-1070 • bitsandbytes 4-bit
Coloque este arquivo em:
C:/Users/anapa/SuperIA/EzioFilhoUnified/autogen_examples/EzioPhi3/
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

MODEL_PATH = (
    "C:/Users/anapa/EzioFilhoUnified/modelos_hf/microsoft--phi-3-mini-4k-instruct"
)

# ---------------- 4-bit quantização -------------------------
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,   # cálculos em bfloat16
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("🔄  Carregando tokenizer …")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

print("🔄  Carregando modelo 4-bit … (≈40 s na 1070)")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_cfg,
    device_map="auto",             # já coloca camadas na GPU/CPU
    local_files_only=True,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.config.use_cache = False     # evita buffers extra

# ---- LIMPA resíduos de VRAM de execuções anteriores ----------
torch.cuda.empty_cache()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
    # IMPORTANTE:  **NÃO** usar device_map aqui.
    device=0,                      # 0 = primeira GPU
    do_sample=True,
    temperature=0.7,
    max_new_tokens=96,             # ↓ se ainda faltar VRAM
)

prompt = "Explique o que é um agente orquestrador em IA."
print("\n📝  Resposta:\n")
print(pipe(prompt)[0]["generated_text"])
