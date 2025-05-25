# C:\Users\anapa\SuperIA\EzioFilhoUnified\autogen_test.py

"""
Exemplo de integração do AutoGen com modelo HuggingFace local via Transformers.
"""

from autogen import AssistantAgent

MODEL_PATH = "C:/Users/anapa/SuperIA/EzioFilhoUnified/modelos_hf/gpt2"  # ajuste para outro se quiser

agent = AssistantAgent(
    name="LocalAgent",
    llm_config={
        "config_list": [
            {
                "provider": "transformers",
                "model": MODEL_PATH,
                "device": "cuda"  # troque para "cpu" se não tiver GPU ou não quiser usar CUDA
            }
        ]
    }
)

res = agent.chat("Explique o que é machine learning em termos simples.")
print("Resposta do modelo local:", res)
