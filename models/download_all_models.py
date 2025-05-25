# download_all_models.py - Download all recommended models
# Audit Mode: Active - Batch model download
# Path: C:\Users\anapa\EzioFilhoUnified\ezio_experts\models
# User: marcobarreto007
# Date: 2025-05-24 16:59:51 UTC

print("📥 DOWNLOADING RECOMMENDED MODELS FOR EZIOFILHO")
print("=" * 60)

models_to_download = [
    {
        "name": "PHI-3 Mini 4K",
        "command": "pip install transformers torch && python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct'); AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct')\""
    },
    {
        "name": "Sentence Transformers (Multilingual)",
        "command": "pip install sentence-transformers && python -c \"from sentence_transformers import SentenceTransformer; model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2'); model.save('models/multilingual-embeddings')\""
    },
    {
        "name": "FinBERT (Financial)",
        "command": "python -c \"from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoTokenizer.from_pretrained('ProsusAI/finbert'); AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')\""
    }
]

import os
import subprocess

for model in models_to_download:
    print(f"\n🔄 Downloading: {model['name']}")
    try:
        subprocess.run(model['command'], shell=True, check=True)
        print(f"✅ {model['name']} downloaded!")
    except:
        print(f"❌ Failed to download {model['name']}")

print("\n✅ Download process completed!")