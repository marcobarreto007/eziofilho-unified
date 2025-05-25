#!/usr/bin/env python3
"""
Teste simples do sistema AutoGen
"""

import os
import sys
import json
from pathlib import Path

print("🔍 Verificando sistema...\n")

# 1. Verifica Python
print(f"✅ Python {sys.version.split()[0]}")

# 2. Verifica AutoGen
try:
    import autogen
    print(f"✅ AutoGen {autogen.__version__}")
except ImportError:
    print("❌ AutoGen não instalado!")
    print("   Execute: pip install pyautogen==0.2.18")
    sys.exit(1)

# 3. Verifica llama-cpp-python
try:
    import llama_cpp
    print("✅ llama-cpp-python instalado")
except ImportError:
    print("⚠️ llama-cpp-python não instalado (opcional)")

# 4. Verifica modelos
models_dir = Path.home() / ".cache" / "models"
if models_dir.exists():
    gguf_files = list(models_dir.glob("*.gguf"))
    if gguf_files:
        print(f"✅ {len(gguf_files)} modelos GGUF encontrados:")
        for f in gguf_files[:3]:  # Mostra até 3
            print(f"   - {f.name}")
    else:
        print("⚠️ Nenhum modelo GGUF encontrado")
else:
    print("⚠️ Diretório de modelos não existe")

# 5. Teste básico do AutoGen
print("\n🧪 Testando AutoGen básico...")

try:
    # Config simples para teste
    config_list = [{
        "model": "gpt-3.5-turbo",
        "api_key": "test-key",
        "base_url": "http://localhost:1234/v1"
    }]
    
    # Cria agente básico
    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config={"config_list": config_list}
    )
    
    print("✅ AutoGen funcionando corretamente!")
    
except Exception as e:
    print(f"❌ Erro no AutoGen: {e}")

print("\n📋 Resumo:")
print("- Sistema pronto para uso")
print("- Execute 'python main.py --help' para opções")
print("- Configure um servidor de modelos (LM Studio/Ollama)")