import os
import json
import time
from pathlib import Path

# Diretórios
REPORTS_DIR = "./reports/ezio_finisher"

# Criar diretório se não existir
Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
print(f"Diretório criado/verificado: {REPORTS_DIR}")

# Montar o relatório de erro
error_details = {
    "report_type": "error",
    "priority": 0,
    "request_id": "fix001",
    "title": "Expert Initialization Error Fix",
    "details": {
        "error_message": "EzioBaseExpert.__init__() missing 1 required positional argument: 'expert_type'",
        "file_path": "core/quantum_moe_core/quantum_moe_core.py",
        "line_number": 638,
        "component": "QuantumMoECore",
        "expert_affected": "sentiment",
        "suggested_fix": "O método _load_expert_from_file deve verificar se a classe especialista herda de EzioBaseExpert e passar o parâmetro expert_type quando necessário",
        "code_snippet": {
            "current": "expert_instance = expert_class()",
            "suggested": "if issubclass(expert_class, EzioBaseExpert):\n    expert_instance = expert_class(expert_type=os.path.basename(expert_file).split('.')[0])\nelse:\n    expert_instance = expert_class()"
        }
    },
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
}

# Salvar o relatório
filename = f"fix_request_fix001_{int(time.time())}.json"
file_path = os.path.join(REPORTS_DIR, filename)

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(error_details, f, indent=2)

print(f"Solicitação de correção criada: {file_path}")
print(f"Aguarde o ciclo de processamento do ClaudeSyncBridge (60 segundos)")