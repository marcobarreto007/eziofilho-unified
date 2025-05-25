import subprocess
import os
import sys
import time

def run_script(path):
    print(f"\n===> Executando: {path}")
    result = subprocess.run([sys.executable, path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Erros:")
        print(result.stderr)

if __name__ == "__main__":
    base_path = os.path.abspath(os.path.dirname(__file__))
    print("🧪 Iniciando execução de testes MoE...")

    scripts = [
        "test_quantum_moe_core.py",
        # Você pode adicionar outros testes aqui
        # "test_model_router.py",
        # "test_rlhf_trainer.py",
    ]

    for script in scripts:
        full_path = os.path.join(base_path, script)
        if os.path.exists(full_path):
            run_script(full_path)
        else:
            print(f"⚠️ Script não encontrado: {script}")

    print("\n✅ Todos os testes finalizados.")
