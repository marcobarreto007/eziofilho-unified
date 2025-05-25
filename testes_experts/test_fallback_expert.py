# Caminho: C:\Users\anapa\SuperIA\EzioFilhoUnified\testes_experts\test_fallback_expert.py

import importlib.util
import os

# Caminho completo para o módulo a ser carregado
module_path = r"C:\Users\anapa\SuperIA\EzioFilhoUnified\experts\fallback_data\fallback_data_expert.py"

# Nome arbitrário do módulo temporário
module_name = "fallback_data_expert"

# Carregamento dinâmico do módulo
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Usa a classe carregada
FallbackDataExpert = module.FallbackDataExpert

def main():
    expert = FallbackDataExpert()

    result = expert.get_data(
        data_type="financial_data",
        query_params={"ticker": "AAPL"},
        force_fallback=True
    )

    print("=== RESULTADO DO TESTE ===")
    print(f"Fonte usada: {result.get('source')}")
    print(f"É fallback? {result.get('is_fallback')}")
    print("Dados retornados:")
    print(result.get('data'))

if __name__ == "__main__":
    main()
