# Caminho: C:\Users\anapa\SuperIA\EzioFilhoUnified\testes_experts\test_cache_expert.py

import importlib.util
import os
import time

# Caminho do especialista
module_path = r"C:\Users\anapa\SuperIA\EzioFilhoUnified\experts\cache_expert\cache_expert.py"
module_name = "cache_expert"

spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

CacheExpert = module.CacheExpert

def main():
    cache = CacheExpert()

    print("\n[TESTE] Salvando item no cache...")
    cache.set_to_cache("btc_price", {"value": 65432}, ttl=3)

    print("[TESTE] Recuperando item...")
    result = cache.get_from_cache("btc_price")
    print("→ Resultado:", result)

    print("[TESTE] Aguardando expiração do cache (3s)...")
    time.sleep(4)

    print("[TESTE] Verificando expiração...")
    result = cache.get_from_cache("btc_price")
    print("→ Resultado pós-expiração:", result)

    print("[TESTE] Setando outro item...")
    cache.set_to_cache("foo", "bar")

    print("[TESTE] Limpando cache...")
    cache.clear_all_cache()

    print("[TESTE] Checando item após limpeza...")
    result = cache.get_from_cache("foo")
    print("→ Resultado após limpeza:", result)

if __name__ == "__main__":
    main()
