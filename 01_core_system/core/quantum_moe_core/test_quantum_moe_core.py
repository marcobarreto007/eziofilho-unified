from quantum_moe_core import QuantumMoECore

query = "What is the current market sentiment for technology stocks?"
context = {
    "query_type": "sentiment",
    "capabilities": ["sentiment_analysis", "market_data"],
    "sector": "technology"
}

moe_core = QuantumMoECore(
    max_workers=4,
    use_redis_cache=False,
    cache_ttl=300
)

result, experts_used = moe_core.process_query(query, context)

print("=== RESULTADO FINAL ===")
print("Experts usados:", experts_used)
print("Resultado:")
print(result)

print("\n=== ESTATÍSTICAS ===")
stats = moe_core.get_expert_stats()
for name, data in stats.items():
    print(f"- {name}: chamadas = {data['performance_metrics']['calls']}, "
          f"sucesso = {data['performance_metrics']['successful_calls']}, "
          f"confiança média = {data['performance_metrics']['average_confidence']:.2f}")

moe_core.shutdown()
