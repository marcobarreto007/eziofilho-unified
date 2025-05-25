from ezio_experts.fact_checker.duckduckgo_fact_checker import DuckDuckGoFactChecker
checker = DuckDuckGoFactChecker()

answer = llm(prompt)  # resposta do modelo
conf, link = checker.verify(prompt, answer)

if conf < 0.4:
    print("\n⚠️  Baixa confiança (%.2f). Verifique manualmente." % conf)
else:
    print("\n✅ Verificado (%.2f) | Evidência: %s" % (conf, link))
