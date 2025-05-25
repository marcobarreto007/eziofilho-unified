"""
Script de teste para o especialista em sentimento financeiro
Demonstra uso avançado do especialista com visualização de resultados
"""
import os
import time
import logging
import argparse
import json
from pathlib import Path
from termcolor import colored
from datetime import datetime

# Importar o especialista
from ezio_experts.sentiment_expert import SentimentExpert

# Configurar logging formatado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Exemplos de textos para testar diferentes sentimentos
TEST_TEXTS = {
    "positivo": """
    A Apple superou todas as expectativas dos analistas no primeiro trimestre de 2025, 
    com receita recorde de $98,7 bilhões, um aumento de 15% em relação ao ano anterior. 
    O lucro por ação subiu 18% para $1.68, significativamente acima das previsões de $1.52. 
    A empresa também anunciou um aumento de dividendos de 10% e um programa de recompra 
    de ações de $120 bilhões, o maior de sua história. As vendas do iPhone cresceram impressionantes 
    22%, enquanto a receita de serviços atingiu máxima histórica com crescimento de 29%.
    O CEO destacou o forte crescimento em todos os segmentos geográficos, com destaque para a 
    expansão de 35% no mercado asiático.
    """,
    
    "negativo": """
    A Tesla anunciou resultados decepcionantes para o primeiro trimestre de 2025, 
    com uma queda de 12% na receita para $17,8 bilhões, muito abaixo das expectativas 
    de $22,3 bilhões. O lucro por ação despencou 45% para $0.35, versus estimativa de $0.81. 
    A empresa citou intensa competição no mercado de EVs e quedas significativas de preço 
    como principais fatores. As margens brutas de automóveis caíram para preocupantes 14.2%, 
    o menor nível em cinco anos. A empresa também anunciou atraso de 9 meses no lançamento 
    de novos modelos e reduziu sua previsão de entregas para o ano, causando preocupação 
    entre investidores sobre sua capacidade de manter liderança no mercado.
    """,
    
    "neutro": """
    O Federal Reserve manteve as taxas de juros inalteradas na faixa de 4,00% a 4,25% em sua 
    reunião de hoje, em linha com as expectativas de mercado. Em seu comunicado, o Fed observou 
    que a inflação "mostrou sinais de moderação, mas permanece elevada" e que a atividade econômica 
    "continua a expandir em ritmo moderado". O comitê reiterou que está "altamente atento aos riscos 
    de inflação" e que está "preparado para ajustar a postura da política monetária conforme apropriado 
    se surgirem riscos". Analistas interpretaram o comunicado como neutro, sem sinais claros sobre o 
    futuro caminho das taxas. O próximo movimento do Fed provavelmente dependerá dos próximos dados de 
    inflação e mercado de trabalho que serão divulgados no próximo mês.
    """,
    
    "misto": """
    A Amazon apresentou resultados mistos no segundo trimestre. A receita cresceu 9% para $121,2 bilhões, 
    ligeiramente acima das estimativas, impulsionada pelo forte desempenho na AWS, que registrou crescimento 
    de 33%. No entanto, o lucro operacional caiu 57% para $3,3 bilhões, impactado pelo aumento dos custos 
    de mão-de-obra e combustível. A empresa reportou uma perda líquida de $2 bilhões devido a um prejuízo 
    de investimento na Rivian Automotive. Para o próximo trimestre, a Amazon prevê receita entre $125-130 bilhões, 
    acima das expectativas, mas alertou que as pressões de custo persistirão. O CEO destacou planos para 
    otimizar a estrutura de custos enquanto continua investindo em áreas de crescimento estratégico.
    """
}

def print_header(text, color="blue"):
    """Imprime cabeçalho formatado"""
    width = min(100, os.get_terminal_size().columns - 2)
    print("\n" + colored("=" * width, color))
    print(colored(f" {text.center(width-2)} ", color, attrs=["bold"]))
    print(colored("=" * width, color) + "\n")

def print_section(title, color="cyan"):
    """Imprime título de seção formatado"""
    width = min(100, os.get_terminal_size().columns - 2)
    print("\n" + colored("-" * width, color))
    print(colored(f" {title} ", color, attrs=["bold"]))
    print(colored("-" * width, color) + "\n")

def print_sentiment_result(result, show_metrics=True):
    """Imprime resultado da análise de sentimento formatado"""
    if result["status"] == "success":
        sentiment = result.get("sentiment", {})
        score = sentiment.get("score")
        classification = sentiment.get("classification", "Não classificado")
        emoji = sentiment.get("emoji", "❓")
        
        # Determinar cor baseada no sentimento
        color = "white"
        if "positiv" in classification.lower():
            color = "green"
        elif "negativ" in classification.lower():
            color = "red"
        elif "neutr" in classification.lower():
            color = "yellow"
            
        # Imprimir cabeçalho de resultado
        print_section(f"{emoji} Sentimento: {classification} ({score:.1f})", color)
        
        # Imprimir resposta principal
        print(result["response"])
        
        # Imprimir detalhes adicionais
        if "key_factors" in sentiment:
            print(colored("\nFatores-Chave:", "cyan", attrs=["bold"]))
            for i, factor in enumerate(sentiment["key_factors"], 1):
                print(colored(f"  {i}. ", "cyan") + factor)
            
        if "market_implications" in sentiment:
            print(colored("\nImplicações de Mercado:", "magenta", attrs=["bold"]))
            for i, imp in enumerate(sentiment["market_implications"], 1):
                print(colored(f"  {i}. ", "magenta") + imp)
        
        if "confidence" in sentiment:
            confidence = sentiment["confidence"]
            conf_color = "green" if confidence.lower() == "alto" else "yellow" if confidence.lower() == "médio" else "red"
            print(colored(f"\nConfiança: ", "blue", attrs=["bold"]) + colored(confidence, conf_color))
        
        # Mostrar métricas se solicitado
        if show_metrics and "metrics" in result:
            print_section("Métricas de Desempenho", "blue")
            for key, value in result["metrics"].items():
                key_formatted = key.replace("_", " ").title()
                if isinstance(value, float):
                    print(f"{key_formatted}: {value:.4f}")
                else:
                    print(f"{key_formatted}: {value}")
    else:
        # Exibir informações de erro
        print_section("❌ Erro na Análise", "red")
        print(colored(f"Status: {result['status']}", "red"))
        print(colored(f"Erro: {result.get('error', 'Desconhecido')}", "red"))

def main():
    """Testar especialista de sentimento com diferentes textos"""
    parser = argparse.ArgumentParser(description="Teste do especialista em sentimento financeiro")
    parser.add_argument("--gpu", type=int, default=None, help="ID da GPU a usar (0, 1, etc.)")
    parser.add_argument("--text", type=str, choices=list(TEST_TEXTS.keys()) + ["all"], default="all", 
                       help="Texto específico para testar ou 'all' para todos")
    parser.add_argument("--save", action="store_true", help="Salvar resultados em arquivo")
    parser.add_argument("--config", type=str, default="models_config.json", help="Caminho para arquivo de configuração")
    parser.add_argument("--metrics", action="store_true", help="Mostrar métricas detalhadas")
    parser.add_argument("--model", type=str, help="Caminho para modelo alternativo")
    parser.add_argument("--raw", action="store_true", help="Incluir resposta bruta do modelo")
    args = parser.parse_args()
    
    print_header("Sistema de Análise de Sentimento Financeiro EzioFilho", "blue")
    print(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Processar caminho de configuração
    config_path = Path(args.config)
    if not config_path.exists():
        print(colored(f"Arquivo de configuração não encontrado: {config_path}", "yellow"))
        print(colored("Usando configuração padrão", "yellow"))
        config_path = None
    else:
        print(f"Usando configuração de: {config_path}")
    
    # Inicializar especialista
    try:
        print_section("Inicializando Especialista", "cyan")
        print(f"GPU ID: {args.gpu if args.gpu is not None else 'Auto-seleção'}")
        
        start_time = time.time()
        expert = SentimentExpert(
            config_path=config_path,
            gpu_id=args.gpu,
            model_path=args.model,
            quantization="8bit"  # Usar 8-bit para economizar memória
        )
        
        init_time = time.time() - start_time
        if not expert.is_initialized:
            raise RuntimeError(f"Erro inicializando especialista: {expert.initialization_error}")
            
        print(colored(f"✅ Especialista inicializado em {init_time:.2f} segundos", "green"))
        
        # Testar especialista
        if args.text == "all":
            # Testar todos os textos de exemplo
            for sentiment_type, text in TEST_TEXTS.items():
                print_header(f"Analisando texto {sentiment_type.upper()}", "cyan")
                print("Texto de entrada:")
                print("---------------")
                print(text.strip())
                print("---------------\n")
                
                result = expert.analyze_sentiment(text, include_raw_response=args.raw)
                print_sentiment_result(result, args.metrics)
                
                # Salvar resultado se solicitado
                if args.save:
                    output_path = expert.save_output(text, result)
                    if output_path:
                        print(colored(f"\nResultado salvo em: {output_path}", "green"))
        else:
            # Testar texto específico
            text = TEST_TEXTS[args.text]
            print_header(f"Analisando texto {args.text.upper()}", "cyan")
            print("Texto de entrada:")
            print("---------------")
            print(text.strip())
            print("---------------\n")
            
            result = expert.analyze_sentiment(text, include_raw_response=args.raw)
            print_sentiment_result(result, args.metrics)
            
            # Salvar resultado se solicitado
            if args.save:
                output_path = expert.save_output(text, result)
                if output_path:
                    print(colored(f"\nResultado salvo em: {output_path}", "green"))
                    
        # Mostrar status final do especialista
        status = expert.get_status()
        print_section("Informações do Sistema", "blue")
        print(f"Especialista: {status['expert_type']} (ID: {status['expert_id']})")
        print(f"Modelo: {status['model_path']}")
        print(f"Dispositivo: {status['device']}")
        print(f"Total de inferências: {status['metrics']['inference_count']}")
        print(f"Tempo de execução: {status['uptime_seconds']:.2f} segundos")
        print(f"Versão: {status['version']}")
        
    except Exception as e:
        import traceback
        print_section("❌ ERRO", "red")
        print(colored(f"Erro durante teste: {e}", "red"))
        traceback.print_exc()

if __name__ == "__main__":
    try:
        import colorama
        colorama.init()  # Para cores no Windows
    except ImportError:
        pass
        
    main()