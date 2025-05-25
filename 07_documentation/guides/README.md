# EzioFilho_LLMGraph - Sistema Financeiro Avançado com IA Adaptativa e Multi-GPU

## Visão Geral

O EzioFilho_LLMGraph é um sistema avançado de IA financeira que utiliza modelos Phi-3 e Phi-2 como cérebros central e especialistas para análise de mercados, otimizado para execução com múltiplas GPUs (RTX 2060 e GTX 1070).

## Arquitetura Revolucionária

O sistema é baseado em cinco camadas tecnológicas interconectadas:

1. **Camada Neural Central**: Phi-3 como cérebro com acesso web irrestrito
2. **Camada de Especialistas**: 12 especialistas Phi-2 ultra-especializados
3. **Camada de Dados Real-Time**: Ingestão massiva de dados financeiros globais
4. **Camada de Execução**: Trading algorithms e risk management automatizado
5. **Camada de Aprendizado**: Auto-evolução contínua via internet e mercados

## Componentes Principais

### Core - Sistema Multi-GPU

- **GPUMonitor**: Sistema de monitoramento em tempo real do uso de memória para múltiplas GPUs.
- **MultiGPUManager**: Orquestrador especializado para balanceamento de modelos entre RTX 2060 e GTX 1070.
- **UniversalModelWrapper**: Interface unificada para diferentes tipos de modelos (Transformers, GGUF, ONNX).

### Especialistas e Agentes

- **Phi-3 Central Brain**: Modelo principal com context length de 128k tokens e acesso à internet.
- **Especialistas Phi-2**: 12 agentes especializados distribuídos em 3 categorias (Mercado, Risco, Quantitativos).
- **Orquestrador Unificado**: Gerencia o fluxo de informações e decisões entre o cérebro central e os especialistas.

### Ferramentas e Diagnósticos

- **GPU Benchmark**: Ferramenta para comparação de performance entre GPUs para diferentes modelos.
- **GPU Monitor Dashboard**: Interface em tempo real para visualização de métricas de GPU.
- **GPU Comparison RTX2060 vs GTX1070**: Análise específica para os modelos de GPU principais.

## Funcionalidades

- **Análise de Sentimento**: Avalia o sentimento em textos com resultados estruturados
- **Verificação de Fatos**: Verifica a veracidade de afirmações (quando o especialista estiver disponível)
- **Perguntas e Respostas**: Responde a perguntas baseadas em texto (quando o especialista estiver disponível)
- **Orquestração Unificada**: Coordena múltiplos especialistas para análises completas
- **Fluxos de Trabalho com Autogen**: Cria conversas entre agentes especialistas
- **Pipelines com LangGraph**: Desenvolve fluxos de processamento direcionados
- **Interface Gráfica Intuitiva**: Acessa funcionalidades do sistema via web

## Instalação

### Requisitos

- Python 3.9+
- CUDA para aceleração GPU (opcional, mas recomendado)

### Pacotes Principais

Instale as dependências com:

```bash
pip install -r requirements-unified.txt
```

### Dependências Opcionais

Para funcionalidades específicas (se desejado):

```bash
# Para suporte PyAutogen
pip install pyautogen>=0.2.0

# Para suporte LangGraph
pip install langgraph>=0.0.15

# Para interface gráfica
pip install gradio>=4.0.0
```

## Uso

### Sistema Unificado Básico

```bash
# Executar análise de texto
python run_unified.py analyze --text "Texto para analisar" --experts sentiment

# Listar especialistas disponíveis
python run_unified.py list-experts

# Analisar texto de um arquivo
python run_unified.py analyze --file caminho/para/arquivo.txt --output resultados.json
```

### Integração com Autogen

```bash
# Executar análise com agentes de IA
python run_unified.py autogen --text "Texto para analisar com agentes" --experts sentiment,factcheck

# Analisar arquivo com Autogen
python run_unified.py autogen --file caminho/para/arquivo.txt --output resultados_autogen.json
```

### Integração com LangGraph

```bash
# Executar análise com grafos direcionados
python run_unified.py graph --text "Texto para analisar com grafos" --experts sentiment,qa
```

### Interface Gráfica

```bash
# Iniciar interface web básica
python run_unified.py gui --basic

# Iniciar interface web avançada
python run_unified.py gui --port 8080 --share
```

## Testes

Execute os testes do sistema unificado:

```bash
# Testar componentes principais
python test_unified.py

# Testar integrações avançadas
python test_integrations.py
```

## Estrutura do Projeto

```
eziofilho-unified/
├── core/                      # Componentes principais unificados
│   ├── unified_base_expert.py       # Classe base de especialistas
│   ├── unified_sentiment_expert.py  # Especialista de sentimento
│   ├── unified_experts.py           # Gerenciamento de especialistas
│   ├── unified_orchestrator.py      # Orquestrador do sistema
│   ├── autogen_integration.py       # Integração com Autogen
│   ├── langgraph_integration.py     # Integração com LangGraph
│   └── gui_interface.py             # Interface gráfica
├── run_unified.py             # Script principal de execução
├── test_unified.py            # Testes de componentes unificados
├── test_integrations.py       # Testes de integrações
└── requirements-unified.txt   # Dependências do sistema
```

## Nota

Este projeto unifica e substitui várias implementações diferentes e duplicadas de especialistas, proporcionando uma base mais sólida e expansível para o sistema.

## Arquitetura Multi-GPU Otimizada

O sistema implementa uma estratégia inteligente de alocação de modelos entre GPUs:

- **RTX 2060**: Priorizada para modelos Phi-3 e tarefas que se beneficiam de Tensor Cores.
- **GTX 1070**: Otimizada para modelos menores como Phi-2, Phi-1.5 e GPT-2.

### Gerenciamento de Recursos

- **Detecção Automática**: Identificação e configuração automática das GPUs disponíveis.
- **Balanceamento Dinâmico**: Algoritmos de balanceamento de carga com pontuação baseada em recência, uso e tamanho dos modelos.
- **Otimização para Phi-3**: Sistema dedicado para priorizar Phi-3 em hardware otimizado.

## Capacidades Revolucionárias

1. **Aprendizado Contínuo via Internet**: Processamento de notícias, papers acadêmicos e regulações em tempo real.
2. **Análise Multi-Dimensional**: Correlações cross-asset, multi-timeframe e global.
3. **Gestão de Risco Avançada**: Sistema integrado de gestão de risco e otimização de portfólio.

## Como Utilizar

### Requisitos

- CUDA 12.1+
- PyTorch 2.1.0+
- 2 GPUs: RTX 2060 (6GB) e GTX 1070 (8GB) ou similares
- Dependências listadas em `requirements.txt`

### Demonstração Multi-GPU

```bash
# Iniciar a demonstração do sistema Multi-GPU
python demo_multi_gpu.py

# Iniciar o painel de monitoramento de GPUs
python gpu_dashboard.py

# Testar a integração Phi-2/Phi-3 (versão simulada)
python test_phi2_phi3_simulated.py

# Testar a integração Phi-2/Phi-3 completa
python test_phi2_phi3_integration.py
```

### Execução do Sistema Completo

```bash
# Executar o sistema EzioFilho_LLMGraph completo (interface de console)
python run_phi2_phi3_system.py --console

# Executar com configuração específica
python run_phi2_phi3_system.py --config config/phi2_phi3_config.json --multi-gpu

# Executar análise de mercado a partir de arquivo JSON
python run_phi2_phi3_system.py --market data/market_analysis_apple.json

# Executar análise de risco a partir de arquivo JSON
python run_phi2_phi3_system.py --risk data/risk_analysis_global.json

# Executar análise quantitativa a partir de arquivo JSON
python run_phi2_phi3_system.py --quant data/quant_analysis_strategy.json

# Definir diretório de saída personalizado
python run_phi2_phi3_system.py --market data/market_analysis_apple.json --output ./resultados_customizados
```

### Otimização para Hardware Específico

```bash
# Executar benchmarks para otimização
python tools/gpu_benchmark.py

# Executar comparação específica RTX 2060 vs GTX 1070
python tools/gpu_comparison_rtx2060_gtx1070.py
```

## Equipe e Contato

- **Marco Barreto**: Arquiteto de Sistemas Financeiros com IA
- **EzioFilho LLMGraph Team**: Especialistas em ML/IA, Finanças e Engenharia de Sistemas

## Licença

Copyright (c) 2025 EzioFilho Systems - Todos os direitos reservados.

### Especialistas Phi-2

O sistema conta com 12 especialistas Phi-2 otimizados para GPUs RTX 2060 e GTX 1070, distribuídos em 3 categorias:

#### Especialistas de Mercado
- **Sentiment Expert**: Análise avançada de sentimento de mercado e notícias
- **Technical Expert**: Análise técnica de padrões gráficos e indicadores
- **Fundamental Expert**: Análise de fundamentos e demonstrações financeiras
- **Macro Expert**: Análise de fatores macroeconômicos e políticos

#### Especialistas de Risco
- **Risk Manager Expert**: Gerenciamento holístico de riscos e métricas consolidadas
- **Volatility Expert**: Análise de padrões e regimes de volatilidade
- **Credit Expert**: Análise de crédito, spreads e ratings
- **Liquidity Expert**: Análise de condições de liquidez e fluxos

#### Especialistas Quantitativos
- **Algorithmic Expert**: Desenvolvimento e avaliação de estratégias algorítmicas
- **Options Expert**: Análise de mercados de opções e derivativos
- **Fixed Income Expert**: Análise de renda fixa e curvas de juros
- **Crypto Expert**: Análise de mercados de criptomoedas e métricas on-chain

### Sistema de Integração Phi-2/Phi-3

O EzioFilho_LLMGraph implementa um sistema sofisticado de integração entre os 12 especialistas Phi-2 e o cérebro central Phi-3:

#### Arquitetura de Integração
- **Roteamento Inteligente**: Direciona consultas para os especialistas mais apropriados
- **Alocação Dinâmica de GPU**: Distribui modelos entre RTX 2060 e GTX 1070 conforme características específicas
- **Agregação de Resultados**: Combina análises de múltiplos especialistas em uma visão coerente
- **Processamento Paralelo**: Executa consultas em paralelo quando possível para maximizar throughput

#### Fluxo de Processamento
1. A consulta é recebida pelo sistema central
2. O orquestrador determina quais especialistas são relevantes para a consulta
3. Os especialistas são carregados nas GPUs disponíveis via MultiGPUManager
4. Os especialistas realizam análises independentes em seus domínios específicos
5. O cérebro Phi-3 integra os resultados dos especialistas, resolvendo contradições
6. Uma resposta consolidada é gerada e entregue ao usuário

#### Uso via API
```python
from core.phi2_phi3_integration import get_phi2_phi3_integrator

# Obter integrador
integrator = get_phi2_phi3_integrator()

# Dados para análise
data = {
    "asset": "AAPL",
    "period": "Q1 2025",
    "description": "Análise após resultados trimestrais da Apple"
}

# Análise completa com todos os especialistas
results = integrator.analyze_with_full_system(data)

# Análise com especialistas específicos
market_results = integrator.analyze_with_full_system(
    data, 
    expert_types=["sentiment_analyst", "technical_analyst"]
)
```
