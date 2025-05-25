# 🚀 Sistema AutoGen 1000

Sistema otimizado e funcional com AutoGen + Modelos Locais

## ✅ Instalação Rápida (5 minutos)

### Opção 1: Instalação Automática (Recomendado)
```cmd
cd 01_core_system\fixed_system
install.bat
```

### Opção 2: Instalação Manual
```cmd
# Criar ambiente virtual
py -m venv venv_autogen
venv_autogen\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

## 🎯 Como Usar

### 1. Método Mais Fácil - Com Ollama

1. **Instale Ollama**: https://ollama.ai/download
2. **Execute o sistema**:
```cmd
python run_with_ollama.py
```

O sistema vai:
- ✅ Iniciar Ollama automaticamente
- ✅ Baixar o modelo Mistral
- ✅ Configurar tudo sozinho
- ✅ Abrir chat interativo

### 2. Método Alternativo - Com LM Studio

1. **Instale LM Studio**: https://lmstudio.ai/
2. **Baixe um modelo** (ex: Mistral 7B GGUF)
3. **Inicie o servidor** no LM Studio (porta 1234)
4. **Execute**:
```cmd
python main.py --interactive
```

### 3. Teste Rápido
```cmd
# Testa se tudo está funcionando
python test_simple.py

# Testa conexões
python main.py --test

# Chat direto
python main.py --chat "Olá! Como você está?"
```

## 📁 Estrutura

```
fixed_system/
├── main.py              # Sistema principal
├── run_with_ollama.py   # Versão fácil com Ollama
├── local_config.py      # Config modelos locais
├── test_simple.py       # Teste de componentes
├── install.bat          # Instalador automático
├── requirements.txt     # Dependências
└── README.md           # Este arquivo
```

## 🔧 Solução de Problemas

### "AutoGen não encontrado"
```cmd
pip install pyautogen==0.2.18
```

### "Ollama não está rodando"
```cmd
# Instale Ollama primeiro
# Depois execute manualmente:
ollama serve
```

### "Erro de conexão"
- Verifique se o servidor (Ollama/LM Studio) está rodando
- Confirme a porta (11434 para Ollama, 1234 para LM Studio)

### "Modelo não encontrado"
```cmd
# Para Ollama:
ollama pull mistral

# Para LM Studio:
# Baixe modelos GGUF e carregue na interface
```

## 🌟 Recursos do Sistema

- ✅ **Multi-modelo**: Suporta Ollama, LM Studio, OpenAI
- ✅ **Auto-configuração**: Detecta e configura automaticamente
- ✅ **Modo interativo**: Chat em tempo real
- ✅ **Tolerante a falhas**: Tenta múltiplas configurações
- ✅ **Logs claros**: Mostra exatamente o que está acontecendo

## 💡 Dicas Pro

1. **Performance**: Use GPU se disponível
   ```python
   # Em local_config.py, ajuste:
   n_gpu_layers = 35  # Para GPU
   ```

2. **Modelos recomendados**:
   - Mistral 7B (equilibrado)
   - Phi-2 (rápido, leve)
   - Llama 2 13B (mais inteligente)

3. **Múltiplos agentes**:
   ```python
   # Adicione mais agentes em main.py
   expert = AssistantAgent(name="expert", ...)
   ```

## 📞 Suporte

Problemas? Abra o chat e digite:
```
python run_with_ollama.py --test
```

Vai mostrar exatamente o que está faltando!

---

**Sistema AutoGen 1000** - Feito para funcionar! 🚀