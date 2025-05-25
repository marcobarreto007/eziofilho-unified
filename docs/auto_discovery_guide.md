# Sistema de Descoberta Automática de Modelos

## Visão Geral

O Sistema de Descoberta Automática de Modelos do EzioFilho_LLMGraph permite que o sistema identifique e configure automaticamente modelos de linguagem disponíveis no seu ambiente. Essa funcionalidade simplifica significativamente o uso do sistema, eliminando a necessidade de configuração manual de modelos.

## Funcionalidades Principais

- **Detecção automática** de modelos locais baseados em diversos formatos (GGUF, ONNX, SafeTensors, etc.)
- **Identificação de capacidades** dos modelos detectados
- **Mapeamento inteligente** para papéis de especialistas financeiros
- **Interface gráfica** para gerenciamento de modelos
- **Configuração automática** otimizada para hardware disponível

## Como Usar

### Via Interface Gráfica

1. Execute o script `run_with_auto_discovery.py` com a opção `--gui`:

```bash
python run_with_auto_discovery.py --gui
```

2. Na interface gráfica, navegue até a aba "Gestão de Modelos"
3. Clique em "Detectar Modelos" para iniciar a descoberta
4. Os modelos detectados serão exibidos na tabela
5. Clique em um modelo para ver seus detalhes
6. Os modelos detectados são automaticamente registrados nos especialistas adequados

### Via Linha de Comando

1. Execute o script principal com a descoberta automática de modelos:

```bash
python main.py
```

2. Para desativar a descoberta automática, use a opção `--no-auto-discovery`:

```bash
python main.py --no-auto-discovery
```

3. Para usar o script de conveniência:

```bash
python run_with_auto_discovery.py
```

## Configuração Avançada

### Diretórios de Busca

Por padrão, o sistema busca modelos nos seguintes diretórios:
- `./models/` - Diretório local de modelos
- `~/.cache/models/` - Cache padrão de modelos
- `~/.local/share/models/` - Diretório de modelos compartilhados
- Diretórios comuns de modelos do Hugging Face

Para personalizar os diretórios de busca, você pode usar a interface gráfica ou modificar o arquivo `model_auto_discovery.py`:

```python
model_discovery = ModelAutoDiscovery()
model_discovery.set_search_paths([
    "/caminho/para/seus/modelos",
    "/outro/caminho"
])
```

### Padrões de Arquivos

Por padrão, o sistema procura pelos seguintes padrões de arquivos:
- `*.gguf` - Modelos GGUF (LLaMa.cpp)
- `*.onnx` - Modelos ONNX 
- `*.safetensors` - Modelos no formato SafeTensors
- `*.bin` - Modelos binários genéricos

Para adicionar novos padrões, use a interface gráfica ou:

```python
model_discovery.set_model_patterns([
    "*.gguf", "*.onnx", "*.safetensors", "*.bin", "seu_padrao.*"
])
```

## Desenvolvimento

### Classes Principais

- `ModelDiscovery` - Lida com a detecção e validação de modelos
- `UniversalModelWrapper` - Fornece uma interface unificada para diferentes modelos
- `AutoConfiguration` - Gera configurações otimizadas para modelos detectados

### Integração com Outros Componentes

O sistema se integra com:
- `UnifiedOrchestrator` - Para registro de modelos como especialistas
- `GUIInterface` - Para gerenciamento via interface gráfica
- `ModelRouter` - Para roteamento de solicitações aos modelos apropriados

## Resolução de Problemas

### Modelos Não Detectados

1. Verifique se os arquivos dos modelos existem nos caminhos esperados
2. Verifique se os padrões de arquivos incluem o formato do seu modelo
3. Verifique as permissões de acesso aos diretórios

### Erros de Configuração

1. Verifique o log para mensagens de erro detalhadas
2. Certifique-se de que as dependências para o tipo de modelo estão instaladas
3. Tente especificar manualmente o tipo de modelo via interface gráfica

## Expansão

O sistema foi projetado para ser facilmente expandido:

1. Adicione suporte para novos tipos de modelos no `UniversalModelWrapper`
2. Implemente novos detectores de capacidades na classe `ModelDiscovery`
3. Adicione novos papéis de especialistas na função de mapeamento

---

Para mais informações, consulte a documentação do sistema ou entre em contato com a equipe de desenvolvimento.
