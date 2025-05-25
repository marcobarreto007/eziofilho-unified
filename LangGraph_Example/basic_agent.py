# Caminho: C:\Users\anapa\SuperIA\EzioFilhoUnified\LangGraph_Example\basic_agent.py

from langgraph.graph import StateGraph, END

class MyState(dict):
    pass

def saudacao_node(state, scratchpad):
    nome = state.get("nome", "visitante")
    # Atualiza o estado adicionando a chave "mensagem"
    state["mensagem"] = f"Olá, {nome}! Seja bem-vindo ao LangGraph."
    return state

# Montando o StateGraph corretamente
workflow = StateGraph(MyState)
workflow.add_node("saudacao", saudacao_node)
workflow.set_entry_point("saudacao")
workflow.add_edge("saudacao", END)
app = workflow.compile()

# Executa o fluxo (com estado inicial)
resultado = app.invoke({"nome": "Marco"})
print(resultado["mensagem"])
