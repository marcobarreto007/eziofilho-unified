REM 1. Abra o arquivo para edição
notepad "C:\Users\anapa\SuperIA\EzioFilhoUnified\LangGraph_Example\chatbot_agent.py"

REM 2. Substitua TODO o conteúdo pelo código abaixo (salve e feche)
REM    --- (código em inglês, pronto) ---
# Path: C:/Users/anapa/SuperIA/EzioFilhoUnified/LangGraph_Example/chatbot_agent.py
"""
Conversational agent using LangGraph with HuggingFace (local) or Claude (Anthropic).
Compatible with Python 3.10+. Requires dependencies installed and ANTHROPIC_API_KEY for Claude.
"""

import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(State)

use_claude = bool(os.getenv("ANTHROPIC_API_KEY"))
if use_claude:
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model="claude-2")
else:
    from transformers import pipeline
    llm = pipeline("text-generation", model="distilgpt2", tokenizer="distilgpt2")

def chatbot_node(state: State) -> State:
    user_msg = state["messages"][-1][1] if state["messages"] else ""
    if use_claude:
        reply = llm.invoke(state["messages"]).content
    else:
        gen = llm(user_msg, max_length=100, do_sample=True, pad_token_id=50256)
        reply = gen[0]["generated_text"][len(user_msg):].strip()
    return {"messages": [("assistant", reply)]}

builder.add_node("chatbot", chatbot_node)
builder.set_entry_point("chatbot")
builder.set_finish_point("chatbot")
graph = builder.compile()

print("Assistant: Hello! How can I help you?")

history = []
while True:
    user = input("You: ")
    if user.lower() in {"q", "quit", "exit", "sair"}:
        print("Assistant: Goodbye!")
        break
    history.append(("user", user))
    for event in graph.stream({"messages": history}):
        for value in event.values():
            reply = value["messages"][-1][1]  # tuple (role, text)
            print(f"Assistant: {reply}")
            history.append(("assistant", reply))
# --- fim do código ---
REM 3. Execute novamente
python "C:\Users\anapa\SuperIA\EzioFilhoUnified\LangGraph_Example\chatbot_agent.py"
