#!/usr/bin/env python3
"""Configuración del agente Kiwi: LLM, tools y memoria de conversación."""

import json
import os
import socket
import subprocess
import uuid

from ddgs import DDGS
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler as LangfuseCallback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

HISTORY_FILE = os.path.expanduser("~/.kiwi_history.json")
SYSTEM_PROMPT_FILE = os.path.join(BASE_DIR, "system_prompt.txt")

MAX_OUTPUT = 2000
MAX_HISTORY_TURNS = 6

# --- TOOLS ---


@tool
def read_file(path: str) -> str:
    """Lee un archivo del disco. Input: ruta real del archivo."""
    try:
        with open(os.path.expanduser(path.strip()), "r") as f:
            content = f.read()
        if len(content) > MAX_OUTPUT:
            content = content[:MAX_OUTPUT] + f"\n... [truncado, {len(content)} chars total]"
        return content
    except FileNotFoundError:
        return (
            f"ERROR: El archivo '{path}' no existe. Este tool solo sirve para archivos "
            "reales del usuario. Responde directamente con tu conocimiento sin usar tools."
        )


@tool
def write_file(path: str, content: str) -> str:
    """Escribe contenido en un archivo.

    Args:
        path: ruta del archivo
        content: contenido a escribir
    """
    with open(os.path.expanduser(path.strip()), "w") as f:
        f.write(content)
    return f"Archivo guardado en {path.strip()}"


@tool
def run_command(command: str) -> str:
    """Ejecuta cualquier comando bash: mkdir, ls, cp, rm, git, etc. Input: comando bash."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout or result.stderr
    if len(output) > MAX_OUTPUT:
        output = output[:MAX_OUTPUT] + "\n... [truncado]"
    return output


@tool
def search_web(query: str) -> str:
    """Busca en internet. Input: query de búsqueda."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    return "\n".join([r["body"] for r in results])[:MAX_OUTPUT]


TOOLS = [read_file, write_file, run_command, search_web]


class ToolEventLogger(BaseCallbackHandler):
    """Reenvía el inicio de cada tool call a un callback externo (usado por la TUI)."""

    def __init__(self, on_tool_start=None):
        self._on_tool_start = on_tool_start

    def on_tool_start(self, serialized, input_str, **kwargs):
        if self._on_tool_start:
            name = serialized.get("name", "tool")
            self._on_tool_start(name, input_str)


def build_executor(on_tool_start=None) -> AgentExecutor:
    """Construye el AgentExecutor de Kiwi.

    on_tool_start: callback opcional (name: str, input_str: str) -> None,
    invocado cada vez que el agente llama a una tool.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    with open(SYSTEM_PROMPT_FILE, "r") as f:
        system_prompt = f.read()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    langfuse = LangfuseCallback()

    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=False,
        handle_parsing_errors=True,
        callbacks=[ToolEventLogger(on_tool_start), langfuse],
    )


# --- MEMORIA ---


def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        raw = json.load(f)
    messages = []
    for msg in raw:
        if msg["role"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


def save_history(history: list) -> None:
    raw = [
        {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
        for m in history
    ]
    with open(HISTORY_FILE, "w") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)


def trim_history(history: list) -> list:
    if len(history) > MAX_HISTORY_TURNS * 2:
        return history[-(MAX_HISTORY_TURNS * 2):]
    return history


def new_session_ids() -> tuple[str, str]:
    session_id = str(uuid.uuid4())[:8]
    user_id = socket.gethostname() or "kiwi-user"
    return session_id, user_id
