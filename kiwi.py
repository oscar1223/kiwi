#!/Users/racso/kiwi/venv/bin/python3

from langchain_ollama import ChatOllama
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from langfuse.langchain import CallbackHandler as LangfuseCallback
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
import subprocess
import os
import json
import uuid
import socket
from dotenv import load_dotenv
from ddgs import DDGS

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

HISTORY_FILE = os.path.expanduser("~/.kiwi_history.json")
SYSTEM_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "system_prompt.txt")

console = Console()

MAX_OUTPUT = 2000

# --- CALLBACKS ---

class ToolLogger(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "tool")
        console.log(f"[dim]→ {name}[/dim]")

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
        return f"ERROR: El archivo '{path}' no existe. Este tool solo sirve para archivos reales del usuario. Responde directamente con tu conocimiento sin usar tools."

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

tools = [read_file, write_file, run_command, search_web]

# --- LLM ---

llm = ChatOllama(model="kimi-k2.5:cloud", temperature=0)

# --- PROMPT ---

with open(SYSTEM_PROMPT_FILE, "r") as f:
    system_prompt = f.read()

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# --- AGENTE ---

agent = create_tool_calling_agent(llm, tools, prompt)
langfuse = LangfuseCallback()

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
    callbacks=[ToolLogger(), langfuse],
)

# --- UI ---

def show_banner():
    console.print(Panel.fit(
        "[bold green]🥝 Kiwi Agent[/bold green]\n[dim]Asistente local · M3 Pro · Kimi K2.5 Cloud[/dim]",
        border_style="green"
    ))
    console.print("[dim]Escribe [bold]salir[/bold] para terminar\n[/dim]")

def show_response(text):
    console.print(Panel(
        Text(text, style="white"),
        title="[bold green]Kiwi[/bold green]",
        border_style="green"
    ))

def show_error(text):
    console.print(Panel(
        Text(str(text), style="red"),
        title="[bold red]Error[/bold red]",
        border_style="red"
    ))

# --- MEMORIA ---

MAX_HISTORY_TURNS = 6

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

def save_history(history: list):
    raw = [
        {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
        for m in history
    ]
    with open(HISTORY_FILE, "w") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

chat_history = load_history()

# --- LANGFUSE SESSION CONFIG ---
SESSION_ID = str(uuid.uuid4())[:8]
USER_ID = socket.gethostname() or "kiwi-user"

# --- MAIN ---

show_banner()

while True:
    try:
        user_input = Prompt.ask("[bold cyan]Tú[/bold cyan]")
    except KeyboardInterrupt:
        save_history(chat_history)
        console.print("\n[dim]Hasta luego! 👋[/dim]")
        break

    if user_input.lower() in ["salir", "exit", "quit", "q"]:
        save_history(chat_history)
        console.print("[dim]Hasta luego! 👋[/dim]")
        break

    try:
        with console.status("[dim yellow]Pensando...[/dim yellow]", spinner="dots"):
            response = executor.invoke(
                {
                    "input": user_input,
                    "chat_history": chat_history,
                },
                config={
                    "metadata": {
                        "langfuse_user_id": USER_ID,
                        "langfuse_session_id": SESSION_ID,
                        "langfuse_tags": ["kiwi-agent", "local", "ollama"],
                    }
                }
            )

        answer = response["output"]
        show_response(answer)

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))

        if len(chat_history) > MAX_HISTORY_TURNS * 2:
            chat_history = chat_history[-(MAX_HISTORY_TURNS * 2):]

    except Exception as e:
        show_error(e)
