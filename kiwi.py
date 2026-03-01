#!/Users/racso/kiwi/venv/bin/python3

from langchain_ollama import ChatOllama
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
import subprocess
import os
from duckduckgo_search import DDGS

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
    """Lee un archivo. Input: ruta del archivo."""
    with open(os.path.expanduser(path.strip()), "r") as f:
        content = f.read()
    if len(content) > MAX_OUTPUT:
        content = content[:MAX_OUTPUT] + f"\n... [truncado, {len(content)} chars total]"
    return content

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

llm = ChatOllama(model="kiwi", temperature=0)

# --- PROMPT ---

prompt = ChatPromptTemplate.from_messages([
    ("system", """Eres Kiwi, un asistente local corriendo en el Mac del usuario.
No eres un servicio en la nube. Responde siempre en español y de forma concisa.

Tools disponibles:
- run_command: para bash (mkdir, ls, mv, cp, rm, git, etc.)
- write_file: para escribir contenido en un archivo (usa path y content por separado)
- read_file: para leer el contenido de un archivo
- search_web: para buscar en internet

Ejemplos de uso correcto:
- Crear carpeta → run_command("mkdir -p ~/Desktop/nueva-carpeta")
- Ver archivos → run_command("ls ~/Desktop")
- Leer archivo → read_file("~/archivo.txt")
- Escribir archivo → write_file(path="~/archivo.txt", content="hola mundo")
- Buscar info → search_web("cómo usar Python asyncio")"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# --- AGENTE ---

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
    callbacks=[ToolLogger()],
)

# --- UI ---

def show_banner():
    console.print(Panel.fit(
        "[bold green]🥝 Kiwi Agent[/bold green]\n[dim]Asistente local · M3 Pro · Llama 3.1 8B[/dim]",
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

chat_history = []
MAX_HISTORY_TURNS = 6  # 6 turnos = 12 mensajes, razonable para modelos pequeños

# --- MAIN ---

show_banner()

while True:
    try:
        user_input = Prompt.ask("[bold cyan]Tú[/bold cyan]")
    except KeyboardInterrupt:
        console.print("\n[dim]Hasta luego! 👋[/dim]")
        break

    if user_input.lower() in ["salir", "exit", "quit", "q"]:
        console.print("[dim]Hasta luego! 👋[/dim]")
        break

    try:
        with console.status("[dim yellow]Pensando...[/dim yellow]", spinner="dots"):
            response = executor.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })

        answer = response["output"]
        show_response(answer)

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))

        if len(chat_history) > MAX_HISTORY_TURNS * 2:
            chat_history = chat_history[-(MAX_HISTORY_TURNS * 2):]

    except Exception as e:
        show_error(e)
