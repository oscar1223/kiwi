#!/usr/bin/env python3
"""Configuración del agente Kiwi: LLM, tools y memoria de conversación."""

import asyncio
import difflib
import json
import os
import re
import socket
import subprocess
import uuid
import warnings
from typing import Callable, Optional

import yaml
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool, tool
from langchain_openai import ChatOpenAI

# langchain_mcp_adapters y langfuse importan transitivamente langgraph.checkpoint, que
# al cargarse emite un LangChainPendingDeprecationWarning ruidoso (avisa de un cambio
# futuro en el uso interno que hace langgraph de allowed_objects, nada que dependa de
# código de kiwi). Los imports de langchain_classic/langchain_core de arriba reactivan
# ese aviso con un filtro "default" cada vez que se cargan, así que el filtro de
# silencio tiene que ir aquí — justo antes de los imports que de verdad lo disparan —
# y no al principio del archivo, o quedaría tapado.
warnings.filterwarnings("ignore", category=PendingDeprecationWarning, module=r"^langgraph\.checkpoint")

from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: E402
from langfuse.langchain import CallbackHandler as LangfuseCallback  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

HISTORY_FILE = os.path.expanduser("~/.kiwi_history.json")
SYSTEM_PROMPT_FILE = os.path.join(BASE_DIR, "system_prompt.txt")
SKILLS_DIR = os.path.join(BASE_DIR, "skills")
MCP_CONFIG_FILE = os.path.join(BASE_DIR, "mcp_servers.json")
MODEL_CONFIG_FILE = os.path.join(BASE_DIR, "models.json")
ENV_FILE = os.path.join(BASE_DIR, ".env")
PROJECT_CONFIG_FILENAMES = ["KIWI.md", "AGENTS.md"]

MAX_OUTPUT = 2000

# Compactación de contexto: si el historial estimado supera este presupuesto de
# tokens, se resumen los turnos más antiguos y se conservan sin tocar los últimos
# KEEP_RECENT_TURNS turnos (par human/ai).
MAX_HISTORY_TOKENS = 6000
KEEP_RECENT_TURNS = 6

# Patrones de comandos que pueden causar daño irreversible o afectar fuera del
# proyecto actual: se marcan como peligrosos en el prompt de permiso, pero
# siguen requiriendo la misma confirmación explícita que cualquier otro comando.
DANGEROUS_COMMAND_PATTERNS = [
    r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f\b",  # rm -rf, rm -fr, ...
    r"\bsudo\b",
    r"\bgit\s+push\b.*--force",
    r"\bgit\s+reset\s+--hard\b",
    r"\bdd\s+if=",
    r"\bmkfs\b",
    r">\s*/dev/sd",
    r"\bchmod\s+-R\s+777\b",
    r"\bcurl\b.*\|\s*(sh|bash)\b",
]


# --- MODOS ---
#
# Cada modo cambia dos cosas: qué tan agresivo es el sistema de permisos
# (resolve_permission_policy) y las instrucciones que recibe el modelo para esa
# tanda (MODE_INSTRUCTIONS, inyectado por tui.py en el input de cada turno).


class Mode:
    ASK = "ask"
    PLAN = "plan"
    WORK = "work"


MODE_LABELS = {
    Mode.ASK: "Ask Mode",
    Mode.PLAN: "Plan Mode",
    Mode.WORK: "Work Mode",
}

# Orden de ciclo al pulsar Shift+Tab, de más a menos conservador.
MODE_ORDER = [Mode.ASK, Mode.PLAN, Mode.WORK]

MODE_INSTRUCTIONS = {
    Mode.ASK: "",
    Mode.PLAN: (
        "[MODO ACTUAL: Plan Mode] No ejecutes cambios: write_file y edit_file están "
        "bloqueados, y run_command solo funciona para comandos de solo lectura (ls, cat, "
        "grep, find, git status/log/diff/show, etc.) — cualquier otro comando se deniega "
        "automáticamente sin preguntar al usuario. Tu tarea es investigar lo necesario y "
        "responder con un plan claro y numerado de los pasos a seguir, sin aplicarlos. Si "
        "una tool es bloqueada, no insistas ni la repitas: menciona en el plan que ese paso "
        "requiere salir de Plan Mode."
    ),
    Mode.WORK: (
        "[MODO ACTUAL: Work Mode] write_file y edit_file se aplican sin pedir confirmación "
        "al usuario — sé especialmente cuidadoso antes de escribir. run_command y las tools "
        "MCP siguen pidiendo confirmación normalmente."
    ),
}

# Heurística de solo-lectura para Plan Mode. No es un sandbox real (no analiza
# subshells, variables, etc.), solo evita interrumpir al usuario con comandos
# que claramente exploran sin modificar nada; cualquier cosa ambigua se trata
# como insegura y se deniega.
PLAN_MODE_READONLY_VERBS = {
    "ls", "cat", "grep", "find", "head", "tail", "wc", "pwd", "tree",
    "file", "which", "echo", "du", "df", "ps", "whoami", "date", "diff",
}
PLAN_MODE_GIT_READONLY_SUBCOMMANDS = {
    "status", "log", "diff", "show", "branch", "remote", "blame", "shortlog",
}
PLAN_MODE_MUTATION_MARKERS = {
    "rm", "mv", "cp", "touch", "mkdir", "chmod", "chown", "sudo", "kill", "dd", "install",
}


def is_plan_mode_safe_command(command: str) -> bool:
    if is_dangerous_command(command):
        return False
    segments = re.split(r"&&|\|\||;|\|", command)
    saw_segment = False
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        saw_segment = True
        if ">" in segment:
            return False
        words = segment.split()
        if not words:
            continue
        if any(marker in words for marker in PLAN_MODE_MUTATION_MARKERS):
            return False
        verb = words[0]
        if verb == "git":
            if len(words) < 2 or words[1] not in PLAN_MODE_GIT_READONLY_SUBCOMMANDS:
                return False
        elif verb not in PLAN_MODE_READONLY_VERBS:
            return False
    return saw_segment


def resolve_permission_policy(mode: str, action: str, detail: str, dangerous: bool) -> Optional[bool]:
    """Decide si una acción se autoriza/deniega automáticamente según el modo
    actual, sin preguntar al usuario. Devuelve True/False si hay una decisión
    automática, o None si hay que preguntar interactivamente (modo Ask, o
    cualquier caso que el modo actual no decide por sí solo)."""
    if mode == Mode.PLAN:
        if action in ("write_file", "edit_file"):
            return False
        if action == "run_command":
            return is_plan_mode_safe_command(detail)
        if action.startswith("mcp:"):
            return False
        return None
    if mode == Mode.WORK:
        if action in ("write_file", "edit_file"):
            return True
        return None
    return None


# --- PERMISOS ---

# Callback global inyectado por build_executor(): decide si una acción
# potencialmente destructiva (escribir un archivo, ejecutar un comando) puede
# proceder. Se ejecuta en el hilo de la tool y debe bloquear hasta tener
# respuesta del usuario (ver tui.py:_request_permission).
_permission_callback: Optional[Callable[[str, str, bool, Optional[str]], bool]] = None


def set_permission_callback(callback: Optional[Callable[[str, str, bool, Optional[str]], bool]]) -> None:
    global _permission_callback
    _permission_callback = callback


def is_dangerous_command(command: str) -> bool:
    return any(re.search(pattern, command) for pattern in DANGEROUS_COMMAND_PATTERNS)


def _ask_permission(action: str, detail: str, dangerous: bool = False, diff: Optional[str] = None) -> bool:
    if _permission_callback is None:
        return True
    return _permission_callback(action, detail, dangerous, diff)


def _make_diff(path: str, before: str, after: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=path,
        tofile=path,
        n=2,
    )
    return "".join(diff)


# --- SKILLS ---
#
# Cada archivo .md en skills/ es una skill: un bloque de instrucciones que el
# modelo carga bajo demanda (vía el tool load_skill) en vez de tenerlo siempre
# en el system prompt. Formato:
#
#   ---
#   name: nombre-skill
#   description: cuándo usar esta skill (una frase)
#   ---
#   Instrucciones completas...


def _parse_skill_file(path: str) -> Optional[dict]:
    with open(path, "r") as f:
        raw = f.read()
    if not raw.startswith("---"):
        return None
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return None
    _, frontmatter_raw, body = parts
    meta = yaml.safe_load(frontmatter_raw) or {}
    name = meta.get("name") or os.path.splitext(os.path.basename(path))[0]
    return {"name": name, "description": meta.get("description", ""), "body": body.strip()}


def load_skills() -> dict:
    """Devuelve {name: {description, body}} para cada skill en skills/."""
    skills = {}
    if not os.path.isdir(SKILLS_DIR):
        return skills
    for fname in sorted(os.listdir(SKILLS_DIR)):
        if not fname.endswith(".md"):
            continue
        parsed = _parse_skill_file(os.path.join(SKILLS_DIR, fname))
        if parsed:
            skills[parsed["name"]] = parsed
    return skills


def save_skill(name: str, description: str, body: str) -> str:
    """Crea o sobrescribe skills/<name>.md con el frontmatter esperado por
    _parse_skill_file. Devuelve la ruta del archivo."""
    os.makedirs(SKILLS_DIR, exist_ok=True)
    path = os.path.join(SKILLS_DIR, f"{name}.md")
    content = f"---\nname: {name}\ndescription: {description}\n---\n{body.strip()}\n"
    with open(path, "w") as f:
        f.write(content)
    return path


def delete_skill(name: str) -> None:
    path = os.path.join(SKILLS_DIR, f"{name}.md")
    if not os.path.exists(path):
        raise ValueError(f"No existe la skill '{name}'.")
    os.remove(path)


def skills_summary(skills: dict) -> str:
    if not skills:
        return ""
    lines = [f"- {name}: {info['description']}" for name, info in skills.items()]
    return (
        "\n\nSkills disponibles (usa load_skill(nombre) para cargar sus "
        "instrucciones completas SOLO si la tarea encaja con la descripción):\n"
        + "\n".join(lines)
    )


# --- CONFIGURACIÓN POR PROYECTO ---

IGNORED_TOP_LEVEL_ENTRIES = {
    ".git", "venv", ".venv", "node_modules", "__pycache__", "dist", "build", ".idea",
}
MAX_TOP_LEVEL_ENTRIES = 40


def working_directory_context() -> str:
    """Describe dónde se está ejecutando kiwi: el cwd absoluto y un listado de
    nivel superior, para que el modelo sepa a qué ruta se refiere el usuario
    cuando dice 'este repo' o 'este proyecto' sin tener que inventársela."""
    cwd = os.getcwd()
    try:
        entries = sorted(os.listdir(cwd))
    except OSError:
        entries = []
    visible = [e for e in entries if e not in IGNORED_TOP_LEVEL_ENTRIES]
    if len(visible) > MAX_TOP_LEVEL_ENTRIES:
        hidden_count = len(visible) - MAX_TOP_LEVEL_ENTRIES
        visible = visible[:MAX_TOP_LEVEL_ENTRIES] + [f"... ({hidden_count} más, usa run_command para verlos)"]

    listing = "\n".join(f"- {e}" for e in visible) if visible else "(directorio vacío)"
    return (
        "\n\n## Directorio de trabajo actual\n"
        f"Kiwi se está ejecutando en: {cwd}\n"
        "Cuando el usuario diga 'este repo', 'este proyecto' o 'la carpeta actual', se "
        "refiere a esta ruta: úsala como base para read_file/write_file/edit_file/"
        "run_command en vez de preguntarla o inventarla. Contenido de nivel superior:\n"
        f"{listing}"
    )


def load_project_config() -> str:
    """Lee KIWI.md o AGENTS.md del directorio desde el que se lanzó kiwi, si existe."""
    for fname in PROJECT_CONFIG_FILENAMES:
        path = os.path.join(os.getcwd(), fname)
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.read().strip()
    return ""


# --- MODELOS ---
#
# Kiwi es agnóstico de proveedor: un "perfil" de modelo es {provider, model,
# [base_url], [api_key_env]} guardado en models.json (gitignored, ver
# models.json.example). El perfil activo se reconstruye en build_executor()
# cada vez que cambia (ver KiwiApp._rebuild_executor en tui.py).

BUILTIN_MODEL_PROFILES = {
    "gpt-4o-mini": {"provider": "openai", "model": "gpt-4o-mini"},
}
DEFAULT_MODEL_NAME = "gpt-4o-mini"

# api_key_env es fijo para openai/anthropic; para "ollama" no hace falta key
# (modelo local); para "openai_compatible" lo define cada perfil, porque cada
# proveedor compatible (DeepSeek, Qwen, Kimi, Groq, OpenRouter...) usa su
# propia variable de entorno.
PROVIDER_INFO = {
    "openai": {"label": "OpenAI", "api_key_env": "OPENAI_API_KEY", "custom_base_url": False},
    "anthropic": {"label": "Anthropic", "api_key_env": "ANTHROPIC_API_KEY", "custom_base_url": False},
    "ollama": {"label": "Ollama (local)", "api_key_env": None, "custom_base_url": False},
    "openai_compatible": {
        "label": "Compatible con API OpenAI (DeepSeek, Qwen, Kimi, Groq, OpenRouter...)",
        "api_key_env": None,
        "custom_base_url": True,
    },
}


def load_model_config() -> dict:
    """Perfiles built-in fusionados con models.json (si existe). Los
    perfiles definidos por el usuario pueden sobrescribir los built-in por
    nombre. Sin models.json, Kiwi arranca igual que antes (gpt-4o-mini)."""
    profiles = dict(BUILTIN_MODEL_PROFILES)
    current = DEFAULT_MODEL_NAME
    if os.path.exists(MODEL_CONFIG_FILE):
        try:
            with open(MODEL_CONFIG_FILE, "r") as f:
                raw = json.load(f)
            profiles.update(raw.get("profiles", {}))
            current = raw.get("current", current)
        except (json.JSONDecodeError, OSError):
            pass
    if current not in profiles:
        current = next(iter(profiles), DEFAULT_MODEL_NAME)
    return {"current": current, "profiles": profiles}


def _save_model_config(cfg: dict) -> None:
    # Solo se persisten los perfiles que no son built-in idénticos, para que
    # models.json quede legible y no repita lo que ya trae Kiwi de fábrica.
    user_profiles = {
        name: profile
        for name, profile in cfg["profiles"].items()
        if BUILTIN_MODEL_PROFILES.get(name) != profile
    }
    with open(MODEL_CONFIG_FILE, "w") as f:
        json.dump({"current": cfg["current"], "profiles": user_profiles}, f, ensure_ascii=False, indent=2)


def set_current_model(name: str) -> None:
    cfg = load_model_config()
    if name not in cfg["profiles"]:
        raise ValueError(f"No existe el perfil de modelo '{name}'.")
    cfg["current"] = name
    _save_model_config(cfg)


def add_model_profile(name: str, provider: str, model: str, base_url: str = None, api_key_env: str = None) -> None:
    if provider not in PROVIDER_INFO:
        raise ValueError(f"Proveedor desconocido: {provider}")
    cfg = load_model_config()
    profile = {"provider": provider, "model": model}
    if base_url:
        profile["base_url"] = base_url
    if api_key_env:
        profile["api_key_env"] = api_key_env
    cfg["profiles"][name] = profile
    _save_model_config(cfg)


def remove_model_profile(name: str) -> None:
    cfg = load_model_config()
    if name not in cfg["profiles"]:
        raise ValueError(f"No existe el perfil de modelo '{name}'.")
    if cfg["current"] == name:
        raise ValueError("No se puede eliminar el perfil de modelo activo. Cambia de modelo primero.")
    del cfg["profiles"][name]
    _save_model_config(cfg)


def get_current_profile() -> dict:
    cfg = load_model_config()
    return cfg["profiles"][cfg["current"]]


def profile_api_key_env(profile: dict) -> Optional[str]:
    """Nombre de la variable de entorno que necesita este perfil, o None si
    no hace falta (p. ej. ollama local)."""
    if profile.get("api_key_env"):
        return profile["api_key_env"]
    return PROVIDER_INFO.get(profile["provider"], {}).get("api_key_env")


def current_model_label() -> str:
    cfg = load_model_config()
    profile = cfg["profiles"][cfg["current"]]
    return f"{cfg['current']} ({profile['provider']}/{profile['model']})"


def build_llm_from_profile(profile: dict):
    provider = profile["provider"]
    model = profile["model"]
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=0)
    if provider == "openai_compatible":
        api_key_env = profile_api_key_env(profile)
        return ChatOpenAI(
            model=model,
            temperature=0,
            base_url=profile.get("base_url"),
            api_key=os.getenv(api_key_env) if api_key_env else None,
        )
    raise ValueError(f"Proveedor desconocido: {provider}")


# --- ENTORNO ---
#
# API keys y demás variables de entorno se guardan en .env (gitignored, ya
# cargado con load_dotenv al importar este módulo). set_env_value/unset_env_value
# reescriben el archivo preservando el resto de líneas y recargan el entorno
# del proceso actual, así los cambios aplican sin reiniciar Kiwi.


def _read_env_lines() -> list:
    if not os.path.exists(ENV_FILE):
        return []
    with open(ENV_FILE, "r") as f:
        return f.readlines()


def list_env_keys() -> list:
    """Unión de las variables que los proveedores conocidos necesitan, las
    que usan los perfiles de modelo guardados, y las que ya estén en .env."""
    keys = set()
    for info in PROVIDER_INFO.values():
        if info["api_key_env"]:
            keys.add(info["api_key_env"])
    cfg = load_model_config()
    for profile in cfg["profiles"].values():
        env_name = profile_api_key_env(profile)
        if env_name:
            keys.add(env_name)
    for line in _read_env_lines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.add(line.split("=", 1)[0].strip())
    return sorted(keys)


def get_masked_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        return "(no configurada)"
    if len(value) <= 4:
        return "*" * len(value)
    return "*" * (len(value) - 4) + value[-4:]


def set_env_value(key: str, value: str) -> None:
    lines = _read_env_lines()
    prefix = f"{key}="
    new_line = f"{key}={value}\n"
    for i, line in enumerate(lines):
        if line.strip().startswith(prefix):
            lines[i] = new_line
            break
    else:
        lines.append(new_line)
    with open(ENV_FILE, "w") as f:
        f.writelines(lines)
    os.environ[key] = value


def unset_env_value(key: str) -> None:
    lines = _read_env_lines()
    prefix = f"{key}="
    lines = [line for line in lines if not line.strip().startswith(prefix)]
    with open(ENV_FILE, "w") as f:
        f.writelines(lines)
    os.environ.pop(key, None)


# --- MCP ---


def _wrap_mcp_tool_with_permission(mcp_tool):
    """Envuelve una tool MCP externa con el mismo sistema de permisos que
    write_file/run_command/edit_file: toda tool MCP pide confirmación antes de
    ejecutarse, porque no controlamos qué hace internamente."""

    async def gated(**kwargs):
        detail = f"{mcp_tool.name}({kwargs})"
        allowed = await asyncio.to_thread(_ask_permission, f"mcp:{mcp_tool.name}", detail, False, None)
        if not allowed:
            return f"PERMISO DENEGADO por el usuario: no se ejecutó la tool MCP '{mcp_tool.name}'."
        return await mcp_tool.ainvoke(kwargs)

    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description,
        args_schema=mcp_tool.args_schema,
        coroutine=gated,
    )


def read_mcp_config() -> dict:
    if not os.path.exists(MCP_CONFIG_FILE):
        return {}
    try:
        with open(MCP_CONFIG_FILE, "r") as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def write_mcp_config(config: dict) -> None:
    with open(MCP_CONFIG_FILE, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def add_mcp_server(name: str, command: str, args: list, env: Optional[dict] = None) -> None:
    config = read_mcp_config()
    entry = {"transport": "stdio", "command": command, "args": args}
    if env:
        entry["env"] = env
    config[name] = entry
    write_mcp_config(config)


def remove_mcp_server(name: str) -> None:
    config = read_mcp_config()
    if name not in config:
        raise ValueError(f"No existe el servidor MCP '{name}'.")
    del config[name]
    write_mcp_config(config)


def load_mcp_tools() -> list:
    """Lee mcp_servers.json (si existe) y conecta a los servidores MCP
    configurados, devolviendo sus tools ya envueltas con permiso. Si el
    archivo no existe o algo falla al conectar, devuelve [] sin impedir que
    kiwi arranque (MCP es opcional)."""
    if not os.path.exists(MCP_CONFIG_FILE):
        return []
    try:
        with open(MCP_CONFIG_FILE, "r") as f:
            config = json.load(f)
        if not config:
            return []
        client = MultiServerMCPClient(config)
        raw_tools = asyncio.run(client.get_tools())
        return [_wrap_mcp_tool_with_permission(t) for t in raw_tools]
    except Exception as exc:  # noqa: BLE001 - no debe impedir que kiwi arranque
        print(f"[kiwi] Aviso: no se pudieron cargar tools MCP desde {MCP_CONFIG_FILE}: {exc}")
        return []


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
    """Escribe contenido en un archivo, creándolo o sobrescribiéndolo entero.
    Para modificar un archivo existente usa mejor edit_file (más seguro, muestra
    un diff en vez de reemplazar todo el contenido).

    Args:
        path: ruta del archivo
        content: contenido a escribir
    """
    resolved = os.path.expanduser(path.strip())
    diff_text = None
    if os.path.exists(resolved):
        try:
            with open(resolved, "r") as f:
                before = f.read()
            diff_text = _make_diff(resolved, before, content)
        except (UnicodeDecodeError, OSError):
            diff_text = None
    detail = resolved if diff_text else f"{resolved} (archivo nuevo, {len(content)} chars)"
    if not _ask_permission("write_file", detail, dangerous=False, diff=diff_text):
        return f"PERMISO DENEGADO por el usuario: no se escribió '{resolved}'."
    with open(resolved, "w") as f:
        f.write(content)
    return f"Archivo guardado en {resolved}"


@tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Reemplaza una única ocurrencia exacta de old_string por new_string en un
    archivo existente. Preferible a write_file para editar: solo cambia lo
    necesario y el usuario ve un diff antes de aprobar. old_string debe incluir
    suficiente contexto (varias líneas si hace falta) para ser único en el archivo.

    Args:
        path: ruta del archivo a editar
        old_string: texto exacto a reemplazar (debe aparecer exactamente una vez)
        new_string: texto de reemplazo
    """
    resolved = os.path.expanduser(path.strip())
    try:
        with open(resolved, "r") as f:
            original = f.read()
    except FileNotFoundError:
        return f"ERROR: '{resolved}' no existe. Usa write_file para crear archivos nuevos."

    count = original.count(old_string)
    if count == 0:
        return (
            "ERROR: old_string no se encontró en el archivo. Revisa el texto exacto "
            "(espacios/indentación) e inclúyelo con más contexto."
        )
    if count > 1:
        return (
            f"ERROR: old_string aparece {count} veces en el archivo; debe ser único. "
            "Añade más contexto (líneas de alrededor) para desambiguar."
        )

    updated = original.replace(old_string, new_string, 1)
    diff_text = _make_diff(resolved, original, updated)

    if not _ask_permission("edit_file", resolved, dangerous=False, diff=diff_text):
        return f"PERMISO DENEGADO por el usuario: no se editó '{resolved}'."

    with open(resolved, "w") as f:
        f.write(updated)
    return f"Archivo editado: {resolved}"


@tool
def run_command(command: str) -> str:
    """Ejecuta cualquier comando bash: mkdir, ls, cp, rm, git, etc. Input: comando bash."""
    if not _ask_permission("run_command", command, dangerous=is_dangerous_command(command)):
        return f"PERMISO DENEGADO por el usuario: no se ejecutó '{command}'."
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


@tool
def load_skill(name: str) -> str:
    """Carga las instrucciones completas de una skill disponible, por su nombre
    (ver la lista de skills disponibles en tus instrucciones de sistema)."""
    skills = load_skills()
    skill = skills.get(name.strip())
    if not skill:
        disponibles = ", ".join(skills) or "(ninguna)"
        return f"ERROR: no existe la skill '{name}'. Skills disponibles: {disponibles}"
    return skill["body"]


TOOLS = [read_file, write_file, edit_file, run_command, search_web, load_skill]


class ToolEventLogger(BaseCallbackHandler):
    """Reenvía el inicio/fin de cada tool call a callbacks externos (usados por la TUI)."""

    def __init__(self, on_tool_start=None, on_tool_end=None):
        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end

    def on_tool_start(self, serialized, input_str, **kwargs):
        if self._on_tool_start:
            name = serialized.get("name", "tool")
            self._on_tool_start(name, input_str)

    def on_tool_end(self, output, **kwargs):
        if self._on_tool_end:
            text = output if isinstance(output, str) else getattr(output, "content", str(output))
            self._on_tool_end(text)


def build_executor(on_tool_start=None, on_tool_end=None, on_permission_request=None) -> AgentExecutor:
    """Construye el AgentExecutor de Kiwi.

    on_tool_start: callback opcional (name: str, input_str: str) -> None,
    invocado cada vez que el agente llama a una tool.
    on_tool_end: callback opcional (output: str) -> None,
    invocado cada vez que una tool devuelve su resultado.
    on_permission_request: callback opcional (action: str, detail: str, dangerous: bool) -> bool,
    invocado antes de write_file/run_command; debe bloquear hasta que el usuario
    apruebe o deniegue la acción. Si no se proporciona, las acciones se permiten
    sin confirmación (compatibilidad con usos no interactivos).
    """
    set_permission_callback(on_permission_request)
    llm = build_llm_from_profile(get_current_profile())

    with open(SYSTEM_PROMPT_FILE, "r") as f:
        system_prompt = f.read()

    system_prompt += working_directory_context()
    system_prompt += skills_summary(load_skills())

    project_config = load_project_config()
    if project_config:
        system_prompt += "\n\n## Instrucciones del proyecto actual\n" + project_config

    mcp_tools = load_mcp_tools()
    all_tools = TOOLS + mcp_tools

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, all_tools, prompt)
    langfuse = LangfuseCallback()

    return AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=False,
        handle_parsing_errors=True,
        callbacks=[ToolEventLogger(on_tool_start, on_tool_end), langfuse],
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


def _estimate_tokens(messages: list) -> int:
    """Heurística barata (chars/4) para no depender de un tokenizer real."""
    return sum(len(m.content) for m in messages) // 4


def summarize_history(messages: list) -> str:
    """Resume una lista de mensajes antiguos en un párrafo conciso."""
    llm = build_llm_from_profile(get_current_profile())
    convo_text = "\n".join(
        f"{'Usuario' if isinstance(m, HumanMessage) else 'Kiwi'}: {m.content}" for m in messages
    )
    prompt = (
        "Resume la siguiente conversación entre un usuario y el asistente Kiwi en un "
        "párrafo breve (máximo 6 líneas). Conserva decisiones tomadas, archivos o rutas "
        "mencionados y cualquier contexto necesario para continuar la conversación con "
        "naturalidad:\n\n" + convo_text
    )
    response = llm.invoke(prompt)
    return response.content


def compact_history(history: list) -> list:
    """Si el historial supera MAX_HISTORY_TOKENS, resume los turnos más antiguos
    en un único par de mensajes y conserva sin tocar los KEEP_RECENT_TURNS más
    recientes. Si no hace falta compactar, devuelve el historial tal cual."""
    if _estimate_tokens(history) <= MAX_HISTORY_TOKENS:
        return history

    keep_from = max(0, len(history) - KEEP_RECENT_TURNS * 2)
    old, recent = history[:keep_from], history[keep_from:]
    if not old:
        return history

    summary = summarize_history(old)
    condensed = [
        HumanMessage(content="(resumen del contexto anterior de esta sesión)"),
        AIMessage(content=summary),
    ]
    return condensed + recent


def new_session_ids() -> tuple[str, str]:
    session_id = str(uuid.uuid4())[:8]
    user_id = socket.gethostname() or "kiwi-user"
    return session_id, user_id
