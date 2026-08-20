# 🥝 Kiwi

Kiwi es un agente de terminal local y privado: un asistente con memoria, tools sobre tu
sistema de archivos, ejecución de comandos y búsqueda web, con una interfaz de texto (TUI)
inspirada en herramientas como Claude Code. Corre en tu máquina, guarda tu conversación
localmente y solo usa herramientas cuando de verdad hace falta.

La TUI abre con el logo `KIWI` en un wordmark degradado (`rich-pyfiglet`, fuente
`ansi_shadow`) — se ve mucho mejor en una terminal real con soporte truecolor que en este
README.

## Qué hace

- **Conversación con memoria**: guarda el historial en `~/.kiwi_history.json` y lo recupera
  entre sesiones.
- **Tools con criterio**: el system prompt le indica al modelo cuándo usar herramientas y
  cuándo responder directamente, para no ejecutar acciones innecesarias.
  - `read_file` — lee archivos reales del disco.
  - `write_file` — crea un archivo o lo reescribe entero.
  - `edit_file` — reemplaza una porción exacta de un archivo existente y muestra un
    diff antes de aplicarlo (más seguro y barato en tokens que reescribir todo).
  - `run_command` — ejecuta comandos bash (`mkdir`, `git`, `ls`, etc.).
  - `search_web` — búsqueda web para información en tiempo real.
- **Observabilidad**: cada sesión se traza en [Langfuse](https://langfuse.com) (tags, tool
  calls, usuario/sesión).
- **Respuesta en streaming**: la respuesta final se muestra token a token a medida que el
  modelo la genera (los pasos intermedios de tool-calling no se muestran como texto, solo
  como líneas de tool call).
- **Compactación de contexto**: cuando el historial supera un presupuesto de tokens
  aproximado, los turnos más antiguos se resumen automáticamente en un solo mensaje en
  vez de descartarse sin más, así la sesión puede alargarse sin perder contexto relevante.
- **Skills**: archivos `.md` en `skills/` con instrucciones que el modelo carga bajo
  demanda (tool `load_skill`) solo cuando la tarea encaja, en vez de cargar todo siempre
  en el system prompt. Trae dos de ejemplo (`conventional-commits`, `code-review-checklist`).
- **Contexto del directorio de trabajo**: al arrancar, Kiwi sabe en qué ruta se está
  ejecutando (`os.getcwd()`) y ve el listado de nivel superior de esa carpeta — así, si
  lanzas `kiwi` desde dentro de otro repo y le hablas de "este proyecto", ya sabe a qué
  ruta te refieres en vez de tener que preguntártela o inventarla.
- **Configuración por proyecto**: si el directorio desde el que lanzas `kiwi` tiene un
  `KIWI.md` o `AGENTS.md`, su contenido se añade automáticamente a las instrucciones del
  agente para esa sesión (reglas del proyecto, convenciones, contexto).
- **Cliente MCP**: si existe `mcp_servers.json` (ver `mcp_servers.json.example`), Kiwi se
  conecta a los servidores [MCP](https://modelcontextprotocol.io) configurados al arrancar
  y añade sus tools al agente — igual que cualquier tool propia, pasan por el mismo
  sistema de permisos antes de ejecutarse. Opcional: sin ese archivo, Kiwi arranca igual.
- **Modos**, como en Claude Code — cambian permisos e instrucciones del agente a la vez:
  - **Ask Mode** (gris, por defecto): pide confirmación para toda acción (`write_file`,
    `edit_file`, `run_command`, tools MCP).
  - **Plan Mode** (celeste): solo lectura. `write_file`/`edit_file` bloqueados, `run_command`
    solo funciona para comandos de solo lectura (`ls`, `cat`, `grep`, `git status/log/diff`,
    etc.) — el resto se deniega sin preguntar. El modelo recibe instrucciones de investigar
    y proponer un plan numerado en vez de ejecutar cambios.
  - **Work Mode** (verde kiwi): `write_file`/`edit_file` se aplican sin pedir confirmación;
    `run_command` y las tools MCP la siguen pidiendo.

  Se cambia con `Shift+Tab` (cicla Ask → Plan → Work) o escribiendo `/ask`, `/plan`, `/work`
  como mensaje. El modo activo se ve siempre en la cabecera de la TUI, y toda decisión
  automática (bloqueo o aprobación sin preguntar) queda registrada en el log para que quede
  rastro de qué pasó y por qué.
- **TUI**: interfaz de terminal construida con [Textual](https://textual.textualize.io/),
  con historial scrollable, indicador de "pensando" y una línea por cada tool call, al
  estilo de Claude Code. El banner usa [`rich-pyfiglet`](https://pypi.org/project/rich-pyfiglet/)
  para el wordmark con degradado de color.

## Arquitectura

```
kiwi.py                 → punto de entrada, lanza la TUI
tui.py                  → interfaz Textual (KiwiApp): layout, widgets, eventos de teclado
agent.py                → LLM, tools, prompt, AgentExecutor y persistencia del historial
kiwi.tcss               → estilos de la TUI
system_prompt.txt       → instrucciones del agente (cuándo usar tools, idioma, tono)
skills/                 → skills cargables bajo demanda (.md con frontmatter name/description)
mcp_servers.json.example → plantilla de configuración de servidores MCP (opcional)
KIWI.md / AGENTS.md     → si existe en el directorio desde el que lanzas kiwi, se añade
                           automáticamente al system prompt para esa sesión (no viene en
                           el repo; es específico de cada proyecto donde uses kiwi)
```

El agente se construye una sola vez con `agent.build_executor()`, que acepta un callback
`on_tool_start` — así la TUI se entera en tiempo real de cada tool que se ejecuta y la
muestra en el log sin acoplar la lógica del agente a la interfaz.

## Instalación

```bash
git clone <este repo>
cd kiwi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Crea un `.env` en la raíz del proyecto con tus credenciales:

```bash
OPENAI_API_KEY=sk-...
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Uso

```bash
./kiwi.py
# o
python3 kiwi.py
```

Escribe tu mensaje y pulsa `Enter`. Para salir: escribe `salir` (o `exit` / `quit` / `q`),
o pulsa `Ctrl+C`. El historial se guarda automáticamente al terminar y tras cada turno.

Kiwi usa el directorio **desde el que lo lanzas** como su directorio de trabajo (no el
directorio donde vive el propio repo de Kiwi). Para usarlo dentro de otro proyecto, entra
a esa carpeta y lánzalo con la ruta completa:

```bash
cd ~/mi-otro-proyecto
/Users/racso/kiwi/kiwi.py
```

O añade un alias en tu `.zshrc` / `.bashrc` para no escribir la ruta cada vez:

```bash
alias kiwi="/Users/racso/kiwi/kiwi.py"
```

## Permisos

Antes de ejecutar `run_command` o `write_file`, Kiwi pide confirmación explícita en la
propia TUI (`y` para permitir, cualquier otra tecla para denegar). Los comandos que
coinciden con patrones potencialmente destructivos (`rm -rf`, `sudo`, `git push --force`,
`git reset --hard`, etc. — ver `DANGEROUS_COMMAND_PATTERNS` en `agent.py`) se marcan con
una advertencia reforzada, pero siguen pidiendo la misma confirmación que cualquier otra
acción. Si el agente se usa fuera de la TUI sin pasar `on_permission_request`, las
acciones se permiten sin confirmación (uso no interactivo).

## MCP

Copia `mcp_servers.json.example` a `mcp_servers.json` (ignorado por git, puede contener
rutas o tokens específicos de tu máquina) y añade los servidores que quieras. Cada tool
que expongan aparece en el agente con el mismo sistema de permisos que `write_file` o
`run_command`: Kiwi no controla qué hace un servidor MCP externo, así que toda llamada
pide confirmación (marcada como "🔌 Tool MCP externa" en el prompt de permiso).

## Roadmap

- Empaquetado para instalación vía `pipx install kiwi` / binario standalone.
- Sandboxing real (confinar `run_command`/`write_file`/`edit_file` al directorio del proyecto).
- Opción de "permitir siempre esta sesión" para no repetir la confirmación en cada llamada.
- Sub-agentes / orquestación para delegar tareas grandes a agentes hijos con su propio contexto.

## Por qué Kiwi

Este proyecto nace como una exploración personal de cómo funcionan agentes de terminal
como Claude Code por dentro: el bucle de tool-calling, la gestión de contexto y memoria,
y la experiencia de usuario en una TUI. Todo corre localmente, con tus propias claves y
sin depender de un producto cerrado.
