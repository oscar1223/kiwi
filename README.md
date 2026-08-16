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
  - `write_file` — escribe contenido en un archivo.
  - `run_command` — ejecuta comandos bash (`mkdir`, `git`, `ls`, etc.).
  - `search_web` — búsqueda web para información en tiempo real.
- **Observabilidad**: cada sesión se traza en [Langfuse](https://langfuse.com) (tags, tool
  calls, usuario/sesión).
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

## Roadmap

- Permisos/confirmación antes de ejecutar `run_command` o `write_file`.
- Streaming de la respuesta token a token.
- Ediciones de archivo basadas en diffs en vez de sobrescritura completa.
- Empaquetado para instalación vía `pipx install kiwi` / binario standalone.

## Por qué Kiwi

Este proyecto nace como una exploración personal de cómo funcionan agentes de terminal
como Claude Code por dentro: el bucle de tool-calling, la gestión de contexto y memoria,
y la experiencia de usuario en una TUI. Todo corre localmente, con tus propias claves y
sin depender de un producto cerrado.
