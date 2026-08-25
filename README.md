# 🥝 Kiwi

Kiwi es un agente de código local-first para la terminal: memoria de conversación, tools
sobre tu sistema de archivos, ejecución de comandos y una TUI inspirada en herramientas
como Claude Code. Corre en tu máquina, guarda cada sesión localmente en SQLite y solo actúa
sobre tu proyecto con el permiso que tú le des.

Esta es la reescritura en Go del [prototipo original en Python](prototype/README.md).

## Instalación

```sh
go install github.com/oscar1223/kiwi/cmd/kiwi@latest
```

O compilando desde el repo:

```sh
go build -o kiwi ./cmd/kiwi
```

## Primer uso

```sh
kiwi
```

Sin ningún proveedor configurado, la primera vez que arrancas `kiwi` un asistente de
onboarding te pide el proveedor (Anthropic o cualquier API compatible con OpenAI — OpenAI,
Ollama, OpenRouter, Groq...), el modelo y la API key, y lo deja todo listo para chatear.
Se puede repetir o ajustar luego con `/settings` → `Model profiles`, o directamente
`/model`.

## Modos de permiso

Kiwi opera siempre en uno de tres modos, que determinan qué puede hacer sin pedirte
confirmación:

| Modo   | Qué permite                                  |
|--------|-----------------------------------------------|
| `ask`  | confirma cada acción antes de ejecutarla (por defecto) |
| `plan` | solo lectura — no puede editar ni ejecutar comandos |
| `work` | aplica ediciones automáticamente, pero sigue preguntando antes de comandos |

Se eligen con `--mode` al arrancar, se cambian en caliente con `/ask` `/plan` `/work`, o
ciclando con `shift+tab`.

## Comandos de la TUI

Escribir `/` muestra y filtra la lista de comandos disponibles según lo que vayas
tecleando (flechas para navegar, `tab` o `enter` para completar).

| Comando      | Qué hace                                            |
|--------------|------------------------------------------------------|
| `/settings`  | menú agrupado con todo lo de abajo                    |
| `/model`     | cambia o gestiona perfiles de modelo                  |
| `/config`    | gestiona variables de `.env`                          |
| `/mcp`       | gestiona servidores MCP                               |
| `/skill`     | gestiona skills                                       |
| `/theme`     | cambia el tema de color, con vista previa en vivo      |
| `/sessions`  | cambia entre conversaciones guardadas del proyecto     |
| `/memory`    | ve o borra la memoria de la conversación               |
| `/clear`     | olvida la conversación actual                          |
| `/help`      | lista todos los comandos y atajos                      |
| `/quit`      | sale de kiwi                                           |

## Herramientas del agente

- `read_file` / `write_file` / `edit_file` — lectura y edición del sistema de archivos,
  con diff antes de aplicar.
- `bash` — ejecuta comandos; en segundo plano con `background_bash` /
  `background_output` / `kill_shell` para procesos de larga duración.
- `task` — lanza subagentes para investigación o trabajo en paralelo.
- `load_skill` — carga instrucciones bajo demanda desde skills instaladas.
- Cualquier tool expuesta por un servidor MCP configurado se añade automáticamente,
  detrás del mismo permiso que las tools nativas.

## Sesiones

Cada conversación se persiste al vuelo en SQLite, por proyecto:

```sh
kiwi --continue          # retoma la sesión más reciente de este directorio
kiwi --resume <id>       # retoma una sesión concreta
kiwi session list        # lista las sesiones guardadas para este proyecto
```

Dentro de la TUI, `/sessions` hace lo mismo sin salir de la conversación.

## `kiwi ask`

Para un solo turno no interactivo, pensado para componer con pipes:

```sh
kiwi ask "¿qué hace este repo?"
git diff | kiwi ask "revisa este patch" -
```

Por defecto es de solo lectura; `--mode work` permite editar, y `--yolo` además ejecuta
comandos sin pedir confirmación.

## Configuración

Todo vive bajo `~/.config/kiwi/` (u `$XDG_CONFIG_HOME/kiwi/`):

- `kiwi.json` — perfiles de modelo y perfil activo. Perfiles de ejemplo ya incluidos:
  `sonnet` y `opus` (Anthropic), `gpt` (OpenAI), `local` (cualquier servidor compatible
  con la API de OpenAI, como Ollama, en `http://localhost:11434/v1`).
- `.env` — API keys y demás variables, gestionadas con `/config`.
- `mcp.json` — servidores MCP, por stdio o remotos (HTTP/SSE).
- `skills/` — skills en Markdown que el modelo carga bajo demanda.

## Telemetría (opcional)

Kiwi traza cada sesión vía OpenTelemetry si detecta configuración:

- Variables `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` — cualquier
  backend OTel estándar.
- O `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` (y opcionalmente `LANGFUSE_HOST` para
  una instancia propia) para [Langfuse](https://langfuse.com).

Sin ninguna de las dos, Kiwi no traza nada.

## Desarrollo

```sh
go build ./...
go vet ./...
gofmt -l cmd internal
go test ./... -race
```

## Licencia

MIT — ver [LICENSE](LICENSE).
