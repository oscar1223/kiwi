# 🥝 Kiwi

Kiwi es un agente de código local-first para la terminal: memoria de conversación, tools
sobre tu sistema de archivos, ejecución de comandos y una TUI inspirada en herramientas
como Claude Code. Corre en tu máquina, guarda cada sesión localmente en SQLite y solo actúa
sobre tu proyecto con el permiso que tú le des.

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
  - `sonnet` y `opus` (Anthropic, `ANTHROPIC_API_KEY`), `gpt` (OpenAI, `OPENAI_API_KEY`),
    `local` (cualquier servidor compatible con la API de OpenAI, como Ollama, en
    `http://localhost:11434/v1`).
  - Cualquier proveedor que hable el formato chat-completions de OpenAI funciona solo
    cambiando `base_url` y la env var de la key — ya vienen listos `openrouter`
    (`OPENROUTER_API_KEY`), `groq` (`GROQ_API_KEY`), `deepseek` (`DEEPSEEK_API_KEY`),
    `mistral` (`MISTRAL_API_KEY`), `xai` (`XAI_API_KEY`), `cerebras`
    (`CEREBRAS_API_KEY`), `perplexity` (`PERPLEXITY_API_KEY`), `together`
    (`TOGETHER_API_KEY`), `fireworks` (`FIREWORKS_API_KEY`), `deepinfra`
    (`DEEPINFRA_API_KEY`), `moonshot` (`MOONSHOT_API_KEY`), `zhipu` (`ZHIPU_API_KEY`) y
    `gemini` (`GEMINI_API_KEY`, vía el endpoint de Gemini compatible con OpenAI). Los
    modelos de estos perfiles son ejemplos — ajústalos con `/model` según lo que tenga
    cada proveedor en cada momento.
  - Tres perfiles más necesitan credenciales de nube, no solo una API key — no son
    zero-config como los de arriba:
    - `azure` — Azure OpenAI. Necesita `AZURE_RESOURCE_NAME` y `AZURE_API_KEY`; el
      `model` del perfil es el nombre del *deployment*, no el modelo en sí.
    - `vertex` — modelos Claude en Google Vertex AI. Necesita `GOOGLE_VERTEX_PROJECT` y
      `GOOGLE_VERTEX_LOCATION`, y credenciales de aplicación por defecto (ADC) — normalmente
      vía `GOOGLE_APPLICATION_CREDENTIALS` apuntando a una service account, o `gcloud auth
      application-default login`.
    - `bedrock` — modelos Claude en AWS Bedrock. Usa la cadena de credenciales estándar de
      AWS (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_REGION`, un perfil compartido, o
      `AWS_BEARER_TOKEN_BEDROCK`).
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
