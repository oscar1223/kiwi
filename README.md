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
| `work` | aplica ediciones, comandos y herramientas MCP automáticamente; solo pregunta ante comandos peligrosos (rm -rf, sudo, force push, ...) |

Se eligen con `--mode` al arrancar, se cambian en caliente con `/ask` `/plan` `/work`, o
ciclando con `shift+tab`.

## Comandos de la TUI

Escribir `/` muestra y filtra la lista de comandos disponibles según lo que vayas
tecleando (flechas para navegar, `tab` o `enter` para completar).

| Comando      | Qué hace                                            |
|--------------|------------------------------------------------------|
| `/init`      | escribe o actualiza `KIWI.md` leyendo el proyecto antes |
| `/status`    | todo lo de esta sesión en una pantalla                 |
| `/context`   | qué está llenando la ventana de contexto               |
| `/tools`     | lista las tools, o apaga una para esta sesión          |
| `/undo`      | deshace el último turno: ficheros y conversación       |
| `/redo`      | rehace lo que `/undo` quitó                            |
| `/diff`      | qué ha cambiado en la sesión (`/diff turn`, el último) |
| `/review`    | revisa los cambios actuales con un agente aparte       |
| `/settings`  | menú agrupado con todo lo de abajo                    |
| `/model`     | cambia o gestiona perfiles de modelo                  |
| `/config`    | gestiona variables de `.env`                          |
| `/mcp`       | gestiona servidores MCP                               |
| `/skill`     | gestiona skills                                       |
| `/theme`     | cambia el tema de color, con vista previa en vivo      |
| `/sessions`  | cambia entre conversaciones guardadas del proyecto     |
| `/memory`    | ve o edita lo que kiwi recuerda                        |
| `/compact`   | resume la conversación para liberar contexto           |
| `/clear`     | olvida la conversación actual                          |
| `/help`      | lista todos los comandos y atajos                      |
| `/quit`      | sale de kiwi                                           |

## Herramientas del agente

- `read_file` / `write_file` / `edit_file` — lectura y edición del sistema de archivos,
  con diff antes de aplicar.
- `multi_edit` — varias sustituciones sobre un mismo fichero en una llamada. O se
  aplican todas o no se toca el fichero: un lote que fallara a medias dejaría el
  fichero en un estado que nadie pidió.
- `grep` / `glob` / `ls` — búsqueda por contenido, por nombre y listado de un
  directorio. No piden permiso (como `read_file`) y saltan solas `.git`,
  `node_modules` y compañía. Preferibles a llamar a `grep` o `find` por `bash`:
  no dependen de qué binarios haya instalados.
- `todo_write` / `todo_read` — la lista de tareas del trabajo en curso. Es lo que
  mantiene el rumbo en tareas largas, y de paso te enseña por dónde va.
- `web_fetch` — lee una URL y devuelve el texto, con el HTML ya limpio. Rechaza
  direcciones de loopback, privadas y link-local —incluida `169.254.169.254`, la de
  metadatos de las nubes— y lo comprueba al abrir la conexión, así que una
  redirección hacia dentro tampoco cuela. Libre en `plan` y `work`, pregunta en `ask`.
- `bash` — ejecuta comandos; en segundo plano con `background_bash` /
  `background_output` / `kill_shell` para procesos de larga duración.
- `task` — lanza subagentes para investigación o trabajo en paralelo.
- `load_skill` — carga instrucciones bajo demanda desde skills instaladas.
- `remember` — guarda un hecho duradero en la memoria (ver más abajo). Pasa por el
  mismo permiso que una escritura de fichero: en `plan` está bloqueado, en `ask` te
  pregunta, en `work` se guarda solo.
- `ask_questions` — el modelo te hace hasta varias preguntas de opción múltiple (con
  hueco para respuesta libre y, si aplica, selección múltiple) antes de darte un plan
  cerrado, igual que hace Claude Code. Disponible en cualquier modo, pero pensada sobre
  todo para `plan`: mejor preguntar una vez que planear sobre una suposición. Solo se
  registra en la TUI — `kiwi ask` no tiene a quién preguntarle, así que ahí ni aparece.
- Cualquier tool expuesta por un servidor MCP configurado se añade automáticamente,
  detrás del mismo permiso que las tools nativas.

## Memoria y contexto

Kiwi distingue dos cosas que se suelen confundir:

- **La conversación** — grande, propia de la sesión, y se compacta sola cuando crece.
  El umbral sale de la ventana de contexto real del modelo del perfil activo (la mitad
  de la ventana), así que un modelo de 200k no se compacta con el mismo criterio que uno
  de 32k. Cuando salta, los mensajes viejos se sustituyen por un resumen y los recientes
  siguen tal cual. `/compact` fuerza esa compactación cuando quieras, sin esperar al
  umbral. La línea de estado muestra `ctx N%` de la ventana ocupada, y cambia de color
  al 50% y al 80%.
- **Las notas guardadas** — un puñado de líneas que sobreviven a la sesión y viajan en
  cada prompt. Dos ámbitos: `global` (sobre ti, en todos los proyectos) y `project`
  (solo sobre este directorio). Las escribe el modelo con la tool `remember`, o tú desde
  `/memory`. Cada ámbito tiene un tope de caracteres; al pasarse se caen las notas más
  antiguas y kiwi lo dice en vez de olvidar en silencio.

Las notas viven en `~/.config/kiwi/memory/` (`global.md` y `projects/<proyecto>.md`), no
dentro del repositorio: lo que kiwi se anota para sí mismo no debería aparecer nunca en
tu `git status`. Para instrucciones compartidas y versionadas ya está `KIWI.md`/`AGENTS.md`,
que kiwi lee y nunca escribe.

### Adjuntar ficheros con `@`

En la TUI, escribir `@ruta/al/fichero` mete su contenido en el mensaje que recibe el
modelo, sin que tenga que llamar a `read_file`:

```
› ¿por qué falla @internal/agent/agent.go en el test de @internal/agent/agent_test.go?
```

La transcripción sigue mostrando lo que escribiste tú; el contenido se añade por debajo.
Si una ruta no existe, kiwi lo dice en vez de preguntar sobre un fichero que nunca llegó.

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
- `memory/` — notas duraderas: `global.md` y una por proyecto en `projects/`.

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

## Deshacer

Antes de cada turno que puede escribir, kiwi hace una **instantánea del árbol de
trabajo**. `/undo` la restaura —los ficheros *y* la conversación, porque devolver
los ficheros y dejar al modelo convencido de una edición que ya no existe es peor
que no deshacer nada— y `/redo` la vuelve a aplicar. `/diff` enseña lo acumulado
desde que arrancó la sesión.

Las instantáneas viven en un repositorio git **propio de kiwi**, bajo
`~/.config/kiwi/checkpoints/`, apuntado al proyecto con `GIT_WORK_TREE`. Tu
índice, tu stash, tus hooks y tu historial no se tocan en ningún momento: kiwi
nunca nombra tu `.git`. Hace falta que el proyecto sea un repositorio git —no por
el repo sombra, sino porque `.gitignore` es lo que impide que una instantánea se
trague `node_modules`—; donde no lo sea, kiwi lo dice una vez al arrancar y sigue
sin red.

Esto importa sobre todo en **work mode**, que aplica ediciones y comandos sin
preguntar: es la marcha atrás que sustituye a la confirmación que ya no hay.

## Diagnósticos y LSP

Después de cada `write_file`, `edit_file` o `multi_edit`, kiwi pasa **el
verificador del propio proyecto** por el paquete tocado —`go vet` con `go.mod`,
`tsc --noEmit` con `tsconfig.json`, `ruff` con `pyproject.toml`— y añade lo que
encuentre a la misma observación que lee el modelo. Así se entera de que ha roto
la compilación ahí mismo, no tres llamadas después, cuando ya ha construido
encima. Se puede sustituir con `KIWI_DIAGNOSTICS_CMD` (un `{}` en el comando
marca dónde va la ruta; si no lo hay, se añade al final).

Dos reglas lo mantienen fuera del terreno del ruido: solo se reportan problemas
**en los ficheros que se acaban de editar**, y **no se repite lo ya dicho**. Un
canal de diagnósticos que grita en cada edición se acaba ignorando entero.

Si hay un **language server** instalado y el proyecto es de ese lenguaje —`gopls`,
`typescript-language-server`, `pyright`, `rust-analyzer`—, kiwi registra además la
tool `lsp` con tres operaciones: `definition`, `references` y `diagnostics`. El
servidor arranca la primera vez que se pregunta, no al abrir kiwi, porque indexar
un módulo entero es caro y la mayoría de las sesiones no lo necesitan.

Justifica su sitio en la única pregunta que `grep` no puede responder con
honestidad: **cuál de estas coincidencias es el mismo símbolo**. `grep` encuentra
un nombre; el servidor lo resuelve. Se configura con `KIWI_LSP`
(`.go=gopls;.rb=solargraph --stdio`), y si no hay servidor la tool no se registra
—una herramienta que el modelo ve pero que solo puede fallar es peor que ninguna.

## Skills

kiwi trae **seis skills instaladas de fábrica**, embebidas en el binario y
sembradas en `~/.config/kiwi/skills/` la primera vez que arranca — sin eso, un
`go install` recién hecho dejaría el directorio vacío y "prefabricadas" no
querría decir nada.

| Skill | Para qué |
| --- | --- |
| `plan` | escribe un `PLAN-*.md`: contexto, decisiones con su porqué, partes ancladas a `fichero.go:línea`, orden y verificación |
| `commit` | mensaje desde el diff staged, comprobando primero **en qué repositorio estás** |
| `code-review` | revisa el cambio: primero corrección, luego reutilización y simplificación |
| `init` | escribe `KIWI.md` leyendo el proyecto antes |
| `security-review` | inyección, secretos, autorización, SSRF, travesía de rutas, defaults inseguros |
| `test` | encuentra los comandos reales de *este* repo, los corre y lee los fallos |

**Nunca se pisa lo que hayas tocado.** Una skill solo se refresca si lo que hay
en disco es byte a byte la versión que kiwi escribió por última vez, que es
justo el caso en el que no se puede perder nada. Y si borras una, no vuelve:
borrarla también es una decisión.

### Skills como comandos

Una skill con `user-invocable: true` en su frontmatter **se convierte en un slash
command**: aparece en el autocompletado y en `/help`, y `/plan lo que sea` lanza
un turno con sus instrucciones y con tu texto añadido. Los comandos nativos de
kiwi siempre ganan — una skill llamada `clear` no puede secuestrar `/clear`.

## Teclado

| Tecla | Qué hace |
| --- | --- |
| `↑` `↓` | recupera un prompt anterior (con el input vacío o a medias; un borrador multilínea se queda sus flechas) |
| `ctrl+r` | busca entre los prompts anteriores |
| `@` | selector difuso de ficheros — `@tumod` encuentra `internal/tui/model.go` |
| `!comando` | ejecuta en la shell sin gastar un turno |
| `ctrl+t` | muestra u oculta la lista de tareas del modelo |
| `ctrl+e` | redacta el prompt en `$EDITOR` |
| `?` | panel de atajos, con el input vacío |
| `shift+tab` | cicla de modo |
| `esc` | descarta la cola, o cancela el turno |

El **historial es por directorio de trabajo** (`~/.local/share/kiwi/history/`):
los prompts de un proyecto son ruido en otro.

**Escribir durante un turno encola** en vez de perderse. Todo lo encolado se
manda junto al terminar, como un solo mensaje: tres cosas que se te ocurren
viendo el mismo trabajo son una sola pieza de feedback, no tres turnos. `esc` la
descarta. Si el turno falla, la cola **no** se envía —estaba escrita sobre un
trabajo que no llegó a pasar, y esa decisión es tuya.

Al acabar un turno de más de 20 segundos **suena la campana del terminal**. Con
work mode desatendido eso deja de ser adorno: es cómo sabes que puedes volver.
`KIWI_NOTIFY=desktop` añade notificación de escritorio; `KIWI_NOTIFY=off` calla
del todo.

`!` respeta el modo: en plan mode un comando que escriba se rechaza, igual que
haría la tool. Un modo de solo lectura del que te puedas escapar tecleando `!`
no es una promesa que merezca la pena hacer.
