#!/usr/bin/env python3
"""Interfaz de terminal (TUI) de Kiwi, construida con Textual."""

import asyncio
import threading
import time

from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Group
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, RichLog, Static

from agent import (
    MODE_INSTRUCTIONS,
    MODE_LABELS,
    MODE_ORDER,
    MODEL_NAME,
    TOOLS,
    Mode,
    build_executor,
    compact_history,
    load_history,
    load_skills,
    new_session_ids,
    resolve_permission_policy,
    save_history,
)

ACCENT = "#7CFC00"
ACCENT_DIM = "#4CAF50"

INPUT_HINT = (
    "[dim]Enter para enviar · Esc para cancelar · Shift+Tab modo · "
    "/ask /plan /work · Ctrl+C para salir[/dim]"
)

EXIT_WORDS = {"salir", "exit", "quit", "q"}

MODE_COMMANDS = {
    "/ask": Mode.ASK,
    "/plan": Mode.PLAN,
    "/work": Mode.WORK,
    # alias por compatibilidad con el nombre anterior del modo
    "/auto-edit": Mode.WORK,
    "/edits": Mode.WORK,
    "/normal": Mode.ASK,
}

CELESTE = "#87CEFA"

MODE_STYLES = {
    Mode.ASK: "dim",
    Mode.PLAN: f"bold {CELESTE}",
    Mode.WORK: f"bold {ACCENT}",
}

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# El system prompt le pide al modelo que separe su razonamiento (opcional) de la
# respuesta final con estas tags, para poder renderizarlos distinto en vez de
# mezclarlos en un único bloque de texto.
THINK_OPEN, THINK_CLOSE = "<thinking>", "</thinking>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"
_ALL_TAGS = (THINK_OPEN, THINK_CLOSE, ANSWER_OPEN, ANSWER_CLOSE)


def strip_tags(text: str) -> str:
    for tag in _ALL_TAGS:
        text = text.replace(tag, "")
    return text.strip()


def split_thinking_answer(text: str) -> tuple[str, str]:
    """Separa <thinking>...</thinking> de <answer>...</answer> en la salida del
    modelo. Tolera texto a medio transmitir (streaming) y respuestas que no usan
    las tags (fallback: todo el texto se trata como respuesta)."""
    thinking = ""
    rest = text
    if THINK_OPEN in rest:
        after_open = rest.split(THINK_OPEN, 1)[1]
        if THINK_CLOSE in after_open:
            thinking, rest = after_open.split(THINK_CLOSE, 1)
            thinking = thinking.strip()
        else:
            return after_open.strip(), ""
    if ANSWER_OPEN in rest:
        rest = rest.split(ANSWER_OPEN, 1)[1]
    return thinking, rest.replace(ANSWER_CLOSE, "").strip()


def visible_answer(text: str) -> str:
    """La respuesta final "limpia" (sin tags ni razonamiento), para guardar en
    el historial de conversación."""
    _, answer = split_thinking_answer(text)
    return answer or strip_tags(text)


class KiwiApp(App):
    """TUI principal de Kiwi."""

    CSS_PATH = "kiwi.tcss"
    TITLE = "Kiwi"
    BINDINGS = [
        ("ctrl+c", "quit", "Salir"),
        ("escape", "cancel", "Cancelar"),
        # priority=True: si no, Textual consume shift+tab para mover el foco
        # (focus_previous) antes de que llegue a nuestra acción.
        Binding("shift+tab", "cycle_mode", "Modo", priority=True),
    ]

    def __init__(self) -> None:
        # ansi_color=True desactiva el filtro ANSIToTruecolor de Textual, que si
        # no reescribe cualquier color "ansi_default" (usado en kiwi.tcss para el
        # fondo) a un RGB casi negro fijo del tema interno, en vez de dejar que
        # el terminal use su propio color de fondo real.
        super().__init__(ansi_color=True)
        self.chat_history = load_history()
        self.session_id, self.user_id = new_session_ids()
        self.mode = Mode.ASK
        self.executor = build_executor(
            on_tool_start=self._on_tool_start,
            on_tool_end=self._on_tool_end,
            on_permission_request=self._request_permission,
        )
        self._busy = False
        self._generation = 0
        self._active_gen = 0
        self._busy_start = 0.0
        self._spinner_index = 0
        self._status_timer = None
        self._awaiting_permission = None  # (threading.Event, dict) mientras se espera y/N

    def _header_text(self) -> str:
        style = MODE_STYLES.get(self.mode, "dim")
        label = MODE_LABELS.get(self.mode, self.mode)
        return (
            f"[bold {ACCENT}]🥝 Kiwi[/bold {ACCENT}] [dim]· {MODEL_NAME} · sesión "
            f"{self.session_id}[/dim] · [{style}]{label}[/{style}]"
        )

    def compose(self) -> ComposeResult:
        yield Static(self._header_text(), id="header")
        yield RichLog(id="chat-log", wrap=True, markup=True, highlight=False)
        yield Static(id="stream-view", markup=False)
        yield Static(id="status-bar")
        with Vertical(id="input-bar"):
            yield Input(placeholder="Escribe un mensaje a Kiwi...", id="prompt-input")
            yield Static(INPUT_HINT, id="input-hint")

    def on_mount(self) -> None:
        self.query_one("#prompt-input", Input).focus()
        if self.chat_history:
            log = self.query_one("#chat-log", RichLog)
            log.write(Text(f"· {len(self.chat_history) // 2} turnos previos cargados", style="dim italic"))

        skills = load_skills()
        if skills:
            self.write_system(f"Skills cargadas: {', '.join(skills)}")

        core_names = {t.name for t in TOOLS}
        mcp_names = [t.name for t in self.executor.tools if t.name not in core_names]
        if mcp_names:
            self.write_system(f"Tools MCP conectadas: {', '.join(mcp_names)}")

        self.write_system(
            f"Modo: {MODE_LABELS[self.mode]} — Shift+Tab o /ask /plan /work para cambiar"
        )

    # --- helpers de UI ---

    def _bullet(self, marker: str, marker_style: str, content) -> Table:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(no_wrap=True)
        grid.add_column(ratio=1)
        grid.add_row(Text(marker, style=marker_style), content)
        return grid

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        prompt = self.query_one("#prompt-input", Input)
        status = self.query_one("#status-bar", Static)
        prompt.disabled = busy
        if busy:
            self._busy_start = time.monotonic()
            self._spinner_index = 0
            status.set_class(True, "-visible")
            self._status_timer = self.set_interval(1 / 10, self._tick_status)
            self._tick_status()
        else:
            if self._status_timer is not None:
                self._status_timer.stop()
                self._status_timer = None
            status.set_class(False, "-visible")
            status.update("")
            prompt.focus()

    def _tick_status(self) -> None:
        status = self.query_one("#status-bar", Static)
        frame = SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)]
        self._spinner_index += 1
        elapsed = int(time.monotonic() - self._busy_start)
        status.update(f"[dim]{frame} Kiwi está pensando… ({elapsed}s · esc para interrumpir)[/dim]")

    def write_user(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("")
        log.write(self._bullet(">", "bold #a6a6a6", Text(text)))

    def write_kiwi(self, text: str) -> None:
        thinking, answer = split_thinking_answer(text)
        log = self.query_one("#chat-log", RichLog)
        log.write("")
        if thinking:
            log.write(self._bullet("🧠", "dim italic", Text(thinking, style="dim italic")))
            log.write("")
        log.write(self._bullet("⏺", f"bold {ACCENT}", Markdown(answer or strip_tags(text))))

    def write_tool_call(self, name: str, input_str: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        preview = input_str.strip().replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:80] + "…"
        log.write("")
        log.write(self._bullet("⏺", f"dim {ACCENT_DIM}", Text(f"{name}({preview})", style="dim")))

    def write_tool_result(self, output: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        preview = output.strip().replace("\n", " ")
        if len(preview) > 100:
            preview = preview[:100] + "…"
        log.write(Text(f"  ⎿  {preview}", style="dim"))

    def write_error(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("")
        log.write(self._bullet("⏺", "bold red", Text(text, style="red")))

    def write_system(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(Text(f"· {text}", style="dim italic"))

    def _on_stream_token(self, partial_text: str) -> None:
        thinking, answer = split_thinking_answer(partial_text)
        parts = []
        if thinking:
            parts.append(self._bullet("🧠", "dim italic", Text(thinking, style="dim italic")))
        if answer:
            parts.append(self._bullet("⏺", f"bold {ACCENT}", Markdown(answer)))
        if not parts:
            return
        stream_view = self.query_one("#stream-view", Static)
        stream_view.set_class(True, "-visible")
        stream_view.update(Group(*parts))

    def _clear_stream_view(self) -> None:
        stream_view = self.query_one("#stream-view", Static)
        stream_view.set_class(False, "-visible")
        stream_view.update("")

    # --- agente ---

    def _on_tool_start(self, name: str, input_str: str) -> None:
        if self._active_gen != self._generation:
            return
        self.call_from_thread(self.write_tool_call, name, input_str)

    def _on_tool_end(self, output_str: str) -> None:
        if self._active_gen != self._generation:
            return
        self.call_from_thread(self.write_tool_result, output_str)

    def _request_permission(self, action: str, detail: str, dangerous: bool, diff: str | None = None) -> bool:
        """Se invoca desde el hilo de la tool (agent.py). Si el modo actual
        decide automáticamente (Plan/Auto-accept ediciones/Auto), no interrumpe
        al usuario. Si no, bloquea hasta que responda y/N en la TUI."""
        if self._active_gen != self._generation:
            return False

        decision = resolve_permission_policy(self.mode, action, detail, dangerous)
        if decision is not None:
            self.call_from_thread(self._log_auto_permission, action, detail, dangerous, decision)
            return decision

        event = threading.Event()
        box = {"answer": False}
        self.call_from_thread(self._show_permission_prompt, action, detail, dangerous, diff, event, box)
        event.wait()
        return box["answer"]

    def _log_auto_permission(self, action: str, detail: str, dangerous: bool, decision: bool) -> None:
        """Deja constancia en el log cuando el modo actual bloquea una acción
        automáticamente, sin interrumpir al usuario. Las auto-aprobaciones
        (p. ej. escrituras en Work Mode) no se registran: no son peligrosas."""
        if decision:
            return
        preview = detail.strip().replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:120] + "…"
        log = self.query_one("#chat-log", RichLog)
        log.write(Text(f"  🚫 bloqueado por {MODE_LABELS[self.mode]}: {action}({preview})", style="dim red"))

    def _show_permission_prompt(
        self,
        action: str,
        detail: str,
        dangerous: bool,
        diff: str | None,
        event: threading.Event,
        box: dict,
    ) -> None:
        log = self.query_one("#chat-log", RichLog)
        preview = detail.strip().replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "…"
        if dangerous:
            label, style = "⚠ COMANDO PELIGROSO", "bold red"
        elif action.startswith("mcp:"):
            label, style = "🔌 Tool MCP externa", "bold yellow"
        else:
            label, style = "Permiso requerido", "bold yellow"
        log.write("")
        log.write(self._bullet("?", style, Text(f"{label}: {action}({preview})", style=style)))
        if diff:
            log.write(Syntax(diff, "diff", theme="ansi_dark", word_wrap=True, background_color="default"))
        log.write(Text("  Escribe 'y' para permitir, cualquier otra cosa para denegar.", style="dim"))
        self._awaiting_permission = (event, box)
        prompt = self.query_one("#prompt-input", Input)
        prompt.disabled = False
        prompt.placeholder = "y = permitir · cualquier otra tecla = denegar"
        prompt.focus()

    def _resolve_permission(self, text: str) -> None:
        event, box = self._awaiting_permission
        self._awaiting_permission = None
        answer = text.strip().lower() in {"y", "yes", "s", "si", "sí"}
        box["answer"] = answer
        prompt = self.query_one("#prompt-input", Input)
        prompt.disabled = self._busy
        prompt.placeholder = "Escribe un mensaje a Kiwi..."
        log = self.query_one("#chat-log", RichLog)
        log.write(Text(f"  → {'permitido' if answer else 'denegado'}", style="dim italic"))
        event.set()

    async def _stream_agent(self, text: str) -> str:
        """Consume astream_events del AgentExecutor y va renderizando la
        respuesta final token a token, ignorando los chunks de tool-calling
        (que llegan con content vacío y tool_call_chunks poblado)."""
        buffer: list[str] = []
        mode_instruction = MODE_INSTRUCTIONS.get(self.mode, "")
        effective_input = f"{mode_instruction}\n\n{text}" if mode_instruction else text
        async for event in self.executor.astream_events(
            {"input": effective_input, "chat_history": self.chat_history},
            version="v2",
            config={
                "metadata": {
                    "langfuse_user_id": self.user_id,
                    "langfuse_session_id": self.session_id,
                    "langfuse_tags": ["kiwi-agent", "tui"],
                }
            },
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content and not getattr(chunk, "tool_call_chunks", None):
                    buffer.append(chunk.content)
                    self._on_stream_token("".join(buffer))
        return "".join(buffer)

    @work(exclusive=True)
    async def send_message(self, text: str) -> None:
        self._generation += 1
        gen = self._generation
        self._active_gen = gen
        self.set_busy(True)
        try:
            answer = await self._stream_agent(text)
        except Exception as exc:  # noqa: BLE001 - se muestra al usuario, no se silencia
            if gen == self._generation:
                self._clear_stream_view()
                self.write_error(str(exc))
        else:
            if gen == self._generation:
                self._clear_stream_view()
                self.write_kiwi(answer)
                self.chat_history.append(HumanMessage(content=text))
                self.chat_history.append(AIMessage(content=visible_answer(answer)))
                turns_before = len(self.chat_history)
                self.chat_history = await asyncio.to_thread(compact_history, self.chat_history)
                if len(self.chat_history) < turns_before:
                    self.write_system("Contexto comprimido: se resumieron los turnos más antiguos.")
                save_history(self.chat_history)
        finally:
            if gen == self._generation:
                self.set_busy(False)

    # --- eventos ---

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if self._awaiting_permission is not None:
            self._resolve_permission(text)
            return
        if not text or self._busy:
            return
        lowered = text.lower()
        if lowered in MODE_COMMANDS:
            self._set_mode(MODE_COMMANDS[lowered])
            return
        if lowered == "/mode":
            self.write_system(f"Modo actual: {MODE_LABELS[self.mode]}")
            return
        if lowered in EXIT_WORDS:
            self.action_quit()
            return
        self.write_user(text)
        self.send_message(text)

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self.query_one("#header", Static).update(self._header_text())
        self.write_system(f"Modo: {MODE_LABELS[self.mode]}")

    def action_cycle_mode(self) -> None:
        idx = MODE_ORDER.index(self.mode)
        self._set_mode(MODE_ORDER[(idx + 1) % len(MODE_ORDER)])

    def action_cancel(self) -> None:
        if self._awaiting_permission is not None:
            self._resolve_permission("n")
            return
        if self._busy:
            self._generation += 1
            self.set_busy(False)
            self.write_system("Cancelado.")

    def action_quit(self) -> None:
        save_history(self.chat_history)
        self.exit()


if __name__ == "__main__":
    KiwiApp().run()
