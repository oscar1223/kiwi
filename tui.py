#!/usr/bin/env python3
"""Interfaz de terminal (TUI) de Kiwi, construida con Textual."""

import asyncio
import time

from langchain_core.messages import AIMessage, HumanMessage
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, RichLog, Static

from agent import MODEL_NAME, build_executor, load_history, new_session_ids, save_history, trim_history

ACCENT = "#7CFC00"
ACCENT_DIM = "#4CAF50"

INPUT_HINT = "[dim]Enter para enviar · Esc para cancelar · Ctrl+C para salir[/dim]"

EXIT_WORDS = {"salir", "exit", "quit", "q"}

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class KiwiApp(App):
    """TUI principal de Kiwi."""

    CSS_PATH = "kiwi.tcss"
    TITLE = "Kiwi"
    BINDINGS = [("ctrl+c", "quit", "Salir"), ("escape", "cancel", "Cancelar")]

    def __init__(self) -> None:
        super().__init__()
        self.chat_history = load_history()
        self.session_id, self.user_id = new_session_ids()
        self.executor = build_executor(on_tool_start=self._on_tool_start, on_tool_end=self._on_tool_end)
        self._busy = False
        self._generation = 0
        self._active_gen = 0
        self._busy_start = 0.0
        self._spinner_index = 0
        self._status_timer = None

    def compose(self) -> ComposeResult:
        yield Static(
            f"[bold {ACCENT}]🥝 Kiwi[/bold {ACCENT}] [dim]· {MODEL_NAME} · sesión {self.session_id}[/dim]",
            id="header",
        )
        yield RichLog(id="chat-log", wrap=True, markup=True, highlight=False)
        yield Static(id="status-bar")
        with Vertical(id="input-bar"):
            yield Input(placeholder="Escribe un mensaje a Kiwi...", id="prompt-input")
            yield Static(INPUT_HINT, id="input-hint")

    def on_mount(self) -> None:
        self.query_one("#prompt-input", Input).focus()
        if self.chat_history:
            log = self.query_one("#chat-log", RichLog)
            log.write(Text(f"· {len(self.chat_history) // 2} turnos previos cargados", style="dim italic"))

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
        log = self.query_one("#chat-log", RichLog)
        log.write("")
        log.write(self._bullet("⏺", f"bold {ACCENT}", Markdown(text)))

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

    # --- agente ---

    def _on_tool_start(self, name: str, input_str: str) -> None:
        if self._active_gen != self._generation:
            return
        self.call_from_thread(self.write_tool_call, name, input_str)

    def _on_tool_end(self, output_str: str) -> None:
        if self._active_gen != self._generation:
            return
        self.call_from_thread(self.write_tool_result, output_str)

    def _invoke_agent(self, text: str) -> str:
        response = self.executor.invoke(
            {"input": text, "chat_history": self.chat_history},
            config={
                "metadata": {
                    "langfuse_user_id": self.user_id,
                    "langfuse_session_id": self.session_id,
                    "langfuse_tags": ["kiwi-agent", "tui"],
                }
            },
        )
        return response["output"]

    @work(exclusive=True)
    async def send_message(self, text: str) -> None:
        self._generation += 1
        gen = self._generation
        self._active_gen = gen
        self.set_busy(True)
        try:
            answer = await asyncio.to_thread(self._invoke_agent, text)
        except Exception as exc:  # noqa: BLE001 - se muestra al usuario, no se silencia
            if gen == self._generation:
                self.write_error(str(exc))
        else:
            if gen == self._generation:
                self.write_kiwi(answer)
                self.chat_history.append(HumanMessage(content=text))
                self.chat_history.append(AIMessage(content=answer))
                self.chat_history = trim_history(self.chat_history)
                save_history(self.chat_history)
        finally:
            if gen == self._generation:
                self.set_busy(False)

    # --- eventos ---

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text or self._busy:
            return
        if text.lower() in EXIT_WORDS:
            self.action_quit()
            return
        self.write_user(text)
        self.send_message(text)

    def action_cancel(self) -> None:
        if self._busy:
            self._generation += 1
            self.set_busy(False)
            self.write_system("Cancelado.")

    def action_quit(self) -> None:
        save_history(self.chat_history)
        self.exit()


if __name__ == "__main__":
    KiwiApp().run()
