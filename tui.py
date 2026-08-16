#!/usr/bin/env python3
"""Interfaz de terminal (TUI) de Kiwi, construida con Textual."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from rich import box
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich_pyfiglet import RichFiglet
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Input, RichLog, Static

from agent import build_executor, load_history, new_session_ids, save_history, trim_history

KIWI_GRADIENT = ["#DCE775", "#7CFC00", "#2E7D32"]

HELP_TEXT = "[dim]Escribe tu mensaje y pulsa Enter · [bold]salir[/bold] / [bold]ctrl+c[/bold] para terminar[/dim]"

EXIT_WORDS = {"salir", "exit", "quit", "q"}


class KiwiApp(App):
    """TUI principal de Kiwi."""

    CSS_PATH = "kiwi.tcss"
    TITLE = "Kiwi"
    BINDINGS = [("ctrl+c", "quit", "Salir")]

    def __init__(self) -> None:
        super().__init__()
        self.chat_history = load_history()
        self.session_id, self.user_id = new_session_ids()
        self.executor = build_executor(on_tool_start=self._on_tool_start)
        self._busy = False

    def compose(self) -> ComposeResult:
        with Vertical(id="banner"):
            yield Static(
                RichFiglet(
                    "KIWI",
                    font="ansi_shadow",
                    colors=KIWI_GRADIENT,
                    horizontal=True,
                    justify="center",
                ),
                id="kiwi-wordmark",
            )
            yield Static(HELP_TEXT, id="banner-help")
        yield RichLog(id="chat-log", wrap=True, markup=True, highlight=False)
        yield Static("🥝 Kiwi está pensando...", id="status-bar")
        with Vertical(id="input-bar"):
            yield Input(placeholder="Escribe un mensaje a Kiwi...", id="prompt-input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt-input", Input).focus()
        if self.chat_history:
            log = self.query_one("#chat-log", RichLog)
            log.write(Text(f"— {len(self.chat_history) // 2} turnos previos cargados —", style="dim italic"))

    # --- helpers de UI ---

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        status = self.query_one("#status-bar", Static)
        prompt = self.query_one("#prompt-input", Input)
        status.set_class(busy, "-visible")
        prompt.disabled = busy
        if not busy:
            prompt.focus()

    def write_user(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(Panel(Text(text), title="Tú", title_align="left", border_style="cyan", box=box.ROUNDED))

    def write_kiwi(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(
            Panel(
                Markdown(text),
                title="🥝 Kiwi",
                title_align="left",
                border_style="green",
                box=box.ROUNDED,
            )
        )

    def write_tool_call(self, name: str, input_str: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        preview = input_str.strip().replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:80] + "…"
        log.write(Text(f"  → {name}({preview})", style="dim yellow"))

    def write_error(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(Panel(Text(text, style="red"), title="Error", title_align="left", border_style="red", box=box.ROUNDED))

    # --- agente ---

    def _on_tool_start(self, name: str, input_str: str) -> None:
        self.call_from_thread(self.write_tool_call, name, input_str)

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
        self.set_busy(True)
        try:
            answer = await asyncio.to_thread(self._invoke_agent, text)
        except Exception as exc:  # noqa: BLE001 - se muestra al usuario, no se silencia
            self.write_error(str(exc))
        else:
            self.write_kiwi(answer)
            self.chat_history.append(HumanMessage(content=text))
            self.chat_history.append(AIMessage(content=answer))
            self.chat_history = trim_history(self.chat_history)
            save_history(self.chat_history)
        finally:
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

    def action_quit(self) -> None:
        save_history(self.chat_history)
        self.exit()


if __name__ == "__main__":
    KiwiApp().run()
