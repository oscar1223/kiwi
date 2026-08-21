#!/usr/bin/env python3
"""Widgets modales reutilizables para los comandos interactivos de Kiwi
(/model /config /mcp /skill /memory): un menú navegable con flechas
(OptionMenu) y un cuadro de texto emergente (TextPrompt), en vez de tener que
escribir comandos literales en el chat."""

from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static

ACCENT = "#7CFC00"
ACCENT_DIM = "#4CAF50"


class OptionMenu(ModalScreen[str | None]):
    """Menú modal navegable con flechas/Enter. Devuelve el value de la
    opción elegida, o None si se cancela con Escape."""

    DEFAULT_CSS = f"""
    OptionMenu {{
        align: center middle;
        background: black 40%;
    }}
    OptionMenu > Vertical {{
        width: auto;
        min-width: 46;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        border: round {ACCENT_DIM};
        background: #101010;
        padding: 1 2;
    }}
    OptionMenu .menu-title {{
        padding-bottom: 1;
        text-style: bold;
        color: {ACCENT};
    }}
    OptionMenu OptionList {{
        background: #101010;
        border: none;
        max-height: 16;
    }}
    """
    BINDINGS = [("escape", "cancel", "Cancelar")]

    def __init__(self, title: str, options: list[tuple[str, str]]):
        super().__init__()
        self._title = title
        self._values = [value for _, value in options]
        self._labels = [label for label, _ in options]

    def compose(self):
        with Vertical():
            yield Static(self._title, classes="menu-title")
            yield OptionList(*self._labels)

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self._values[event.option_index])

    def action_cancel(self) -> None:
        self.dismiss(None)


class TextPrompt(ModalScreen[str | None]):
    """Pide una línea de texto en un cuadro modal. Devuelve el texto escrito
    (puede ser ""), o None si se cancela con Escape."""

    DEFAULT_CSS = f"""
    TextPrompt {{
        align: center middle;
        background: black 40%;
    }}
    TextPrompt > Vertical {{
        width: 60;
        max-width: 90%;
        height: auto;
        border: round {ACCENT_DIM};
        background: #101010;
        padding: 1 2;
    }}
    TextPrompt .menu-title {{
        padding-bottom: 1;
        text-style: bold;
        color: {ACCENT};
    }}
    TextPrompt Input {{
        border: round {ACCENT_DIM};
        background: #101010;
    }}
    TextPrompt Input:focus {{
        border: round {ACCENT};
    }}
    """
    BINDINGS = [("escape", "cancel", "Cancelar")]

    def __init__(self, title: str, placeholder: str = "", default: str = ""):
        super().__init__()
        self._title = title
        self._placeholder = placeholder
        self._default = default

    def compose(self):
        with Vertical():
            yield Static(self._title, classes="menu-title")
            yield Input(placeholder=self._placeholder, value=self._default)

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


async def confirm(app, question: str) -> bool:
    """Atajo para un OptionMenu de sí/no. Escape o 'No' cuentan como False."""
    result = await app.push_screen_wait(OptionMenu(question, [("Sí", "y"), ("No", "n")]))
    return result == "y"
