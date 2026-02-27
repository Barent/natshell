"""Custom Textual widgets for the NatShell TUI."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from natshell.inference.engine import ToolCall


# ─── Logo frames ─────────────────────────────────────────────────────────────
# Tree art based on the project SVG: three foliage tiers (cyan/teal/green),
# gold star, green sparkle dots, with a neon sign box for the title.

_LOGO_STATIC = (
    " [#00ff88]✧[/] [bold #ffcc00]★[/]\n"
    "  [bold #00ffff]███[/]   [bold #00ffff]NatShell[/]\n"
    " [#00ccaa]█████[/]\n"
    "  [#30304a]██[/] [#00ff88]·[/]"
)

_LOGO_FRAMES = [
    _LOGO_STATIC,
    # Frame 1: sparkle upper-right
    (
        "   [bold #ffcc00]★[/] [#66ffff]✧[/]\n"
        " [#00ff88]·[/] [#00ccaa]███[/]   [bold #00ffff]NatShell[/]\n"
        " [#00aa88]█████[/]\n"
        "  [#30304a]██[/]"
    ),
    # Frame 2: sparkle upper-left, dot mid-right
    (
        "[#66ffff]✧[/]  [bold #ffcc00]★[/]\n"
        "  [#00aa88]███[/]   [bold #00ffff]NatShell[/]\n"
        " [bold #00ffff]█████[/] [#00ff88]·[/]\n"
        "  [#30304a]██[/]"
    ),
    # Frame 3: dot between tree and text, dot lower-left
    (
        "   [bold #ffcc00]★[/]\n"
        "  [bold #00ffff]███[/] [#00ff88]·[/] [bold #00ffff]NatShell[/]\n"
        " [#00ccaa]█████[/]\n"
        " [#00ff88]·[/] [#30304a]██[/]"
    ),
]

# Thinking indicator with bouncing highlight dot
_THINKING_FRAMES = [
    "  [#00ccaa]Thinking[/]  [bold #00ffff]●[/] [#007766]●[/] [#007766]●[/]",
    "  [#00ccaa]Thinking[/]  [#007766]●[/] [bold #00ffff]●[/] [#007766]●[/]",
    "  [#00ccaa]Thinking[/]  [#007766]●[/] [#007766]●[/] [bold #00ffff]●[/]",
    "  [#00ccaa]Thinking[/]  [#007766]●[/] [bold #00ffff]●[/] [#007766]●[/]",
]


class LogoBanner(Horizontal):
    """Tree logo with neon NatShell branding and sparkle animation."""

    DEFAULT_CSS = """
    LogoBanner {
        dock: top;
        height: auto;
        padding: 0 2;
        background: #16213e;
    }
    #logo-lines {
        width: 1fr;
        height: auto;
    }
    #logo-lines > Static {
        height: 1;
    }
    #banner-btn-wrap {
        width: auto;
        height: auto;
        align: right top;
    }
    """

    def __init__(self) -> None:
        super().__init__(id="logo-banner")
        self._animation_timer = None
        self._frame_index = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="logo-lines"):
            for line in _LOGO_STATIC.split("\n"):
                yield Static(line, classes="logo-line")
        with Vertical(id="banner-btn-wrap"):
            yield Button("\U0001f4cb Copy Chat", id="copy-chat-btn")

    def _update_content(self, content: str) -> None:
        lines = content.split("\n")
        children = list(self.query(".logo-line"))
        for i, child in enumerate(children):
            if i < len(lines):
                child.update(lines[i])

    def start_animation(self) -> None:
        """Start sparkle animation around the tree."""
        if self._animation_timer is None:
            self._animation_timer = self.set_interval(0.4, self._next_frame)

    def stop_animation(self) -> None:
        """Stop animation and reset to static logo."""
        if self._animation_timer is not None:
            self._animation_timer.stop()
            self._animation_timer = None
        self._frame_index = 0
        self._update_content(_LOGO_STATIC)

    def _next_frame(self) -> None:
        self._frame_index = (self._frame_index + 1) % len(_LOGO_FRAMES)
        self._update_content(_LOGO_FRAMES[self._frame_index])


class CopyableMessage(Horizontal):
    """Base for message widgets with a copy button."""

    def __init__(self, formatted: str, raw_text: str) -> None:
        super().__init__()
        self._formatted = formatted
        self._raw_text = raw_text

    @property
    def copyable_text(self) -> str:
        return self._raw_text

    def compose(self) -> ComposeResult:
        yield Static(self._formatted, classes="msg-text")
        yield Button("\U0001f4cb", classes="msg-copy-btn")

    @on(Button.Pressed, ".msg-copy-btn")
    def on_copy_pressed(self) -> None:
        from natshell.ui import clipboard

        if clipboard.copy(self._raw_text, self.app):
            self.app.notify("Copied!", timeout=2)
        else:
            self.app.notify("Copy failed — no clipboard tool found", severity="error", timeout=3)


class UserMessage(CopyableMessage):
    """A message from the user."""

    def __init__(self, text: str) -> None:
        super().__init__(f"[bold cyan]You:[/] {text}", text)


class AssistantMessage(CopyableMessage):
    """A text response from the assistant."""

    def __init__(self, text: str) -> None:
        super().__init__(f"[bold green]NatShell:[/] {text}", text)


class PlanningMessage(CopyableMessage):
    """The assistant's planning/reasoning text before tool calls."""

    def __init__(self, text: str) -> None:
        super().__init__(f"[dim italic]{text}[/]", text)


class CommandBlock(Vertical):
    """A command execution block showing the command, output, and a copy button."""

    def __init__(self, command: str, output: str = "", exit_code: int = 0) -> None:
        super().__init__()
        self._command = command
        self._output = output
        self._exit_code = exit_code

    def compose(self) -> ComposeResult:
        color = "green" if self._exit_code == 0 else "red"
        with Horizontal(classes="cmd-header"):
            yield Static(
                f"[bold {color}]$[/] [bold]{self._command}[/]",
                classes="cmd-text",
            )
            yield Button("Copy", classes="copy-btn")
        if self._output:
            yield Static(self._output, classes="cmd-output")

    def set_result(self, output: str, exit_code: int) -> None:
        """Update the block with command output after execution."""
        self._output = output
        self._exit_code = exit_code
        color = "green" if exit_code == 0 else "red"
        try:
            self.query_one(".cmd-text", Static).update(
                f"[bold {color}]$[/] [bold]{self._command}[/]"
            )
        except Exception:
            pass
        existing = self.query(".cmd-output")
        if existing:
            existing.first().update(output)
        elif output:
            self.mount(Static(output, classes="cmd-output"))

    @property
    def copyable_text(self) -> str:
        text = f"$ {self._command}"
        if self._output:
            text += f"\n{self._output}"
        return text

    @on(Button.Pressed, ".copy-btn")
    def on_copy_pressed(self) -> None:
        from natshell.ui import clipboard

        if clipboard.copy(self.copyable_text, self.app):
            self.app.notify("Copied!", timeout=2)
        else:
            self.app.notify("Copy failed — no clipboard tool found", severity="error", timeout=3)


class BlockedMessage(CopyableMessage):
    """Warning that a command was blocked."""

    def __init__(self, command: str) -> None:
        super().__init__(f"⛔ BLOCKED: {command}", f"BLOCKED: {command}")


class SystemMessage(CopyableMessage):
    """A system/feedback message for slash command responses."""

    def __init__(self, text: str) -> None:
        super().__init__(f"[bold yellow]System:[/] {text}", text)


class HelpMessage(CopyableMessage):
    """A bordered help display showing available commands."""

    def __init__(self, text: str) -> None:
        super().__init__(text, text)


class ThinkingIndicator(Static):
    """Animated thinking indicator with bouncing dots."""

    def __init__(self) -> None:
        super().__init__(_THINKING_FRAMES[0])
        self._frame = 0
        self._timer = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.3, self._tick)

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(_THINKING_FRAMES)
        self.update(_THINKING_FRAMES[self._frame])


class ConfirmScreen(ModalScreen[bool]):
    """Modal confirmation dialog for dangerous commands."""

    def __init__(self, tool_call: ToolCall) -> None:
        super().__init__()
        self.tool_call = tool_call

    def compose(self) -> ComposeResult:
        command = self.tool_call.arguments.get("command", str(self.tool_call.arguments))
        with Vertical(id="confirm-dialog"):
            yield Label(f"[bold yellow]⚠ Confirmation Required[/]\n")
            yield Label(f"Tool: [bold]{self.tool_call.name}[/]")
            yield Label(f"Command: [bold]{command}[/]\n")
            yield Label("Do you want to execute this command?")
            with Vertical(id="confirm-buttons"):
                yield Button("Yes, execute", variant="warning", id="btn-yes")
                yield Button("No, skip", variant="default", id="btn-no")

    @on(Button.Pressed, "#btn-yes")
    def on_yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#btn-no")
    def on_no(self) -> None:
        self.dismiss(False)


class SudoPasswordScreen(ModalScreen[str | None]):
    """Modal dialog for entering the sudo password."""

    def __init__(self, command: str) -> None:
        super().__init__()
        self._command = command

    def compose(self) -> ComposeResult:
        with Vertical(id="sudo-dialog"):
            yield Label("[bold yellow]Sudo Password Required[/]\n")
            yield Label(f"Command: [bold]{self._command}[/]\n")
            yield Label("Enter your password:")
            yield Input(password=True, id="sudo-password")
            with Horizontal(id="sudo-buttons"):
                yield Button("Submit", variant="warning", id="btn-sudo-ok")
                yield Button("Cancel", variant="default", id="btn-sudo-cancel")

    def on_mount(self) -> None:
        self.query_one("#sudo-password", Input).focus()

    @on(Input.Submitted, "#sudo-password")
    def on_password_submitted(self) -> None:
        password = self.query_one("#sudo-password", Input).value
        self.dismiss(password if password else None)

    @on(Button.Pressed, "#btn-sudo-ok")
    def on_ok(self) -> None:
        password = self.query_one("#sudo-password", Input).value
        self.dismiss(password if password else None)

    @on(Button.Pressed, "#btn-sudo-cancel")
    def on_cancel(self) -> None:
        self.dismiss(None)
