"""Custom Textual widgets for the NatShell TUI."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from natshell.inference.engine import ToolCall


class UserMessage(Static):
    """A message from the user."""

    def __init__(self, text: str) -> None:
        super().__init__(f"[bold cyan]You:[/] {text}")


class AssistantMessage(Static):
    """A text response from the assistant."""

    def __init__(self, text: str) -> None:
        super().__init__(f"[bold green]NatShell:[/] {text}")


class PlanningMessage(Static):
    """The assistant's planning/reasoning text before tool calls."""

    def __init__(self, text: str) -> None:
        super().__init__(f"[dim italic]{text}[/]")


class CommandBlock(Static):
    """A command execution block showing the command and its output."""

    def __init__(self, command: str, output: str = "", exit_code: int = 0) -> None:
        color = "green" if exit_code == 0 else "red"
        parts = [f"[bold {color}]$[/] [bold]{command}[/]"]
        if output:
            parts.append(f"\n{output}")
        super().__init__("\n".join(parts))


class BlockedMessage(Static):
    """Warning that a command was blocked."""

    def __init__(self, command: str) -> None:
        super().__init__(f"⛔ BLOCKED: {command}")


class SystemMessage(Static):
    """A system/feedback message for slash command responses."""

    def __init__(self, text: str) -> None:
        super().__init__(f"[bold yellow]System:[/] {text}")


class HelpMessage(Static):
    """A bordered help display showing available commands."""

    def __init__(self, text: str) -> None:
        super().__init__(text)


class ThinkingIndicator(Static):
    """Animated thinking indicator."""

    def __init__(self) -> None:
        super().__init__("[dim]⏳ Thinking...[/]")


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
