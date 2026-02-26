"""NatShell TUI application — the main user interface."""

from __future__ import annotations

import asyncio
from typing import Any

from rich.markdown import Markdown
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, Static

from natshell.agent.loop import AgentEvent, AgentLoop, EventType
from natshell.inference.engine import ToolCall


# ─── Custom Widgets ─────────────────────────────────────────────────────────


class UserMessage(Static):
    """A message from the user."""

    DEFAULT_CSS = """
    UserMessage {
        margin: 1 0 0 4;
        padding: 0 1;
        color: $accent;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__(f"[bold cyan]You:[/] {text}")


class AssistantMessage(Static):
    """A text response from the assistant."""

    DEFAULT_CSS = """
    AssistantMessage {
        margin: 1 0 0 0;
        padding: 0 1;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__(f"[bold green]NatShell:[/] {text}")


class PlanningMessage(Static):
    """The assistant's planning/reasoning text before tool calls."""

    DEFAULT_CSS = """
    PlanningMessage {
        margin: 1 0 0 0;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__(f"[dim italic]{text}[/]")


class CommandBlock(Static):
    """A command execution block showing the command and its output."""

    DEFAULT_CSS = """
    CommandBlock {
        margin: 0 2;
        padding: 1 2;
        border: round $surface-lighten-2;
        background: $surface;
    }
    """

    def __init__(self, command: str, output: str = "", exit_code: int = 0) -> None:
        color = "green" if exit_code == 0 else "red"
        parts = [f"[bold {color}]$[/] [bold]{command}[/]"]
        if output:
            parts.append(f"\n{output}")
        super().__init__("\n".join(parts))


class BlockedMessage(Static):
    """Warning that a command was blocked."""

    DEFAULT_CSS = """
    BlockedMessage {
        margin: 0 2;
        padding: 1 2;
        background: $error-darken-3;
        color: $error;
        border: round $error;
    }
    """

    def __init__(self, command: str) -> None:
        super().__init__(f"⛔ BLOCKED: {command}")


class ThinkingIndicator(Static):
    """Animated thinking indicator."""

    DEFAULT_CSS = """
    ThinkingIndicator {
        margin: 1 0 0 0;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self) -> None:
        super().__init__("[dim]⏳ Thinking...[/]")


# ─── Confirmation Screen ────────────────────────────────────────────────────


class ConfirmScreen(ModalScreen[bool]):
    """Modal confirmation dialog for dangerous commands."""

    DEFAULT_CSS = """
    ConfirmScreen {
        align: center middle;
    }
    #confirm-dialog {
        width: 70;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        padding: 2 4;
        border: thick $warning;
        background: $surface;
    }
    #confirm-buttons {
        margin-top: 2;
        align: center middle;
    }
    #confirm-buttons Button {
        margin: 0 2;
    }
    """

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


# ─── Main Application ───────────────────────────────────────────────────────


class NatShellApp(App):
    """The NatShell TUI application."""

    TITLE = "NatShell"
    SUB_TITLE = "Natural Language Shell"

    CSS = """
    Screen {
        background: $surface;
    }
    #conversation {
        height: 1fr;
        padding: 1 2;
        scrollbar-size: 1 1;
    }
    #user-input {
        dock: bottom;
        margin: 0 1 1 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear Chat"),
    ]

    def __init__(self, agent: AgentLoop, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield ScrollableContainer(
            Static("[dim]Welcome to NatShell. Type a request to get started.[/]\n"),
            id="conversation",
        )
        yield Input(
            placeholder="Ask me anything about your system...",
            id="user-input",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#user-input", Input).focus()

    @on(Input.Submitted, "#user-input")
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text or self._busy:
            return

        input_widget = self.query_one("#user-input", Input)
        input_widget.value = ""
        self._busy = True

        # Add user message to conversation
        conversation = self.query_one("#conversation", ScrollableContainer)
        conversation.mount(UserMessage(user_text))

        # Run the agent loop
        self.run_agent(user_text)

    @work(exclusive=True, thread=False)
    async def run_agent(self, user_text: str) -> None:
        """Run the agent loop in a background worker."""
        conversation = self.query_one("#conversation", ScrollableContainer)
        thinking = None

        async def confirm_callback(tool_call: ToolCall) -> bool:
            """Show confirmation dialog and return user's choice."""
            return await self.push_screen_wait(ConfirmScreen(tool_call))

        try:
            async for event in self.agent.handle_user_message(
                user_text, confirm_callback=confirm_callback
            ):
                # Remove thinking indicator when we get a real event
                if thinking and event.type != EventType.THINKING:
                    thinking.remove()
                    thinking = None

                match event.type:
                    case EventType.THINKING:
                        if not thinking:
                            thinking = ThinkingIndicator()
                            conversation.mount(thinking)

                    case EventType.PLANNING:
                        conversation.mount(PlanningMessage(event.data))

                    case EventType.EXECUTING:
                        cmd = event.tool_call.arguments.get(
                            "command", str(event.tool_call.arguments)
                        )
                        # Mount a command block with just the command, no output yet
                        block = CommandBlock(cmd)
                        block.id = f"cmd-{event.tool_call.id}"
                        conversation.mount(block)

                    case EventType.TOOL_RESULT:
                        # Update the command block with output
                        block_id = f"cmd-{event.tool_call.id}"
                        try:
                            block = self.query_one(f"#{block_id}", CommandBlock)
                            cmd = event.tool_call.arguments.get(
                                "command", str(event.tool_call.arguments)
                            )
                            block.update(
                                CommandBlock(
                                    cmd,
                                    event.tool_result.output or event.tool_result.error,
                                    event.tool_result.exit_code,
                                ).render()
                            )
                        except Exception:
                            # Fallback: just mount a new block
                            conversation.mount(
                                CommandBlock(
                                    str(event.tool_call.arguments),
                                    event.tool_result.output or event.tool_result.error,
                                    event.tool_result.exit_code,
                                )
                            )

                    case EventType.BLOCKED:
                        cmd = event.tool_call.arguments.get(
                            "command", str(event.tool_call.arguments)
                        )
                        conversation.mount(BlockedMessage(cmd))

                    case EventType.RESPONSE:
                        conversation.mount(AssistantMessage(event.data))

                    case EventType.ERROR:
                        conversation.mount(
                            Static(f"[bold red]Error:[/] {event.data}")
                        )

                # Auto-scroll to bottom
                conversation.scroll_end()

        except Exception as e:
            conversation.mount(Static(f"[bold red]Agent error:[/] {e}"))

        finally:
            if thinking:
                thinking.remove()
            self._busy = False
            self.query_one("#user-input", Input).focus()

    def action_clear_chat(self) -> None:
        """Clear the conversation and agent history."""
        conversation = self.query_one("#conversation", ScrollableContainer)
        conversation.remove_children()
        conversation.mount(
            Static("[dim]Chat cleared. Type a new request.[/]\n")
        )
        self.agent.clear_history()
