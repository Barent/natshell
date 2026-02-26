"""NatShell TUI application — the main user interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer
from textual.widgets import Footer, Header, Input, Static

from natshell.agent.loop import AgentEvent, AgentLoop, EventType
from natshell.inference.engine import ToolCall
from natshell.ui.widgets import (
    AssistantMessage,
    BlockedMessage,
    CommandBlock,
    ConfirmScreen,
    PlanningMessage,
    ThinkingIndicator,
    UserMessage,
)


class NatShellApp(App):
    """The NatShell TUI application."""

    TITLE = "NatShell"
    SUB_TITLE = "Natural Language Shell"
    CSS_PATH = Path("ui/styles.tcss")

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
