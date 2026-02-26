"""NatShell TUI application — the main user interface."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.events import MouseUp
from textual.widgets import Footer, Header, Input, Static

from natshell.agent.loop import AgentEvent, AgentLoop, EventType
from natshell.config import NatShellConfig, save_ollama_default
from natshell.inference.engine import ToolCall
from natshell.inference.ollama import list_models, normalize_base_url, ping_server
from natshell.safety.classifier import Risk
from natshell.tools.execute_shell import execute_shell
from natshell.ui import clipboard
from natshell.ui.widgets import (
    AssistantMessage,
    BlockedMessage,
    CommandBlock,
    ConfirmScreen,
    HelpMessage,
    PlanningMessage,
    SystemMessage,
    ThinkingIndicator,
    UserMessage,
)

logger = logging.getLogger(__name__)


SLASH_COMMANDS = [
    ("/help", "Show available commands"),
    ("/clear", "Clear chat and model context"),
    ("/cmd", "Execute a shell command directly"),
    ("/model", "Show current engine/model info"),
    ("/model list", "List models on the remote server"),
    ("/model use", "Switch to a remote model"),
    ("/model local", "Switch back to local model"),
    ("/model default", "Set default remote model"),
    ("/history", "Show conversation context size"),
]


class NatShellApp(App):
    """The NatShell TUI application."""

    TITLE = "NatShell"
    SUB_TITLE = "Natural Language Shell"
    CSS_PATH = Path("ui/styles.tcss")

    ALLOW_SELECT = True

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear Chat"),
        Binding("ctrl+y", "copy_selection", "Copy", show=False),
    ]

    def __init__(self, agent: AgentLoop, config: NatShellConfig | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self._config = config or NatShellConfig()
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield ScrollableContainer(
            Static("[dim]Welcome to NatShell. Type a request to get started. Use /help for commands.[/]\n"),
            id="conversation",
        )
        with Vertical(id="input-area"):
            yield Static(id="slash-suggestions")
            yield Input(
                placeholder="Ask me anything about your system...",
                id="user-input",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#user-input", Input).focus()

    @on(Input.Changed, "#user-input")
    def on_input_changed(self, event: Input.Changed) -> None:
        """Show/hide slash command suggestions as the user types."""
        text = event.value
        suggestions = self.query_one("#slash-suggestions", Static)

        if text.startswith("/") and " " not in text:
            matches = [
                (cmd, desc) for cmd, desc in SLASH_COMMANDS
                if cmd.startswith(text.lower())
            ]
            if matches:
                lines = [
                    f"  [bold cyan]{cmd}[/]  [dim]{desc}[/]"
                    for cmd, desc in matches
                ]
                suggestions.update("\n".join(lines))
                suggestions.display = True
                return

        suggestions.display = False

    @on(Input.Submitted, "#user-input")
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        self.query_one("#slash-suggestions", Static).display = False
        if not user_text or self._busy:
            return

        input_widget = self.query_one("#user-input", Input)
        input_widget.value = ""

        # Intercept slash commands before the agent
        if user_text.startswith("/"):
            await self._handle_slash_command(user_text)
            return

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
                            block.set_result(
                                event.tool_result.output or event.tool_result.error,
                                event.tool_result.exit_code,
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

    async def _handle_slash_command(self, text: str) -> None:
        """Parse and dispatch slash commands."""
        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        conversation = self.query_one("#conversation", ScrollableContainer)

        match command:
            case "/help":
                self._show_help(conversation)
            case "/clear":
                self.action_clear_chat()
            case "/cmd":
                await self._handle_cmd(args, conversation)
            case "/model":
                await self._handle_model_command(args, conversation)
            case "/history":
                self._show_history_info(conversation)
            case _:
                conversation.mount(
                    SystemMessage(f"Unknown command: {command}. Type /help for available commands.")
                )

        conversation.scroll_end()

    async def _handle_cmd(self, command: str, conversation: ScrollableContainer) -> None:
        """Execute a shell command directly, bypassing the AI."""
        if not command:
            conversation.mount(SystemMessage("Usage: /cmd <command>"))
            return

        conversation.mount(UserMessage(f"/cmd {command}"))

        # Safety classification
        risk = self.agent.safety.classify_command(command)

        if risk == Risk.BLOCKED:
            conversation.mount(BlockedMessage(command))
            return

        if risk == Risk.CONFIRM:
            synthetic_call = ToolCall(id="slash-cmd", name="execute_shell", arguments={"command": command})
            confirmed = await self.push_screen_wait(ConfirmScreen(synthetic_call))
            if not confirmed:
                conversation.mount(SystemMessage("Command cancelled."))
                return

        self._busy = True
        try:
            result = await execute_shell(command)
            output = result.output or result.error
            conversation.mount(CommandBlock(command, output, result.exit_code))

            # Inject into agent context so the model knows what the user ran
            self.agent.messages.append({
                "role": "user",
                "content": (
                    f"[The user directly ran a shell command: `{command}`]\n"
                    f"Exit code: {result.exit_code}\n"
                    f"Output:\n{output}"
                ),
            })
        finally:
            self._busy = False
            self.query_one("#user-input", Input).focus()

    def _show_help(self, conversation: ScrollableContainer) -> None:
        """Show available slash commands."""
        help_text = (
            "[bold]Available Commands[/]\n\n"
            "  [bold cyan]/help[/]                  Show this help message\n"
            "  [bold cyan]/clear[/]                 Clear chat and model context\n"
            "  [bold cyan]/cmd <command>[/]         Execute a shell command directly\n"
            "  [bold cyan]/model[/]                 Show current engine/model info\n"
            "  [bold cyan]/model list[/]            List models on remote server\n"
            "  [bold cyan]/model use <name>[/]      Switch to a remote model\n"
            "  [bold cyan]/model local[/]           Switch back to local model\n"
            "  [bold cyan]/model default <name>[/]  Save default remote model\n"
            "  [bold cyan]/history[/]               Show conversation context size\n\n"
            "[bold]Copy & Paste[/]\n\n"
            "  Click [bold cyan]Copy[/] on any command block to copy it.\n"
            "  [bold cyan]Shift+drag[/] to select text, then use terminal copy.\n"
            "  [bold cyan]Right-click[/] or [bold cyan]Ctrl+Y[/] to copy Textual selection.\n"
            "  [bold cyan]Ctrl+Shift+V[/] or terminal paste to paste.\n"
            f"  Clipboard: [bold cyan]{clipboard.backend_name()}[/]\n\n"
            "[dim]Tip: Use /cmd when you know the exact command to run.[/]"
        )
        conversation.mount(HelpMessage(help_text))

    async def _handle_model_command(self, args: str, conversation: ScrollableContainer) -> None:
        """Dispatch /model subcommands."""
        if not args:
            self._show_model_info(conversation)
            return

        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower()
        subargs = parts[1] if len(parts) > 1 else ""

        match subcmd:
            case "list":
                await self._model_list(conversation)
            case "use":
                if not subargs:
                    conversation.mount(SystemMessage("Usage: /model use <model-name>"))
                else:
                    await self._model_use(subargs.strip(), conversation)
            case "local":
                await self._model_switch_local(conversation)
            case "default":
                if not subargs:
                    conversation.mount(SystemMessage("Usage: /model default <model-name>"))
                else:
                    self._model_set_default(subargs.strip(), conversation)
            case _:
                conversation.mount(SystemMessage(
                    "Unknown subcommand. Usage:\n"
                    "  /model         — show current info\n"
                    "  /model list    — list remote models\n"
                    "  /model use <n> — switch to remote model\n"
                    "  /model local   — switch to local model\n"
                    "  /model default <n> — set default model"
                ))

    def _get_remote_base_url(self) -> str | None:
        """Find the remote base URL from config or current engine."""
        if self._config.ollama.url:
            return normalize_base_url(self._config.ollama.url)
        if self._config.remote.url:
            return normalize_base_url(self._config.remote.url)
        info = self.agent.engine.engine_info()
        if info.base_url:
            return normalize_base_url(info.base_url)
        return None

    def _show_model_info(self, conversation: ScrollableContainer) -> None:
        """Show current model/engine information."""
        info = self.agent.engine.engine_info()
        parts = ["[bold]Model Info[/]"]
        parts.append(f"  Engine: {info.engine_type}")
        if info.model_name:
            parts.append(f"  Model: {info.model_name}")
        if info.base_url:
            parts.append(f"  URL: {info.base_url}")
        if info.n_ctx:
            parts.append(f"  Context: {info.n_ctx} tokens")
        if info.n_gpu_layers:
            parts.append(f"  GPU layers: {info.n_gpu_layers}")
        remote_url = self._get_remote_base_url()
        if remote_url:
            parts.append(f"\n[dim]Tip: /model list — see available remote models[/]")
        conversation.mount(SystemMessage("\n".join(parts)))

    async def _model_list(self, conversation: ScrollableContainer) -> None:
        """Ping server and list available models."""
        base_url = self._get_remote_base_url()
        if not base_url:
            conversation.mount(SystemMessage(
                "No remote server configured.\n"
                "Set [ollama] url in ~/.config/natshell/config.toml\n"
                "or use --remote <url> at startup."
            ))
            return

        conversation.mount(SystemMessage(f"Checking {base_url}..."))

        reachable = await ping_server(base_url)
        if not reachable:
            conversation.mount(SystemMessage(f"[red]Cannot reach server at {base_url}[/]"))
            return

        models = await list_models(base_url)
        if not models:
            conversation.mount(SystemMessage("Server is running but returned no models."))
            return

        current_info = self.agent.engine.engine_info()
        lines = ["[bold]Available Models[/]"]
        for m in models:
            marker = " [green]◀ active[/]" if m.name == current_info.model_name else ""
            detail = ""
            if m.size_gb:
                detail += f" ({m.size_gb} GB)"
            if m.parameter_size:
                detail += f" [{m.parameter_size}]"
            lines.append(f"  {m.name}{detail}{marker}")
        lines.append(f"\n[dim]Use /model use <name> to switch[/]")
        conversation.mount(SystemMessage("\n".join(lines)))

    async def _model_use(self, model_name: str, conversation: ScrollableContainer) -> None:
        """Switch to a remote model."""
        base_url = self._get_remote_base_url()
        if not base_url:
            conversation.mount(SystemMessage(
                "No remote server configured. Set [ollama] url in config."
            ))
            return

        reachable = await ping_server(base_url)
        if not reachable:
            conversation.mount(SystemMessage(f"[red]Cannot reach server at {base_url}[/]"))
            return

        from natshell.inference.remote import RemoteEngine

        # Ensure URL has /v1 for the OpenAI-compatible endpoint
        api_url = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
        new_engine = RemoteEngine(base_url=api_url, model=model_name)
        await self.agent.swap_engine(new_engine)
        conversation.mount(SystemMessage(
            f"Switched to [bold]{model_name}[/] on {base_url}\n"
            "[dim]Conversation history cleared.[/]"
        ))

    async def _model_switch_local(self, conversation: ScrollableContainer) -> None:
        """Switch back to the local model."""
        info = self.agent.engine.engine_info()
        if info.engine_type == "local":
            conversation.mount(SystemMessage("Already using the local model."))
            return

        mc = self._config.model
        model_path = mc.path
        if model_path == "auto":
            model_dir = Path.home() / ".local" / "share" / "natshell" / "models"
            model_path = str(model_dir / mc.hf_file)

        if not Path(model_path).exists():
            conversation.mount(SystemMessage(
                f"[red]Local model not found at {model_path}[/]\n"
                "Run natshell --download to fetch it."
            ))
            return

        conversation.mount(SystemMessage("Loading local model..."))

        from natshell.inference.local import LocalEngine

        try:
            engine = await asyncio.to_thread(
                LocalEngine,
                model_path=model_path,
                n_ctx=mc.n_ctx,
                n_threads=mc.n_threads,
                n_gpu_layers=mc.n_gpu_layers,
            )
            await self.agent.swap_engine(engine)
            conversation.mount(SystemMessage(
                f"Switched to local model: [bold]{Path(model_path).name}[/]\n"
                "[dim]Conversation history cleared.[/]"
            ))
        except Exception as e:
            conversation.mount(SystemMessage(f"[red]Failed to load local model: {e}[/]"))

    def _model_set_default(self, model_name: str, conversation: ScrollableContainer) -> None:
        """Persist the default model and remote URL to user config."""
        remote_url = self._get_remote_base_url()
        config_path = save_ollama_default(model_name, url=remote_url)
        parts = [f"Default model set to [bold]{model_name}[/]"]
        if remote_url:
            parts.append(f"Server: {remote_url}")
        parts.append(f"Saved to {config_path}")
        conversation.mount(SystemMessage("\n".join(parts)))

    def _show_history_info(self, conversation: ScrollableContainer) -> None:
        """Show conversation context size."""
        msg_count = len(self.agent.messages)
        # Estimate token count from message content
        char_count = sum(
            len(str(m.get("content", "")))
            for m in self.agent.messages
        )
        conversation.mount(SystemMessage(
            f"Conversation: {msg_count} messages, ~{char_count} chars"
        ))

    def on_mouse_up(self, event: MouseUp) -> None:
        """Copy selected text to clipboard on right-click."""
        if event.button == 3:
            selected = self.screen.get_selected_text()
            if selected:
                if clipboard.copy(selected, self):
                    self.notify("Copied to clipboard", timeout=2)
                else:
                    self.notify("Copy failed — install xclip", severity="error", timeout=3)

    def action_copy_selection(self) -> None:
        """Copy selected text to clipboard (Ctrl+Y)."""
        selected = self.screen.get_selected_text()
        if selected:
            if clipboard.copy(selected, self):
                self.notify("Copied to clipboard", timeout=2)
            else:
                self.notify("Copy failed — install xclip", severity="error", timeout=3)

    def action_clear_chat(self) -> None:
        """Clear the conversation and agent history."""
        conversation = self.query_one("#conversation", ScrollableContainer)
        conversation.remove_children()
        conversation.mount(
            Static("[dim]Chat cleared. Type a new request.[/]\n")
        )
        self.agent.clear_history()
