"""Standalone slash-command handlers extracted from app.py."""

from __future__ import annotations

from textual.containers import ScrollableContainer

from natshell.agent.loop import AgentLoop
from natshell.ui import clipboard
from natshell.ui.widgets import HelpMessage, SystemMessage, _escape


def show_help(conversation: ScrollableContainer) -> None:
    """Show available slash commands."""
    help_text = (
        "[bold]Available Commands[/]\n\n"
        "  [bold cyan]/help[/]                  Show this help message\n"
        "  [bold cyan]/clear[/]                 Clear chat and model context\n"
        "  [bold cyan]/compact[/]               Compact context, keeping key facts\n"
        "  [bold cyan]/cmd <command>[/]         Execute a shell command directly\n"
        "  [bold cyan]/exeplan <file>[/]        Preview a multi-step plan\n"
        "  [bold cyan]/exeplan run <file>[/]    Execute all plan steps\n"
        "  [bold cyan]/plan <description>[/]    Generate a multi-step plan\n"
        "  [bold cyan]/model[/]                 Show current engine/model info\n"
        "  [bold cyan]/model list[/]            List models on remote server\n"
        "  [bold cyan]/model use <name>[/]      Switch to a remote model\n"
        "  [bold cyan]/model switch[/]          Switch local model (or Ctrl+P)\n"
        "  [bold cyan]/model local[/]           Switch back to local model\n"
        "  [bold cyan]/model default <name>[/]  Save default remote model\n"
        "  [bold cyan]/profile[/]               List configuration profiles\n"
        "  [bold cyan]/profile <name>[/]        Apply a configuration profile\n"
        "  [bold cyan]/keys[/]                  Show keyboard shortcuts\n"
        "  [bold cyan]/history[/]               Show conversation context size\n"
        "  [bold cyan]/undo[/]                  Undo last file edit or write\n"
        "  [bold cyan]/save [name][/]           Save current session\n"
        "  [bold cyan]/load [id][/]             Load a saved session\n"
        "  [bold cyan]/sessions[/]              List saved sessions\n\n"
        "[bold]Copy & Paste[/]\n\n"
        "  Click [bold cyan]\U0001f4cb[/] on any message to copy it.\n"
        "  Click [bold cyan]\U0001f4cb Copy Chat[/] or press"
        " [bold cyan]Ctrl+E[/] to copy entire chat.\n"
        "  [bold cyan]Shift+drag[/] to select text, then use terminal copy.\n"
        "  [bold cyan]Right-click[/] or [bold cyan]Ctrl+Y[/] to copy Textual selection.\n"
        "  [bold cyan]Ctrl+Shift+V[/] or terminal paste to paste.\n"
        f"  Clipboard: [bold cyan]{clipboard.backend_name()}[/]\n\n"
        "[dim]Tip: Use /cmd when you know the exact command to run.[/]"
    )
    conversation.mount(HelpMessage(help_text))


def compact_chat(agent: AgentLoop, conversation: ScrollableContainer) -> None:
    """Compact conversation context, keeping key facts."""
    # Dry-run to preview what will happen
    preview = agent.compact_history(dry_run=True)
    if not preview["compacted"]:
        conversation.mount(SystemMessage("Nothing to compact — conversation is too short."))
        return

    # Actually compact
    stats = agent.compact_history()

    conversation.remove_children()
    saved_tokens = stats["before_tokens"] - stats["after_tokens"]
    dropped_msgs = stats["before_msgs"] - stats["after_msgs"]
    summary_lines = [
        "[bold]Context compacted[/]\n",
        f"  Messages: {stats['before_msgs']} \u2192 {stats['after_msgs']}"
        f" ({dropped_msgs} dropped)",
        f"  Tokens:   ~{stats['before_tokens']} \u2192 ~{stats['after_tokens']}"
        f" (~{saved_tokens} freed)",
    ]
    if stats["summary"]:
        summary_lines.append(f"\n[dim]Preserved facts:[/]\n{_escape(stats['summary'])}")
    conversation.mount(SystemMessage("\n".join(summary_lines)))


def show_history_info(agent: AgentLoop, conversation: ScrollableContainer) -> None:
    """Show conversation context size and context window usage."""
    msg_count = len(agent.messages)
    char_count = sum(len(str(m.get("content", ""))) for m in agent.messages)

    parts = [f"Conversation: {msg_count} messages, ~{char_count} chars"]

    cm = agent._context_manager
    if cm:
        used = cm.estimate_tokens(agent.messages)
        budget = cm.context_budget
        pct = min(100, int(used / budget * 100)) if budget > 0 else 0
        bar_len = 20
        filled = int(bar_len * pct / 100)
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

        if pct >= 90:
            color = "red"
        elif pct >= 70:
            color = "yellow"
        else:
            color = "green"

        parts.append(f"  Context: [{color}]{bar}[/] {pct}% ({used}/{budget} tokens)")

        if cm.trimmed_count > 0:
            parts.append(f"  Trimmed: {cm.trimmed_count} messages compressed")

    conversation.mount(SystemMessage("\n".join(parts)))


def handle_undo(conversation: ScrollableContainer) -> None:
    """Undo the last file edit or write by restoring from backup."""
    from natshell.backup import get_backup_manager

    success, message = get_backup_manager().undo_last()
    conversation.mount(SystemMessage(message))
