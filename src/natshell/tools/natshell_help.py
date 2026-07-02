"""Look up NatShell documentation by topic — self-help for the agent."""

from __future__ import annotations

import logging
from pathlib import Path

from natshell.config import SafetyConfig
from natshell.tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# ── Safety config injection (same pattern as set_sudo_password) ────────────

_safety_config: SafetyConfig | None = None


def set_safety_config(config: SafetyConfig) -> None:
    """Inject the live SafetyConfig so the 'safety' topic can report patterns."""
    global _safety_config
    _safety_config = config


# ── Static topic loading ──────────────────────────────────────────────────

from importlib import resources


def _topics_dir():
    return resources.files("natshell.tools").joinpath("help_topics")


def _load_static_topic(topic: str) -> str | None:
    """Read a static help topic from the bundled markdown files."""
    try:
        return _topics_dir().joinpath(f"{topic}.md").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None


def _static_topic_names() -> list[str]:
    try:
        return sorted(
            entry.name[:-3]
            for entry in _topics_dir().iterdir()
            if entry.name.endswith(".md")
        )
    except (FileNotFoundError, OSError):
        return []

# ── Dynamic topic handlers ────────────────────────────────────────────────


def _topic_config() -> str:
    """Read the user's config.toml and return its contents."""
    from natshell.platform import config_dir

    user_config = config_dir() / "config.toml"
    if not user_config.exists():
        return (
            "No user config file found at ~/.config/natshell/config.toml\n"
            "NatShell is using built-in defaults. To customize, copy "
            "config.default.toml to that path and edit it."
        )
    try:
        text = user_config.read_text()
        # Truncate if very large
        if len(text) > 3000:
            text = text[:3000] + "\n... [truncated]"
        return f"User config (~/.config/natshell/config.toml):\n\n{text}"
    except Exception as e:
        return f"Error reading config: {e}"


def _topic_config_reference() -> str:
    """Read the bundled config.default.toml reference."""
    default_path = Path(__file__).parent.parent / "config.default.toml"
    if not default_path.exists():
        return (
            "Bundled config.default.toml not found. "
            "This file is included in the NatShell source tree."
        )
    try:
        text = default_path.read_text()
        if len(text) > 3000:
            text = text[:3000] + "\n... [truncated]"
        return f"Default config reference (config.default.toml):\n\n{text}"
    except Exception as e:
        return f"Error reading default config: {e}"


def _topic_safety() -> str:
    """Format the live safety configuration."""
    if _safety_config is None:
        return "Safety config not available (not injected at startup)."

    lines = [f"Safety mode: {_safety_config.mode}\n"]

    lines.append(
        f"Commands requiring confirmation ({len(_safety_config.always_confirm)} patterns):"
    )
    for pattern in _safety_config.always_confirm:
        lines.append(f"  {pattern}")

    lines.append(f"\nBlocked commands ({len(_safety_config.blocked)} patterns):")
    for pattern in _safety_config.blocked:
        lines.append(f"  {pattern}")

    lines.append(
        "\nSensitive file paths (read_file requires confirmation):\n"
        "  /.ssh/, /id_rsa, /id_ed25519, /etc/shadow, /etc/sudoers, "
        "/proc/self/environ, .env, /.aws/credentials, /.kube/config, "
        "/.docker/config.json"
    )
    return "\n".join(lines)


# ── Topic registry ────────────────────────────────────────────────────────

_DYNAMIC_TOPICS: dict[str, callable] = {
    "config": _topic_config,
    "config_reference": _topic_config_reference,
    "safety": _topic_safety,
}

VALID_TOPICS = sorted(_static_topic_names() + list(_DYNAMIC_TOPICS.keys()))

# ── Tool definition ──────────────────────────────────────────────────────

DEFINITION = ToolDefinition(
    name="natshell_help",
    description=(
        "Look up NatShell documentation by topic. Use this when the user asks "
        "about NatShell itself — its commands, configuration, available tools, "
        "model setup, safety rules, troubleshooting, profiles, sessions, "
        "plans, plugins, headless mode, MCP server, backups, working memory, "
        "or keyboard shortcuts."
    ),
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "enum": VALID_TOPICS,
                "description": (
                    "The documentation topic to look up. Options: "
                    + ", ".join(VALID_TOPICS)
                ),
            },
        },
        "required": ["topic"],
    },
)


# ── Handler ──────────────────────────────────────────────────────────────


async def natshell_help(topic: str) -> ToolResult:
    """Return documentation for the requested topic."""
    static_text = _load_static_topic(topic)
    if static_text is not None:
        return ToolResult(output=static_text)

    if topic in _DYNAMIC_TOPICS:
        try:
            content = _DYNAMIC_TOPICS[topic]()
            return ToolResult(output=content)
        except Exception as e:
            return ToolResult(
                error=f"Error retrieving {topic}: {e}",
                exit_code=1,
            )

    return ToolResult(
        error=f"Unknown topic: {topic}. Valid topics: {', '.join(VALID_TOPICS)}",
        exit_code=1,
    )
