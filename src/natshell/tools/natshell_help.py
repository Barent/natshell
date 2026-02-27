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


# ── Static topic content ──────────────────────────────────────────────────

_STATIC_TOPICS: dict[str, str] = {
    "overview": (
        "NatShell is an agentic TUI that provides a natural language interface "
        "to Linux, macOS, and WSL. Users type requests in plain English and a "
        "bundled local LLM plans and executes multi-step shell operations. "
        "It uses the ReAct agent pattern (reason → act → observe), with a "
        "bundled llama.cpp backend, optional Ollama or remote API fallback, "
        "and a Textual-based TUI. The safety classifier is regex-based and "
        "deterministic. Config file: ~/.config/natshell/config.toml"
    ),
    "commands": (
        "Available slash commands:\n"
        "  /help                  — Show available commands and keybindings\n"
        "  /clear                 — Clear chat history and model context\n"
        "  /model                 — Show current engine/model info\n"
        "  /model list            — List models on the remote server\n"
        "  /model use <name>      — Switch to a remote model\n"
        "  /model switch          — Switch to a different local .gguf model\n"
        "  /model local           — Switch back to the local model\n"
        "  /model default <name>  — Save default remote model to config\n"
        "\n"
        "Keybindings:\n"
        "  Enter        — Send message\n"
        "  Ctrl+P       — Open model switcher (command palette)\n"
        "  Ctrl+C       — Copy selected text / cancel\n"
        "  Ctrl+Shift+C — Copy entire chat"
    ),
    "tools": (
        "Agent tools available during operation:\n"
        "  execute_shell  — Run a bash command and return output (with safety classification)\n"
        "  read_file      — Read file contents (truncated at 200 lines by default)\n"
        "  write_file     — Write/append to a file (always requires confirmation)\n"
        "  list_directory — List directory contents with sizes and types\n"
        "  search_files   — Text search (grep) or file search (find) in a directory\n"
        "  natshell_help  — Look up NatShell documentation by topic (this tool)"
    ),
    "models": (
        "Model configuration:\n"
        "  Default: Qwen3-4B Q4_K_M GGUF (~2.5 GB), auto-downloaded on first run.\n"
        "  Model storage: ~/.local/share/natshell/models/\n"
        "\n"
        "Local model config ([model] section in config.toml):\n"
        "  path         — Path to .gguf file, or 'auto' for default download\n"
        "  n_ctx        — Context window (0 = auto: 4096 for ≤4B, 8192 for larger)\n"
        "  n_gpu_layers — GPU layers (-1 = all, 0 = CPU only)\n"
        "  main_gpu     — GPU device index (-1 = auto-detect)\n"
        "\n"
        "Remote/Ollama config:\n"
        "  [remote] section: url, model, api_key\n"
        "  [ollama] section: url, default_model\n"
        "  CLI flags: --remote <url>, --remote-model <name>\n"
        "  To install Ollama: curl -fsSL https://ollama.com/install.sh | sh\n"
        "  To pull a model: ollama pull <model-name>"
    ),
    "troubleshooting": (
        "Common issues:\n"
        "  'GPU offloading requested but not supported'\n"
        "    → Reinstall llama-cpp-python with GPU flags:\n"
        "      CMAKE_ARGS=\"-DGGML_VULKAN=on\" pip install llama-cpp-python "
        "--no-binary llama-cpp-python --no-cache-dir\n"
        "\n"
        "  'Remote server unreachable'\n"
        "    → Check the URL in [ollama] or [remote] config section.\n"
        "      Ensure the server is running (ollama serve, or check the API host).\n"
        "\n"
        "  'No local model found'\n"
        "    → Run: natshell --download  (downloads the default Qwen3-4B model)\n"
        "\n"
        "  Slow inference / high CPU\n"
        "    → Check n_gpu_layers in config (set to -1 to offload all layers to GPU)\n"
        "    → Set n_threads to match physical core count\n"
        "\n"
        "  Self-update: natshell --update (git installs only)"
    ),
}

# ── Dynamic topic handlers ────────────────────────────────────────────────


def _topic_config() -> str:
    """Read the user's config.toml and return its contents."""
    user_config = Path.home() / ".config" / "natshell" / "config.toml"
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
    default_path = Path(__file__).parent.parent.parent.parent / "config.default.toml"
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

    lines.append(f"Commands requiring confirmation ({len(_safety_config.always_confirm)} patterns):")
    for pattern in _safety_config.always_confirm:
        lines.append(f"  {pattern}")

    lines.append(f"\nBlocked commands ({len(_safety_config.blocked)} patterns):")
    for pattern in _safety_config.blocked:
        lines.append(f"  {pattern}")

    lines.append(
        "\nSensitive file paths (read_file requires confirmation):\n"
        "  /.ssh/, /id_rsa, /id_ed25519, /etc/shadow, /etc/sudoers, "
        "/proc/self/environ, .env"
    )
    return "\n".join(lines)


# ── Topic registry ────────────────────────────────────────────────────────

_DYNAMIC_TOPICS: dict[str, callable] = {
    "config": _topic_config,
    "config_reference": _topic_config_reference,
    "safety": _topic_safety,
}

VALID_TOPICS = sorted(list(_STATIC_TOPICS.keys()) + list(_DYNAMIC_TOPICS.keys()))

# ── Tool definition ──────────────────────────────────────────────────────

DEFINITION = ToolDefinition(
    name="natshell_help",
    description=(
        "Look up NatShell documentation by topic. Use this when the user asks "
        "about NatShell itself — its commands, configuration, available tools, "
        "model setup, safety rules, or troubleshooting."
    ),
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "enum": VALID_TOPICS,
                "description": (
                    "The documentation topic to look up. "
                    "Options: " + ", ".join(VALID_TOPICS)
                ),
            },
        },
        "required": ["topic"],
    },
)


# ── Handler ──────────────────────────────────────────────────────────────

async def natshell_help(topic: str) -> ToolResult:
    """Return documentation for the requested topic."""
    if topic in _STATIC_TOPICS:
        return ToolResult(output=_STATIC_TOPICS[topic])

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
