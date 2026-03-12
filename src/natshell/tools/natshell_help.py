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
        "It also serves as a coding assistant — it can read, edit, and write "
        "source files, and execute code snippets in 10 languages.\n\n"
        "It uses the ReAct agent pattern (reason → act → observe), with a "
        "bundled llama.cpp backend, optional Ollama or remote API fallback, "
        "and a Textual-based TUI. The safety classifier is regex-based and "
        "deterministic. Config file: ~/.config/natshell/config.toml"
    ),
    "commands": (
        "Available slash commands:\n"
        "  /help                  — Show available commands and keybindings\n"
        "  /clear                 — Clear chat history and model context\n"
        "  /compact               — Compact context, keeping key facts\n"
        "  /cmd <command>         — Execute a shell command directly\n"
        "  /plan <description>    — Generate a multi-step plan\n"
        "  /exeplan <file>        — Preview a multi-step plan\n"
        "  /exeplan run <file>    — Execute all plan steps\n"
        "  /model                 — Show current engine/model info\n"
        "  /model list            — List models on the remote server\n"
        "  /model use <name>      — Switch to a remote model\n"
        "  /model switch          — Switch to a different local .gguf model\n"
        "  /model local           — Switch back to the local model\n"
        "  /model default <name>  — Save default remote model to config\n"
        "  /model download        — Download a bundled model tier\n"
        "  /profile               — List configuration profiles\n"
        "  /profile <name>        — Apply a configuration profile\n"
        "  /keys                  — Show keyboard shortcuts\n"
        "  /history               — Show conversation context size\n"
        "  /undo                  — Undo last file edit or write\n"
        "  /save [name]           — Save current session\n"
        "  /load [id]             — Load a saved session\n"
        "  /sessions              — List saved sessions"
    ),
    "tools": (
        "Agent tools available during operation:\n"
        "  execute_shell  — Run a bash command and return output "
        "(with safety classification)\n"
        "  read_file      — Read file contents (line limit scales with context "
        "window)\n"
        "  write_file     — Write/append to a file (always requires "
        "confirmation)\n"
        "  edit_file      — Search-and-replace edit in an existing file "
        "(requires confirmation, unique match)\n"
        "  list_directory — List directory contents with sizes and types\n"
        "  search_files   — Text search (grep) or file search (find) in a "
        "directory\n"
        "  run_code       — Execute code snippets in 10 languages (python, "
        "javascript, bash, ruby, perl, php, c, cpp, rust, go)\n"
        "  git_tool       — Structured git operations (status, diff, log, "
        "branch, commit, stash)\n"
        "  fetch_url      — Fetch a URL and return its content (blocks "
        "private/internal IPs)\n"
        "  kiwix_search   — Search a local kiwix-serve instance for offline "
        "Wikipedia and documentation\n"
        "  natshell_help  — Look up NatShell documentation by topic "
        "(this tool)\n"
        "  update_config  — Update a NatShell config value (saves to disk + "
        "applies live)"
    ),
    "models": (
        "Model configuration:\n"
        "  Three local model tiers:\n"
        "    Light:    Qwen3-4B       (~2.5 GB, low RAM)\n"
        "    Standard: Qwen3-8B       (~5 GB, general purpose)\n"
        "    Enhanced: Mistral Nemo 12B (~7.5 GB, 128K context) — recommended\n"
        "\n"
        "  Default: Qwen3-4B Q4_K_M GGUF, auto-downloaded on first run.\n"
        "  Model storage: ~/.local/share/natshell/models/\n"
        "  Download after setup: /model download <tier>  (light, standard, "
        "enhanced)\n"
        "\n"
        "Local model config ([model] section in config.toml):\n"
        "  path         — Path to .gguf file, or 'auto' for default download\n"
        "  n_ctx        — Context window (0 = auto: 4096 for ≤4B, 32768 for "
        "Mistral Nemo)\n"
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
        '      CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python '
        "--no-binary llama-cpp-python --no-cache-dir\n"
        "\n"
        "  'Remote server unreachable'\n"
        "    → Check the URL in [ollama] or [remote] config section.\n"
        "      Ensure the server is running (ollama serve, or check the API "
        "host).\n"
        "\n"
        "  'No local model found'\n"
        "    → Run: natshell --download  (downloads the default model)\n"
        "    → Or use /model download <tier> from within NatShell\n"
        "\n"
        "  Slow inference / high CPU\n"
        "    → Check n_gpu_layers in config (set to -1 to offload all layers "
        "to GPU)\n"
        "    → Set n_threads to match physical core count\n"
        "\n"
        "  Self-update: natshell --update (git installs only)"
    ),
    # ── New topics ────────────────────────────────────────────────────
    "kiwix": (
        "kiwix_search — Offline Wikipedia and documentation search\n\n"
        "The kiwix_search tool queries a local kiwix-serve instance, which "
        "serves ZIM archive files (offline copies of Wikipedia, Stack "
        "Overflow, documentation sets, and more) without internet access.\n\n"
        "Requirements:\n"
        "  kiwix-serve must be running before using this tool.\n"
        "  Start it with: kiwix-serve /path/to/file.zim\n"
        "  Download ZIM files from: https://download.kiwix.org/\n\n"
        "Parameters:\n"
        "  query          — Search query (required)\n"
        "  book           — Filter to a specific ZIM book by name "
        "(e.g. 'wikipedia_en_mini')\n"
        "  results        — Number of results (default 5, max 20)\n"
        "  fetch_article  — If true, fetch and return full text of the top "
        "result\n\n"
        "Example usage:\n"
        "  'search kiwix for Albert Einstein'\n"
        "  'what does Wikipedia say about the Eiffel Tower' "
        "(with fetch_article: true)\n"
        "  'look up Python decorators in the documentation'\n\n"
        "Configuration:\n"
        "  Default URL: http://localhost:8888\n"
        "  To change: update_config kiwix.url http://myserver:8888\n"
        "  Or set in config.toml:\n"
        "    [kiwix]\n"
        "    url = \"http://localhost:8888\""
    ),
    "getting_started": (
        "Getting started with NatShell:\n\n"
        "  1. First run — NatShell runs a setup wizard to pick a model tier:\n"
        "     Light (4B), Standard (8B), or Enhanced (12B).\n"
        "     You can also choose Remote only (Ollama/API) or Skip.\n\n"
        "  2. Download models later — use /model download from within "
        "NatShell:\n"
        "     /model download              — show tiers and download status\n"
        "     /model download standard     — download the 8B model\n\n"
        "  3. Connect to Ollama — set the URL in config.toml:\n"
        "     [ollama]\n"
        "     url = \"http://localhost:11434\"\n"
        "     default_model = \"qwen3:8b\"\n\n"
        "  4. Basic usage — type a request in plain English:\n"
        "     \"scan my local network for computers\"\n"
        "     \"find all Python files larger than 1MB\"\n"
        "     \"edit config.py and change the timeout to 30\"\n\n"
        "  5. Use /help to see all commands, or ask about any topic:\n"
        "     getting_started, commands, tools, models, profiles, "
        "prompt_customization, sessions, plans, plugins, headless, mcp, "
        "backup, keyboard_shortcuts, safety, config, troubleshooting"
    ),
    "profiles": (
        "Configuration profiles let you switch between different settings "
        "quickly.\n\n"
        "Commands:\n"
        "  /profile           — List available profiles\n"
        "  /profile <name>    — Apply a profile\n\n"
        "Defining profiles in config.toml:\n"
        "  [profiles.coding]\n"
        "  engine = \"remote\"        # Switch engine (local/remote)\n"
        "  model = \"qwen3:30b\"     # Remote model name\n"
        "  n_ctx = 32768            # Context window override\n"
        "  temperature = 0.3        # Lower = more deterministic\n"
        "  n_gpu_layers = -1        # GPU layers override\n\n"
        "  [profiles.creative]\n"
        "  engine = \"local\"\n"
        "  temperature = 0.9\n\n"
        "Profiles can override: engine, model, url, api_key, n_ctx, "
        "temperature, n_gpu_layers. Only specified fields are changed."
    ),
    "sessions": (
        "Session persistence lets you save and resume conversations.\n\n"
        "Commands:\n"
        "  /save [name]   — Save current conversation (optional name)\n"
        "  /load [id]     — Load a saved session by ID or pick from list\n"
        "  /sessions      — List all saved sessions\n\n"
        "Details:\n"
        "  Storage: ~/.local/share/natshell/sessions/\n"
        "  Format: JSON with conversation history and metadata\n"
        "  Size limit: 10 MB per session (configurable)\n"
        "  Session IDs: 32-character hex (UUID format)\n"
        "  Directory permissions: 0o700 (user-only access)"
    ),
    "plans": (
        "Plans let you break complex tasks into structured steps.\n\n"
        "Commands:\n"
        "  /plan <description>    — Generate a plan from a description\n"
        "  /exeplan <file>        — Preview a plan file (shows steps)\n"
        "  /exeplan run <file>    — Execute all steps sequentially\n\n"
        "Plan file format (Markdown):\n"
        "  ## Step 1: Description\n"
        "  Details of what to do in this step.\n\n"
        "  ## Step 2: Another step\n"
        "  More details here.\n\n"
        "Each step is executed with a dedicated agent budget. The agent "
        "reads the step description and works autonomously to complete it "
        "before moving to the next step."
    ),
    "plugins": (
        "Plugins let you add custom tools to NatShell.\n\n"
        "Plugin directory: ~/.config/natshell/plugins/\n"
        "Each .py file must define a register(registry) function.\n\n"
        "Example plugin (~/.config/natshell/plugins/hello.py):\n\n"
        "  from natshell.tools.registry import ToolDefinition, ToolResult\n\n"
        "  def register(registry):\n"
        "      registry.register(\n"
        "          ToolDefinition(\n"
        "              name=\"hello\",\n"
        "              description=\"Say hello\",\n"
        "              parameters={\"type\": \"object\", \"properties\": {}},\n"
        "          ),\n"
        "          handler=hello_handler,\n"
        "      )\n\n"
        "  async def hello_handler(**kwargs):\n"
        "      return ToolResult(output=\"Hello from plugin!\")\n\n"
        "Plugins are loaded at startup. Restart NatShell after adding new "
        "plugins."
    ),
    "headless": (
        "Headless mode runs NatShell non-interactively for scripting.\n\n"
        "Usage:\n"
        "  natshell --headless \"your prompt here\"\n\n"
        "Options:\n"
        "  --danger-fast    Auto-approve all confirmations (use with caution)\n\n"
        "Output:\n"
        "  Response text → stdout\n"
        "  Diagnostics   → stderr\n\n"
        "Exit codes:\n"
        "  0 — Success\n"
        "  1 — Error (agent failure, tool error, etc.)\n\n"
        "Examples:\n"
        "  natshell --headless \"list files in /tmp\" > output.txt\n"
        "  natshell --headless --danger-fast \"update packages\"\n"
        "  echo $(natshell --headless \"what is my IP address\")"
    ),
    "mcp": (
        "MCP (Model Context Protocol) server mode exposes NatShell's tools "
        "via JSON-RPC.\n\n"
        "Usage:\n"
        "  natshell --mcp\n\n"
        "Protocol: JSON-RPC over stdin/stdout\n"
        "All NatShell tools are available as MCP methods.\n\n"
        "Safety configuration:\n"
        "  [mcp] section in config.toml controls safety mode.\n"
        "  The same safety classifier applies as in the TUI.\n\n"
        "This allows external editors or AI agents to use NatShell's "
        "tools (execute_shell, read_file, edit_file, etc.) as an MCP "
        "tool provider."
    ),
    "backup": (
        "NatShell automatically backs up files before edits.\n\n"
        "How it works:\n"
        "  - Before every edit_file or write_file, a timestamped copy is "
        "saved\n"
        "  - Backups are stored in ~/.local/share/natshell/backups/\n"
        "  - Directory permissions: 0o700 (user-only access)\n"
        "  - Symlinks are refused (security measure)\n\n"
        "Commands:\n"
        "  /undo    — Restore the most recent backup\n\n"
        "Configuration:\n"
        "  Backup pruning keeps a limited number of backups per file.\n"
        "  Old backups are automatically cleaned up."
    ),
    "prompt_customization": (
        "Customize NatShell's system prompt via the [prompt] section in "
        "~/.config/natshell/config.toml.\n\n"
        "Available keys:\n"
        "  persona             — Replace the default role description\n"
        "  extra_instructions  — Append extra instructions to the prompt\n\n"
        "Example config:\n"
        "  [prompt]\n"
        '  persona = "expert Python developer and DevOps engineer"\n'
        '  extra_instructions = "Always suggest the simplest solution first"\n\n'
        "Details:\n"
        "  - 'persona' replaces the role in the opening line: \"You are NatShell, "
        "a {persona}...\"\n"
        "  - 'extra_instructions' is appended as an \"Additional Instructions\" "
        "section at the end of the system prompt.\n"
        "  - Core safety rules, behavior rules, and code editing guidance are "
        "always included regardless of customization.\n"
        "  - Changes take effect on the next message (prompt is rebuilt each "
        "turn).\n\n"
        "You can also set these at runtime:\n"
        '  update_config section="prompt" key="persona" '
        'value="senior Rust developer"\n'
        '  update_config section="prompt" key="extra_instructions" '
        'value="Prefer functional style"'
    ),
    "keyboard_shortcuts": (
        "Keyboard shortcuts:\n"
        "  Enter          — Send message\n"
        "  Ctrl+C         — Copy selected text / cancel current operation\n"
        "  Ctrl+E         — Copy entire chat to clipboard\n"
        "  Ctrl+L         — Clear screen\n"
        "  Ctrl+P         — Open model switcher (command palette)\n"
        "  Ctrl+Y         — Copy Textual selection to clipboard\n"
        "  Ctrl+Shift+C   — Copy entire chat\n"
        "  Ctrl+Shift+V   — Paste from clipboard\n"
        "  Shift+drag     — Select text in terminal\n"
        "  Up/Down arrows — Navigate input history\n"
        "  Click 📋       — Copy individual message"
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

VALID_TOPICS = sorted(list(_STATIC_TOPICS.keys()) + list(_DYNAMIC_TOPICS.keys()))

# ── Tool definition ──────────────────────────────────────────────────────

DEFINITION = ToolDefinition(
    name="natshell_help",
    description=(
        "Look up NatShell documentation by topic. Use this when the user asks "
        "about NatShell itself — its commands, configuration, available tools, "
        "model setup, safety rules, troubleshooting, profiles, sessions, "
        "plans, plugins, headless mode, MCP server, backups, or keyboard "
        "shortcuts."
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
