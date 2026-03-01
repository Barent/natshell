# NatShell

[![PyPI version](https://img.shields.io/pypi/v/natshell)](https://pypi.org/project/natshell/)

Natural language shell interface for Linux, macOS, and WSL — a local-first agentic TUI powered by a bundled LLM.

Type requests in plain English and NatShell plans and executes shell commands to fulfill them, using a ReAct-style agent loop with a small local model (Qwen3-4B via llama.cpp). Supports optional remote inference via Ollama or any OpenAI-compatible API.

## Install

### From PyPI

```bash
pip install natshell              # Remote/Ollama mode (no C++ compiler needed)
pip install natshell[local]       # Includes llama-cpp-python for local inference
```

### From source (recommended for GPU acceleration)

```bash
git clone https://github.com/Barent/natshell.git && cd natshell
bash install.sh
```

The installer handles everything — Python venv, GPU detection (Vulkan/Metal/CPU), llama.cpp build, model download, and Ollama configuration. No sudo required. Missing system dependencies (C++ compiler, clipboard tools, Vulkan headers, etc.) are detected and offered for install automatically.

### Development setup

```bash
git clone https://github.com/Barent/natshell.git && cd natshell
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install llama-cpp-python                # CPU-only
# CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --no-cache-dir  # Vulkan (Linux)
# CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir   # Metal (macOS)
natshell
```

## Usage

```bash
natshell                          # Launch with defaults (local model)
natshell --model ./my-model.gguf  # Use a specific GGUF model
natshell --remote http://host:11434/v1 --remote-model qwen3:4b  # Use Ollama/remote API
natshell --download               # Download the default model and exit
natshell --update                 # Self-update from git and reinstall
natshell --config path/to/config.toml  # Custom config file
natshell --verbose                # Enable debug logging
natshell --headless "list files"  # Single-shot non-interactive mode (stdout pipeable)
natshell --headless --danger-fast "deploy" # Headless with auto-approve confirmations
natshell --mcp                    # Start as MCP server (stdin/stdout JSON-RPC)
```

## Features

### Agent Loop
NatShell uses a ReAct-style agent loop — the model reasons about your request, calls tools (shell commands, file operations, etc.), observes results, and iterates until the task is complete. Up to 15 tool calls per request by default.

### Inference Backends
- **Local**: Bundled llama.cpp via llama-cpp-python. Default model is Qwen3-4B (Q4_K_M, ~2.5 GB), auto-downloaded on first run.
- **Remote**: Any OpenAI-compatible API — Ollama, vLLM, LM Studio, etc.
- **Fallback**: If the remote server is unreachable, NatShell automatically falls back to the local model.
- **Runtime switching**: Switch models on the fly with `/model` commands without restarting.

### GPU Acceleration
- Auto-detects GPUs via vulkaninfo, nvidia-smi, and lspci
- Prefers discrete GPUs over integrated on multi-GPU systems
- Supports Vulkan (Linux/AMD/NVIDIA), Metal (macOS), and CPU fallback
- Prints helpful reinstall instructions if GPU support is missing

### Tools
The agent has access to 9 tools:
- **execute_shell** — Run any shell command via bash
- **read_file** — Read file contents
- **write_file** — Write or append to files (always requires confirmation)
- **edit_file** — Targeted search-and-replace edits (always requires confirmation)
- **run_code** — Execute code snippets in 10 languages (Python, JS, Bash, Ruby, Perl, PHP, C, C++, Rust, Go)
- **list_directory** — List directory contents with sizes and types
- **search_files** — Search file contents (grep) or find files by name
- **git_tool** — Structured git operations (status, diff, log, branch, commit, stash)
- **natshell_help** — Look up NatShell documentation by topic

### TUI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Clear chat and model context |
| `/cmd <command>` | Execute a shell command directly (bypasses AI, respects safety) |
| `/model` | Show current engine and model info |
| `/model list` | List models available on the remote server |
| `/model use <name>` | Switch to a remote model |
| `/model switch` | Switch local GGUF model (opens command palette) |
| `/model local` | Switch back to local model |
| `/model default <name>` | Save default remote model to config |
| `/compact` | Summarize conversation to free context window space |
| `/plan <description>` | Generate a step-by-step plan (PLAN.md) from natural language |
| `/exeplan run PLAN.md` | Execute a previously generated plan |
| `/undo` | Undo the last file edit (restores from backup) |
| `/save [name]` | Save current conversation to a session file |
| `/load <id>` | Load a saved conversation session |
| `/sessions` | List all saved sessions |
| `/keys` | Show keyboard shortcuts |
| `/history` | Show conversation message count |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+C` | Quit |
| `Ctrl+E` | Copy entire chat to clipboard |
| `Ctrl+L` | Clear chat |
| `Ctrl+P` | Command palette (model switching) |
| `Ctrl+Y` | Copy selected text |

### Backup & Undo
Every file edit creates a timestamped backup in `~/.local/share/natshell/backups/`. Use `/undo` to restore the most recent edit. Backups are pruned to 10 per file by default.

### Session Persistence
Save and restore conversations with `/save`, `/load`, and `/sessions`. Sessions are stored as JSON in `~/.local/share/natshell/sessions/`.

### Headless Mode
Run NatShell non-interactively with `--headless "prompt"`. Response text goes to stdout (pipeable), everything else to stderr. Use `--danger-fast` to auto-approve confirmations.

### MCP Server
Run NatShell as an MCP (Model Context Protocol) server with `--mcp`. Exposes all tools via JSON-RPC over stdin/stdout for integration with other AI tools.

### Plugin System
Extend NatShell with custom tools by placing Python files in `~/.config/natshell/plugins/`. Each plugin defines a `register()` function that receives the tool registry.

### Prompt Caching
System prompt tokens are cached across requests to reduce latency on local inference. Cache is invalidated when the system prompt changes.

### Diff Preview
File edits show a unified diff preview in the confirmation dialog, making it easier to review changes before approving.

## Safety

Commands are classified into three risk levels by a fast, deterministic regex-based classifier:

- **Safe** — auto-executed (ls, cat, df, grep, etc.)
- **Confirm** — requires user approval (rm, sudo, apt install, docker rm, iptables, etc.)
- **Blocked** — never executed (fork bombs, rm -rf /, destructive dd/mkfs to disks, etc.)

Additional safety features:
- Commands chained with `&&`, `||`, `;`, `&`, or `|` are split and each sub-command is classified independently
- Subshell expressions (`$(...)`) and backtick expansions are flagged for confirmation
- Sensitive file paths (SSH keys, `/etc/shadow`, `.env`) require confirmation for read_file
- Sensitive environment variables (API keys, tokens, credentials) are filtered from subprocesses
- Sudo passwords are cached for 5 minutes with automatic expiry
- LLM output is escaped to prevent Rich markup injection in the TUI
- API keys sent over plaintext HTTP trigger a warning

Safety modes are configurable: `confirm` (default), `warn`, or `yolo`. All patterns are customizable in config.

## Configuration

Default configuration is bundled with the package. Copy it to `~/.config/natshell/config.toml` to customize:

```bash
python -c "from pathlib import Path; import natshell; p = Path(natshell.__file__).parent / 'config.default.toml'; print(p.read_text())" > ~/.config/natshell/config.toml
```

Or if installed from source, copy `src/natshell/config.default.toml` directly.

### Sections

- **[model]** — GGUF path, HuggingFace repo/file for auto-download, context size (0 = auto-detect from model), GPU layers, device selection
- **[remote]** — URL, model name, API key for OpenAI-compatible endpoints
- **[ollama]** — Ollama server URL and default model (used by `/model list` and `/model use`)
- **[agent]** — max steps (15), temperature (0.3), max tokens (2048)
- **[safety]** — mode, confirmation regex patterns, blocked regex patterns
- **[backup]** — backup directory, max backups per file
- **[mcp]** — MCP server safety mode
- **[ui]** — theme (dark/light)

### Environment Variables

- `NATSHELL_API_KEY` — API key for remote inference (alternative to storing in config file)

## Cross-Platform Support

| Feature | Linux | macOS | WSL |
|---------|-------|-------|-----|
| Shell execution | bash | bash | bash |
| GPU | Vulkan | Metal | Vulkan |
| Clipboard | wl-copy, xclip, xsel | pbcopy | clip.exe |
| Package manager | apt, dnf, pacman, zypper, apk, emerge | brew | apt |
| System context | lscpu, free, ip, systemctl | sw_vers, sysctl, vm_stat, ifconfig | lscpu, free, ip |
| Safety patterns | Linux + generic | macOS-specific (brew, launchctl, diskutil) | Linux + generic |

Clipboard auto-detects the best backend with fallback to OSC52 terminal escape sequences for remote/VM sessions.

## Architecture

```
src/natshell/
├── __main__.py              # CLI entry point, model download, engine wiring
├── app.py                   # Textual TUI application
├── backup.py                # Pre-edit backup system with undo support
├── commands.py              # Slash command dispatch (refactored from app.py)
├── config.py                # TOML config loading with env var support
├── config.default.toml      # Bundled default configuration
├── gpu.py                   # GPU detection (vulkaninfo/nvidia-smi/lspci)
├── headless.py              # Non-interactive single-shot CLI mode
├── mcp_server.py            # MCP server (JSON-RPC over stdin/stdout)
├── model_manager.py         # Model discovery, download, and switching
├── platform.py              # Platform detection (Linux/macOS/WSL)
├── plugins.py               # Plugin system for custom tools
├── session.py               # Conversation session persistence
├── agent/
│   ├── loop.py              # ReAct agent loop with safety checks
│   ├── system_prompt.py     # Platform-aware system prompt builder
│   ├── context.py           # System info gathering (CPU, RAM, disk, network, etc.)
│   ├── context_manager.py   # Conversation context window management
│   ├── plan.py              # Plan generation and markdown parsing
│   └── plan_executor.py     # Step-by-step plan execution engine
├── inference/
│   ├── engine.py            # Inference engine protocol + CompletionResult types
│   ├── local.py             # llama-cpp-python backend with GPU support
│   ├── remote.py            # OpenAI-compatible API backend (httpx)
│   └── ollama.py            # Ollama server discovery and model listing
├── safety/
│   └── classifier.py        # Regex-based command risk classifier
├── tools/
│   ├── registry.py          # Tool registration and dispatch
│   ├── execute_shell.py     # Shell execution with sudo, env filtering, truncation
│   ├── read_file.py         # File reading
│   ├── write_file.py        # File writing
│   ├── edit_file.py         # Targeted search-and-replace edits
│   ├── run_code.py          # Code execution in 10 languages
│   ├── list_directory.py    # Directory listing
│   ├── search_files.py      # Text/file search
│   ├── git_tool.py          # Structured git operations
│   ├── limits.py            # Context-aware output truncation limits
│   └── natshell_help.py     # Self-documentation by topic
└── ui/
    ├── widgets.py           # TUI widgets (messages, command blocks, modals)
    ├── commands.py          # Command palette providers
    ├── clipboard.py         # Cross-platform clipboard integration
    ├── escape.py            # Rich markup escaping utilities
    └── styles.tcss          # Textual CSS stylesheet
```

## Development

```bash
source .venv/bin/activate
pytest                    # Run tests (641+ tests)
ruff check src/ tests/    # Lint
```

## License

MIT
