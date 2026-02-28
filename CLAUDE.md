# NatShell — Claude Code Build Instructions

## What is this?

NatShell is an agentic TUI that provides a natural language interface to Linux, macOS, and WSL. Users type requests in plain English (e.g., "scan my local network for computers") and a bundled local LLM plans and executes multi-step shell operations to fulfill them. NatShell also serves as a coding assistant — it can read, edit, and write source files, and execute code snippets in 10 languages.

## Architecture

- **Agent loop**: ReAct pattern — model reasons, calls tools, observes results, repeats (max 15 steps)
- **Inference**: Bundled llama.cpp via llama-cpp-python, with optional Ollama or remote API fallback. Runtime engine swapping supported.
- **TUI**: Textual framework with custom widgets, command palette, clipboard integration
- **Safety**: Regex-based command classifier (safe/confirm/blocked) with command chaining detection, sensitive path gating, and env var filtering
- **Tools**: execute_shell, read_file, write_file, edit_file, run_code, list_directory, search_files, natshell_help

## Key Design Decisions

1. `execute_shell` runs commands via `bash -c` with the user's real environment — this is intentional, not a sandbox
2. The safety classifier is pattern-based (fast, deterministic), NOT LLM-based
3. System context is gathered once at startup and injected into the system prompt
4. Local inference uses `llama-cpp-python` — tool definitions are injected as plain text (not llama-cpp-python's built-in tool format) because Qwen3 outputs `<tool_call>` XML tags that are parsed manually
5. The agent loop is async — local inference calls are wrapped in `asyncio.to_thread()` to avoid blocking the TUI
6. Output from commands is truncated to ~4000 chars to fit context windows
7. Platform detection is centralized in `src/natshell/platform.py` (cached `lru_cache`). Use `is_macos()`, `is_wsl()`, `is_linux()` — don't scatter `sys.platform` checks
8. GPU detection is in `src/natshell/gpu.py` (cached `lru_cache`). Tries vulkaninfo, nvidia-smi, lspci in order. Prefers discrete over integrated GPUs.
9. The system prompt presents NatShell as both a system administration and coding assistant, with dedicated guidance for code editing (use `edit_file` for targeted changes, `write_file` for new files) and `natshell_help` for self-documentation.
10. Engine preference is persisted via `[engine]` section in config.toml (`preferred = "auto" | "local" | "remote"`), allowing startup behavior to respect the user's last engine choice.

## Modules

### Core
- `src/natshell/__main__.py` — CLI entry point with argparse, model download, engine wiring, GPU check
- `src/natshell/app.py` — Textual TUI application with slash commands, confirmation dialogs, sudo prompts, model switching. `/plan` generates a PLAN.md via the agent loop with a specialized prompt, then previews it; feeds into `/exeplan run PLAN.md`.
- `src/natshell/config.py` — TOML config loading with defaults, env var API key support (`NATSHELL_API_KEY`), file permission warnings, engine preference persistence (`EngineConfig`, `save_engine_preference()`)

### Agent
- `src/natshell/agent/loop.py` — ReAct agent loop with safety classification, sudo password retry, engine fallback
- `src/natshell/agent/system_prompt.py` — Platform-aware system prompt with behavior rules, coding/development guidance, natshell_help integration, and `/no_think` directive
- `src/natshell/agent/context.py` — System context gathering (CPU, RAM, disk, network, services, containers, tools) with per-platform commands

### Inference
- `src/natshell/inference/engine.py` — Protocol types: `CompletionResult`, `ToolCall`, `EngineInfo`
- `src/natshell/inference/local.py` — llama-cpp-python backend with auto context sizing (inferred from model filename), GPU device selection, `<tool_call>` XML parsing, `<think>` tag stripping
- `src/natshell/inference/remote.py` — OpenAI-compatible API backend with httpx, HTTPS warning for plaintext API keys
- `src/natshell/inference/ollama.py` — Ollama server ping, model listing, URL normalization

### Tools
- `src/natshell/tools/registry.py` — Tool registration and dispatch with OpenAI-compatible schemas (8 tools)
- `src/natshell/tools/execute_shell.py` — Shell execution with sudo password caching (5-min timeout), sensitive env var filtering, output truncation, process group isolation
- `src/natshell/tools/read_file.py` — File reading with line limits
- `src/natshell/tools/write_file.py` — File writing (always requires confirmation)
- `src/natshell/tools/edit_file.py` — Targeted search-and-replace edits to existing files (unique match required, always requires confirmation)
- `src/natshell/tools/run_code.py` — Execute code snippets in 10 languages (python, javascript, bash, ruby, perl, php, c, cpp, rust, go). Handles temp file creation, compilation, execution, and cleanup. Always requires confirmation.
- `src/natshell/tools/list_directory.py` — Directory listing with sizes, types, hidden file toggle
- `src/natshell/tools/search_files.py` — Text search (grep) and file search (find)
- `src/natshell/tools/natshell_help.py` — Self-documentation tool with static topics (overview, commands, tools, models, troubleshooting) and dynamic topics (config, config_reference, safety). Accepts injected `SafetyConfig` for live safety pattern reporting.

### Safety
- `src/natshell/safety/classifier.py` — Command risk classifier. Splits chained commands (`&&`, `||`, `;`, `&`, `|`) and classifies each sub-command. Detects subshell/backtick expansion. Sensitive file path gating for read_file. Three modes: confirm (default), warn, yolo.

### UI
- `src/natshell/ui/widgets.py` — Custom Textual widgets: HistoryInput (shell-like up/down arrow input history with draft save/restore), LogoBanner, CopyableMessage (UserMessage, AssistantMessage, PlanningMessage, BlockedMessage, SystemMessage, HelpMessage), CommandBlock, ThinkingIndicator, ConfirmScreen, SudoPasswordScreen. Rich markup escaping (`_escape()`) on all untrusted content.
- `src/natshell/ui/commands.py` — Command palette provider for local model switching
- `src/natshell/ui/clipboard.py` — Cross-platform clipboard: macOS (pbcopy), WSL (clip.exe), Wayland (wl-copy), X11 (xclip/xsel), OSC52 fallback
- `src/natshell/ui/styles.tcss` — Textual CSS stylesheet

### Utilities
- `src/natshell/platform.py` — Platform detection (macOS/WSL/Linux), used by clipboard, context, system prompt, installer
- `src/natshell/gpu.py` — GPU hardware detection (vulkaninfo/nvidia-smi/lspci), best-device selection, vendor classification

## Security Hardening

These security features were added in the security refactor:

1. **Rich markup escaping** — LLM output and command results are escaped via `_escape()` before rendering to prevent markup injection
2. **Command chaining detection** — Safety classifier splits on shell operators and classifies each sub-command independently; also checks full command against patterns first (for fork bombs and pipe-spanning patterns)
3. **Sudo password timeout** — Cached password expires after 5 minutes via `time.monotonic()` timestamp
4. **Environment variable filtering** — Sensitive env vars (AWS keys, GitHub tokens, API keys, etc.) are stripped from subprocess environments
5. **HTTPS warning** — `RemoteEngine` logs a warning when API keys are sent over plaintext HTTP to non-localhost hosts
6. **Sensitive file path gating** — `read_file` requires confirmation for SSH keys, `/etc/shadow`, `.env`, etc.
7. **Sudo regex correctness** — `_SUDO_RE.sub()` uses `count=1` to only replace the first `sudo` occurrence
8. **Config permissions warning** — Warns if config file containing an API key has permissive permissions (world/group readable)
9. **Log redaction** — Sudo password plumbing is redacted from verbose log output

## Tech Stack

- Python 3.11+
- llama-cpp-python (with Vulkan, Metal, or CPU backend)
- Textual >= 1.0
- httpx for remote API
- huggingface-hub for model download
- Platforms: Linux, macOS, WSL

## Model

Default: Qwen3-4B Q4_K_M GGUF (~2.5GB). Supports tool calling natively via `<tool_call>` XML tags.
Auto-downloaded on first run to `~/.local/share/natshell/models/`
Context size is auto-detected from model filename (4B → 4096, 8B → 8192) when `n_ctx = 0`.

## Key Files Reference

- `PROJECT_SCAFFOLD.md` — Original architectural specification (reference only — implementation has evolved)
- `src/natshell/config.default.toml` — Default configuration with all safety patterns (23 confirm + 8 blocked, includes macOS-specific: brew, launchctl, diskutil). Bundled as package data.
- `pyproject.toml` — Dependencies and entry point
- `install.sh` — Cross-platform installer with GPU detection, Ollama setup, model download

## Testing

Run tests with `pytest` (267 tests across 13 test files). Mock the InferenceEngine for agent loop tests. Tools can be tested directly against the real system (be careful with write_file tests — use /tmp).

Test files:
- `test_agent.py` — Agent loop and event handling
- `test_safety.py` — Safety classifier patterns, chaining, file sensitivity
- `test_tools.py` — Individual tool execution
- `test_coding_tools.py` — edit_file and run_code tool tests
- `test_natshell_help.py` — natshell_help static/dynamic topic tests
- `test_history_input.py` — HistoryInput widget history navigation
- `test_engine_preference.py` — Engine preference persistence and loading
- `test_clipboard.py` — Clipboard backends
- `test_platform.py` — Platform detection
- `test_engine_swap.py` — Model switching and engine swapping
- `test_ollama.py` — Ollama/OpenAI API parsing
- `test_ollama_config.py` — Config file persistence
- `test_slash_commands.py` — TUI slash commands

## Cross-Platform Notes

- **Clipboard**: macOS uses `pbcopy`/`pbpaste`, WSL uses `clip.exe`/`powershell.exe Get-Clipboard`, Linux uses `wl-copy`/`xclip`/`xsel` with OSC52 fallback
- **System context** (`agent/context.py`): macOS branch uses `sw_vers`, `sysctl`, `vm_stat`, `ifconfig`, `launchctl`; Linux/WSL branch uses `lscpu`, `free`, `ip`, `systemctl`
- **System prompt**: Role string adapts per platform ("macOS" / "Linux (WSL)" / "Linux")
- **Installer** (`install.sh`): Detects macOS via `uname -s`, uses `brew` for packages, `xcode-select` for compiler, Metal for GPU
- **Safety patterns**: macOS-specific patterns for `brew`, `launchctl`, `diskutil` in `config.default.toml`
- **Package managers**: Auto-detected — brew, apt, dnf, yum, pacman, zypper, apk, emerge, rpm-ostree

## Installing for development

```bash
cd natshell
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# For CPU-only llama.cpp:
pip install llama-cpp-python
# For Vulkan (AMD/Linux GPUs):
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --no-binary llama-cpp-python --no-cache-dir
# For Metal (macOS):
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-binary llama-cpp-python --no-cache-dir
```

**Important**: Always use `--no-binary llama-cpp-python --no-cache-dir` when installing with GPU flags, otherwise pip may use a cached CPU-only wheel.
