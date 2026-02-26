# NatShell — Claude Code Build Instructions

## What is this?

NatShell is an agentic TUI that provides a natural language interface to Linux, macOS, and WSL. Users type requests in plain English (e.g., "scan my local network for computers") and a bundled local LLM plans and executes multi-step shell operations to fulfill them.

## Architecture

- **Agent loop**: ReAct pattern — model reasons, calls tools, observes results, repeats
- **Inference**: Bundled llama.cpp via llama-cpp-python (no Ollama dependency), with optional remote API fallback
- **TUI**: Textual framework
- **Safety**: Regex-based command classifier (safe/confirm/blocked)
- **Tools**: execute_shell, read_file, write_file, list_directory, search_files

## Key Design Decisions

1. `execute_shell` runs commands via `bash -c` with the user's real environment — this is intentional, not a sandbox
2. The safety classifier is pattern-based (fast, deterministic), NOT LLM-based
3. System context is gathered once at startup and injected into the system prompt
4. Local inference uses `llama-cpp-python` with `chat_format="chatml-function-calling"` for Hermes/Qwen tool calling
5. The agent loop is async — local inference calls are wrapped in `asyncio.to_thread()` to avoid blocking the TUI
6. Output from commands is truncated to ~4000 chars to fit context windows
7. Platform detection is centralized in `src/natshell/platform.py` (cached `lru_cache`). Use `is_macos()`, `is_wsl()`, `is_linux()` — don't scatter `sys.platform` checks

## Build Order

1. `src/natshell/config.py` — TOML config loading
2. `src/natshell/tools/` — All 5 tools (test standalone)
3. `src/natshell/safety/classifier.py` — Command risk classification
4. `src/natshell/inference/local.py` — llama-cpp-python wrapper
5. `src/natshell/inference/engine.py` + `remote.py` — Abstraction + API backend
6. `src/natshell/agent/context.py` — System info gathering
7. `src/natshell/agent/system_prompt.py` — Prompt construction
8. `src/natshell/agent/loop.py` — ReAct agent loop
9. `src/natshell/app.py` + `src/natshell/ui/` — Textual TUI
10. `src/natshell/__main__.py` — Entry point, model download, wiring

## Tech Stack

- Python 3.11+
- llama-cpp-python (with Vulkan, Metal, or CPU backend)
- Textual >= 1.0
- httpx for remote API
- huggingface-hub for model download
- Platforms: Linux, macOS, WSL

## Model

Default: Qwen3-4B Q4_K_M GGUF (~2.5GB). Supports Hermes-style function calling natively.
Auto-downloaded on first run to ~/.local/share/natshell/models/

## Key Files Reference

- `PROJECT_SCAFFOLD.md` — Full architectural specification with detailed pseudocode for every module
- `src/natshell/platform.py` — Platform detection (macOS/WSL/Linux), used by clipboard, context, system prompt, installer
- `config.default.toml` — Default configuration with all safety patterns (includes macOS-specific: brew, launchctl, diskutil)
- `pyproject.toml` — Dependencies and entry point

## Testing

Run tests with `pytest`. Mock the InferenceEngine for agent loop tests. Tools can be tested directly against the real system (be careful with write_file tests — use /tmp).

## Cross-Platform Notes

- **Clipboard**: macOS uses `pbcopy`/`pbpaste`, WSL uses `clip.exe`/`powershell.exe Get-Clipboard`, Linux uses `xclip`/`xsel`/`wl-copy`
- **System context** (`agent/context.py`): macOS branch uses `sw_vers`, `sysctl`, `vm_stat`, `ifconfig`, `launchctl`; Linux/WSL branch uses `lscpu`, `free`, `ip`, `systemctl`
- **System prompt**: Role string adapts per platform ("macOS" / "Linux (WSL)" / "Linux")
- **Installer** (`install.sh`): Detects macOS via `uname -s`, uses `brew` for packages, `xcode-select` for compiler, Metal for GPU
- **Safety patterns**: macOS-specific patterns for `brew`, `launchctl`, `diskutil` in `config.default.toml`

## Installing for development

```bash
cd natshell
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# For CPU-only llama.cpp:
pip install llama-cpp-python
# For Vulkan (AMD/Linux GPUs):
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python
# For Metal (macOS):
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```
