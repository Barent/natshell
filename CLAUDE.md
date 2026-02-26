# NatShell — Claude Code Build Instructions

## What is this?

NatShell is an agentic TUI that provides a natural language interface to Linux. Users type requests in plain English (e.g., "scan my local network for computers") and a bundled local LLM plans and executes multi-step shell operations to fulfill them.

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
- llama-cpp-python (with Vulkan or CPU backend)
- Textual >= 1.0
- httpx for remote API
- huggingface-hub for model download
- Target: Debian 13 (Trixie)

### System dependencies

- `xclip` — clipboard copy support (falls back to OSC52 terminal escapes without it)

## Model

Default: Qwen3-4B Q4_K_M GGUF (~2.5GB). Supports Hermes-style function calling natively.
Auto-downloaded on first run to ~/.local/share/natshell/models/

## Key Files Reference

- `PROJECT_SCAFFOLD.md` — Full architectural specification with detailed pseudocode for every module
- `config.default.toml` — Default configuration with all safety patterns
- `pyproject.toml` — Dependencies and entry point

## Testing

Run tests with `pytest`. Mock the InferenceEngine for agent loop tests. Tools can be tested directly against the real system (be careful with write_file tests — use /tmp).

## Installing for development

```bash
cd natshell
sudo apt install xclip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# For CPU-only llama.cpp:
pip install llama-cpp-python
# For Vulkan (AMD GPUs):
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python
```
