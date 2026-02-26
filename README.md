# NatShell

Natural language shell interface for Linux — a local-first agentic TUI powered by a bundled LLM.

Type requests in plain English and NatShell plans and executes shell commands to fulfill them, using a ReAct-style agent loop with a small local model (Qwen3-4B via llama.cpp).

## Install

```bash
git clone https://github.com/Barent/natshell.git && cd natshell
bash install.sh
```

The installer handles everything — Python venv, GPU detection, llama.cpp build, and model download. No sudo required. Missing system dependencies (C++ compiler, etc.) are detected and offered for install automatically.

### Development setup

```bash
git clone https://github.com/Barent/natshell.git && cd natshell
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install llama-cpp-python                # CPU-only
# CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --no-cache-dir  # Vulkan GPU
natshell
```

## Usage

```bash
natshell                          # Launch with defaults
natshell --model ./my-model.gguf  # Use a specific model
natshell --remote http://host:11434/v1 --remote-model qwen3:4b  # Use Ollama
natshell --download               # Download the default model and exit
natshell --config ~/.config/natshell/config.toml  # Custom config
```

## Configuration

Copy `config.default.toml` to `~/.config/natshell/config.toml` to customize. Key settings:

- **Model**: path, context window, GPU layers, thread count
- **Safety**: confirm/warn/yolo modes, custom regex patterns for dangerous commands
- **Agent**: max steps, temperature, max tokens

## Safety

Commands are classified into three risk levels:

- **Safe** — auto-executed (ls, cat, df, grep, etc.)
- **Confirm** — requires user approval (rm, sudo, apt install, etc.)
- **Blocked** — never executed (fork bombs, rm -rf /, etc.)

Safety patterns are fully customizable in config.

## Development

```bash
source .venv/bin/activate
pytest                    # Run tests
ruff check src/ tests/    # Lint
```

## Architecture

- **Agent loop** (`src/natshell/agent/loop.py`): ReAct pattern — reason, act, observe, repeat
- **Tools** (`src/natshell/tools/`): execute_shell, read_file, write_file, list_directory, search_files
- **Inference** (`src/natshell/inference/`): Local llama.cpp or remote OpenAI-compatible API
- **Safety** (`src/natshell/safety/`): Regex-based command classifier
- **TUI** (`src/natshell/app.py`): Textual framework

## License

MIT
