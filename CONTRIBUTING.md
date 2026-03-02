# Contributing to NatShell

## Development Setup

```bash
git clone https://github.com/Barent/natshell.git && cd natshell
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

For GPU-accelerated local inference:

```bash
# Vulkan (Linux — AMD/NVIDIA)
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --no-binary llama-cpp-python --no-cache-dir

# Metal (macOS)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-binary llama-cpp-python --no-cache-dir

# CPU only
pip install llama-cpp-python
```

Always use `--no-binary llama-cpp-python --no-cache-dir` when installing with GPU flags to prevent pip from reusing a cached CPU-only wheel.

## Running Tests

```bash
pytest                    # Full suite (669 tests)
pytest --tb=short -q      # Compact output
pytest tests/test_safety.py  # Single file
```

## Linting

```bash
ruff check src/ tests/
```

Line length limit is 100 characters. Ruff is configured in `pyproject.toml`.

## Code Style

- Python 3.11+ — use `from __future__ import annotations` for type hints
- Platform detection: use `is_macos()`, `is_wsl()`, `is_linux()` from `natshell.platform` — don't scatter `sys.platform` checks
- GPU detection: use functions from `natshell.gpu` — both are cached with `lru_cache`
- Lazy imports: `llama-cpp-python` is optional — all `from llama_cpp` imports must be inside functions wrapped in try/except
- Mock the `InferenceEngine` protocol for agent loop tests — don't require a real model
- Use `/tmp` for any write_file tests to avoid polluting the repo

## Pull Request Process

1. Create a feature branch from `main`
2. Write or update tests for your changes
3. Ensure `pytest` passes and `ruff check` is clean
4. Open a PR with a clear description of what changed and why
5. CI will run the test matrix (Python 3.11, 3.12, 3.13) automatically

## Architecture Guidelines

- The safety classifier is regex-based and deterministic — don't add LLM-based classification
- System context is gathered once at startup, not on every request
- Tool definitions follow the OpenAI function calling schema
- The agent loop is async — wrap blocking calls in `asyncio.to_thread()`
- Output from commands is truncated to ~4000–64000 chars depending on context window size
