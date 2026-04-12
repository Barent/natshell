# NatShell — Claude Code Instructions

## What is this?

NatShell is an agentic TUI that provides a natural language interface to Linux, macOS, and WSL. Users type requests in plain English and a bundled local LLM plans and executes multi-step shell operations. Also serves as a coding assistant — reads, edits, writes files, and runs code in 10 languages.

## Architecture

- **Agent loop**: ReAct pattern — model reasons, calls tools, observes results, repeats (max 15 steps, auto-scales to 75 for large context windows)
- **Inference**: llama-cpp-python backend, with optional Ollama or remote API fallback. Runtime engine swapping supported.
- **TUI**: Textual framework with custom widgets, command palette, clipboard integration
- **Safety**: Regex-based command classifier (safe/confirm/blocked) with command chaining detection, sensitive path gating, and env var filtering
- **Tools**: execute_shell, read_file, write_file, edit_file, run_code, list_directory, search_files, git_tool, natshell_help, fetch_url

## Key Design Decisions

1. `execute_shell` runs via `bash -c` with the user's real environment — intentional, not a sandbox
2. Safety classifier is pattern-based (fast, deterministic), NOT LLM-based
3. System context is gathered once at startup and injected into the system prompt
4. Local inference uses `llama-cpp-python` with plain-text tool definitions. Qwen3 outputs `<tool_call>` XML; Mistral outputs `[TOOL_CALLS]` JSON — both parsed in `_parse_response()` with model family auto-detection
5. Agent loop is async — local inference wrapped in `asyncio.to_thread()` to avoid blocking TUI
6. Output truncated to ~4000 chars; auto-scales to 64K for 256K context windows
7. Platform detection centralized in `src/natshell/platform.py` (`lru_cache`). Use `is_macos()`, `is_wsl()`, `is_linux()`.
8. GPU detection in `src/natshell/gpu.py` (`lru_cache`). Tries vulkaninfo → nvidia-smi → lspci. Prefers discrete GPUs.
9. Engine preference persisted via `[engine]` in config.toml (`preferred = "auto" | "local" | "remote"`)
10. Context window adaptive scaling — max_tokens, max_steps, output truncation, read_file limits all auto-scale with n_ctx (4K→256K tiers). See `_effective_*` methods in loop.py.
11. Auto-timeout for long-running commands — `_LONG_RUNNING_PATTERNS` in execute_shell.py ensures nmap, apt, make, etc. get adequate time
12. Plan generation/execution — `/plan` generates structured markdown plans; `/exeplan run` executes step-by-step with per-step agent budgets
13. `read_file` truncation emits `⚠ FILE TRUNCATED` with offset hint. `FileReadTracker` blocks `edit_file` on partially-read files.
14. Edit failure tracking — escalating warnings after 2+ failures; completion guard prevents declaring success when all edits failed
15. Headless mode — `--headless "prompt"` for single-shot invocations. `--danger-fast` auto-approves. Exit 1 on error.
16. Session persistence — JSON in `~/.local/share/natshell/sessions/`. IDs validated as 32-char hex. 10 MB size limit. 0o700 perms.
17. Plugin system — `~/.config/natshell/plugins/*.py`, each with `register(registry)` entry point
18. MCP server mode — `--mcp` exposes all tools via JSON-RPC over stdin/stdout
19. Backup & undo — `BackupManager` snapshots files before edits; `/undo` restores. Symlinks refused. 0o700 perms.
20. Small context tool filtering — n_ctx ≤ 8192 auto-filters to 5 core tools (`SMALL_CONTEXT_TOOLS`: execute_shell, read_file, write_file, edit_file, list_directory)
21. Sudo passthrough — `_inject_sudo_dash_s()` replaces `sudo` with `sudo -S` only at command positions (not inside strings). Password piped once. `y\n` auto-confirm only for package managers. On sudo failure, agent auto-prepends `sudo` and re-classifies before retry.

## Security Features (summary)

Rich markup escaping on all LLM output; command chaining splits on `&&`/`||`/`;`/`&`/`|` before classification; sudo password cached 5 min; sensitive env vars filtered from subprocess; HTTPS warning for plaintext API keys; SSH key/shadow/`.env` path gating; session/backup dir 0o700; session path traversal validation; git commit flag blocklist (`--amend`, `--author=`, `--date=`); SSRF blocking in fetch_url.

## Modules

### Core
- `__main__.py` — CLI entry, argparse, model download, engine wiring
- `app.py` — Textual TUI, confirmation/sudo dialogs, model switching
- `commands.py` — Slash command dispatch: `/plan`, `/exeplan`, `/undo`, `/save`, `/load`, `/sessions`, `/compact`, `/keys`, etc.
- `config.py` — TOML config, `NATSHELL_API_KEY` env var, permission warnings, engine preference persistence
- `backup.py` — Pre-edit snapshots, undo, symlink rejection, 0o700
- `headless.py` — `--headless` single-shot mode
- `session.py` — Session save/load/delete/list with security hardening
- `mcp_server.py` — MCP JSON-RPC server
- `plugins.py` — Plugin loader
- `model_manager.py` — Model discovery, download, switching

### Agent
- `agent/loop.py` — ReAct loop with safety, sudo retry, engine fallback, small-context filtering, edit failure tracking
- `agent/system_prompt.py` — Platform-aware system prompt with behavior rules and `/no_think` directive
- `agent/context.py` — System context gathering (CPU, RAM, disk, network, services, containers)
- `agent/context_manager.py` — Token budget management, message trimming, `/compact` summarization, `build_summary_with_refs()` for retrieval-augmented compaction
- `agent/memory_files.py` — Plain-file chunk store backing retrieval-augmented compaction. One subdirectory per session at `~/.local/share/natshell/memory/<session_id>/`, one `.txt` file per chunk named by SHA-12 prefix of its content (dir `0o700`, files `0o600`). Stdlib-only: `write_chunk`, `cleanup_session`, `prune_old`, `healthy`. Per-chunk cap 64 KB (oversize content stored head+tail with elision marker). `prune_old(max_age_days)` is called lazily from `compact_history` at most once per process. The agent retrieves chunks via the existing `read_file` / `search_files` tools — no dedicated recall tool. Self-disables after 3 consecutive OSError write failures; `compact_history()` then falls back to the legacy extractive summary.
- `agent/plan.py` — Markdown plan parser, extracts `PlanStep` from H2 headings
- `agent/plan_executor.py` — Step-by-step plan execution with per-step agent budgets

### Inference
- `inference/engine.py` — Protocol types: `CompletionResult`, `ToolCall`, `EngineInfo`
- `inference/local.py` — llama-cpp-python backend, auto context sizing, GPU selection, XML tool parsing, `<think>` stripping
- `inference/remote.py` — OpenAI-compatible API backend (httpx)
- `inference/ollama.py` — Ollama ping, model listing, URL normalization

### Tools
- `tools/registry.py` — Tool registration/dispatch, `PLAN_SAFE_TOOLS`, `SMALL_CONTEXT_TOOLS`
- `tools/execute_shell.py` — Shell exec with sudo caching, position-aware injection, env filtering, output truncation, auto-timeouts
- `tools/read_file.py` — File reading with offset/limit, truncation warning
- `tools/write_file.py` — File writing (always confirms)
- `tools/file_tracker.py` — Tracks partial/full read state; blocks edit on partially-read files
- `tools/edit_file.py` — Search-and-replace edits; unique match required; fuzzy suggestions on miss; `start_line`/`end_line` for duplicates
- `tools/run_code.py` — Run code in 10 languages (always confirms)
- `tools/list_directory.py` — Directory listing with sizes, hidden file toggle
- `tools/search_files.py` — grep + find wrapper
- `tools/git_tool.py` — Git operations; read-only safe, mutating confirms; commit flag blocklist
- `tools/limits.py` — Centralized context-aware truncation limits
- `tools/natshell_help.py` — Self-documentation with static/dynamic topics, injected `SafetyConfig`
- `tools/fetch_url.py` — URL fetch with SSRF blocking, 1 MB cap, GET-only

### Safety & UI
- `safety/classifier.py` — Command risk classifier, chained command splitting, subshell detection, sensitive path gating
- `ui/widgets.py` — All custom Textual widgets (messages, inputs, indicators, dialogs)
- `ui/commands.py` — Command palette for model switching
- `ui/clipboard.py` — Cross-platform clipboard (pbcopy/clip.exe/wl-copy/xclip/OSC52)
- `ui/escape.py` — Rich markup escaping
- `ui/styles.tcss` — Textual CSS

### Utilities
- `platform.py` — `is_macos()`, `is_wsl()`, `is_linux()` (cached)
- `gpu.py` — GPU detection, vendor classification, best device selection

## Tech Stack

- Python 3.11+, llama-cpp-python (Vulkan/Metal/CPU), Textual ≥ 1.0, httpx, huggingface-hub
- Platforms: Linux, macOS, WSL

## Models

| Tier | Model | Size | Context | Tool format |
|------|-------|------|---------|-------------|
| Light | Qwen3-4B Q4_K_M | ~2.5 GB | 4096 | `<tool_call>` XML |
| Standard | Qwen3-8B Q4_K_M | ~5 GB | 8192 | `<tool_call>` XML |
| Enhanced | Mistral Nemo 12B Q4_K_M | ~7.5 GB | 32768 | `[TOOL_CALLS]` JSON |

Default: Qwen3-4B, auto-downloaded to `~/.local/share/natshell/models/`. Context size auto-detected from filename when `n_ctx = 0`.

## Key Files

- `src/natshell/config.default.toml` — Default config with all safety patterns (23 confirm + 8 blocked, incl. macOS-specific)
- `pyproject.toml` — Dependencies and entry point
- `install.sh` — Cross-platform installer with GPU detection

## Testing

Run with `pytest` (1,175 tests, 39 files). Mock `InferenceEngine` for agent loop tests. Use `/tmp` for write_file tests.

Key test files: `test_agent.py`, `test_safety.py`, `test_tools.py`, `test_coding_tools.py`, `test_file_tracker.py`, `test_sessions.py`, `test_backup.py`, `test_headless.py`, `test_git_tool.py`, `test_mcp_server.py`, `test_plugins.py`, `test_plan_*.py`, `test_engine_*.py`, `test_ollama*.py`, `test_slash_commands.py`, `test_context_manager.py`, `test_widgets.py`, `test_commands.py`, `test_gpu.py`, `test_platform.py`, `test_clipboard.py`, `test_fetch_url.py`, `test_natshell_help.py`, `test_history_input.py`, `test_prompt_cache.py`, `test_context.py`

## Cross-Platform Notes

- **Clipboard**: macOS `pbcopy`, WSL `clip.exe`, Linux `wl-copy`/`xclip`/OSC52
- **System context**: macOS uses `sw_vers`/`sysctl`/`vm_stat`; Linux/WSL uses `lscpu`/`free`/`ip`/`systemctl`
- **Installer**: macOS uses `brew`, `xcode-select`, Metal; Linux uses system package managers
- **Package managers**: Auto-detected — brew, apt, dnf, yum, pacman, zypper, apk, emerge, rpm-ostree

## Dev Install

```bash
cd natshell
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# CPU-only:
pip install llama-cpp-python
# Vulkan (AMD/Linux):
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --no-binary llama-cpp-python --no-cache-dir
# Metal (macOS):
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-binary llama-cpp-python --no-cache-dir
```

Always use `--no-binary llama-cpp-python --no-cache-dir` with GPU flags to avoid cached CPU-only wheels.
