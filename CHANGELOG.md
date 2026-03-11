# Changelog

All notable changes to NatShell will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] - 2026-03-11

### Security

- Block destructive `git branch` flags (`-D`, `-M`, `--force`, `--delete`) in git_tool — use execute_shell for these operations
- Block destructive `git stash` subcommands (`drop`, `clear`) in git_tool
- Remove silent `git reset --hard` from `--update` — now prints manual instructions and exits
- Add redirect-based SSRF protection in fetch_url — checks final URL hostname after redirects

### Fixed

- edit_file error preview now scales with context window instead of hardcoded 200 lines
- RemoteEngine HTTP client properly closed on shutdown (try/finally around main run section)

### Added

- System prompt customization via `[prompt]` config section — custom `persona` and `extra_instructions` (core safety rules always included)
- `--profile NAME` CLI flag to apply a named configuration profile at startup
- Warning logged when loading a model with unrecognized family (neither qwen nor mistral)

### Improved

- Deduplicated tool prompt formatters in local.py — shared `_format_tool_entries()` helper
- Deduplicated TOML save functions — `save_ollama_default`, `save_model_config`, and `save_engine_preference` now use `save_config_value`/`save_config_values`
- Deduplicated `_merge_toml` — loop over `_SECTIONS` tuple instead of 10 copy-pasted blocks
- Updated classifier from "Alpha" to "Beta" in pyproject.toml
- Added `package-lock.json` to `.gitignore`

## [0.1.28] - 2026-03-02

### Added

- Mistral Nemo 12B (Q4_K_M) as third local model tier ("Enhanced") with 128K context window
- Dual-parser tool calling: Qwen3 (`<tool_call>` XML) and Mistral (`[TOOL_CALLS]` JSON array) with auto-detection from model filename
- Installer now offers Light (Qwen3-4B), Standard (Qwen3-8B), and Enhanced (Mistral Nemo 12B) presets

## [0.1.27] - 2026-03-02

### Fixed

- Fix lint failures — import sorting, unused variables, line length in app.py, test_agent.py, test_fetch_url.py

## [0.1.26] - 2026-03-02

### Added

- `fetch_url` tool: fetch web content with SSRF protection (private IP blocking), 1 MB response cap, 60s timeout, GET-only
- Plan-aware tool filtering: agent loop detects plan requests and restricts tools during plan generation
- Pre-flight compaction: force context compaction when estimated tokens exceed context window
- Tool schema filtering: `get_tool_schemas(allowed=...)` for selective tool exposure

## [0.1.25] - 2026-03-02

### Improved

- Repetitive read detection: warn after reading the same file 3+ times without modifying it
- Step budget awareness: inject info at 50%, warning at 75%, urgent alert at 90% of max_steps
- Updated edit failure warnings: guide toward write_file on repeated failures, STOP directive at 3+ failures
- System prompt: add compile-check guidance, directory awareness, stronger edit recovery rules

## [0.1.24] - 2026-03-01

### Added

- Configuration profiles (`/profile` command): named presets that bundle ollama/remote/engine settings, switchable at runtime

## [0.1.23] - 2026-03-01

### Added

- `--log-file` flag for real-time error capture to a specified log file

## [0.1.22] - 2026-03-01

### Documentation

- Add project landing page (`docs/index.html`) with GitHub Pages support
- Update documentation for 8B model visibility in installer

## [0.1.21] - 2026-03-01

### Added

- Add safe `n_ctx = 32768` default to `[ollama]` config section
- Update documentation: model preset table in README, mention 8B option in installer

### Fixed

- Fix stale auto-scaling tier comments in config.default.toml (missing 256K tier)

## [0.1.20] - 2026-03-01

### Fixed

- Fix stale auto-scaling comments in default config to include 256K context tier values

## [0.1.19] - 2026-03-01

### Fixed

- Fix plan executor not stopping after step completion — executor now correctly exits after the last step

## [0.1.18] - 2026-03-01

### Fixed

- Fix `[ollama] n_ctx` config being silently ignored — context size from Ollama config section is now properly applied

## [0.1.17] - 2026-03-01

### Fixed

- Fix remote model disconnecting on context overflow — `ContextOverflowError` is now caught and reported as a user-facing message instead of triggering engine fallback

## [0.1.16] - 2026-03-01

### Fixed

- Fix flaky CI test: env var filtering tests used `env` which truncated on GitHub Actions; switched to targeted `echo` commands

## [0.1.15] - 2026-03-01

### Fixed

- Fix remote-to-local fallback: pass `main_gpu` config to LocalEngine (was missing, broke multi-GPU auto-selection)
- Fix pool timeout firing prematurely on large models — pool timeout now scales with read timeout instead of fixed 30s
- Expand fallback exception list: `ReadTimeout`, `PoolTimeout`, `RemoteProtocolError`, `OSError` now trigger fallback (previously only `ConnectError`, `ConnectTimeout`, `ConnectionError`)
- Add retry with exponential backoff (2 retries, 1s/2s) for transient connection failures and 502/503/504 server errors
- Surface GPU warning to user when fallback model runs on CPU-only llama-cpp-python build

## [0.1.14] - 2026-03-01

### Security

- **CRITICAL**: Fix path traversal in session persistence — session IDs are now validated as 32-char hex (UUID format)
- Add session size limit (10 MB default) to prevent disk exhaustion
- Set 0o700 permissions on session and backup directories (owner-only access)
- Reject symlinks in backup system to prevent silent exfiltration
- Block dangerous git commit flags (--amend, --author=, --date=, --reset-author, --allow-empty-message) in git_tool
- Fix headless exit code — any error now returns exit code 1, even if a response was also produced

### Added

- Security tests for session path traversal, size limits, backup symlinks, git commit flags, headless exit codes

### Documentation

- Update README.md with all new features (backup/undo, sessions, headless, MCP, plugins, prompt caching, diff preview)
- Update CLAUDE.md with new modules, security hardening items 10-17, 28 test files
- Add CHANGELOG entries for v0.1.8 through v0.1.14
- Update test counts across all documentation

## [0.1.13] - 2026-02-28

### Fixed

- Fix lint errors from v0.1.12

## [0.1.12] - 2026-02-28

### Improved

- Improve code editing reliability with file read tracker, fuzzy match suggestions, and completion guard

## [0.1.11] - 2026-02-28

### Improved

- Return context around edit point on successful edit_file operations

## [0.1.10] - 2026-02-28

### Fixed

- Prevent editing partially-read files (FileReadTracker enforcement)

## [0.1.9] - 2026-02-28

### Added

- Backup & undo system — pre-edit file backups with `/undo` support
- Headless mode — `--headless` for non-interactive single-shot execution
- Session persistence — `/save`, `/load`, `/sessions` for conversation management
- Git tool — structured access to common git operations (status, diff, log, branch, commit, stash)
- Plugin system — custom tools from `~/.config/natshell/plugins/`
- MCP server mode — `--mcp` for JSON-RPC integration
- Prompt caching for reduced inference latency
- Diff preview in edit confirmations
- Slash command refactoring (commands.py extracted from app.py)
- Plan executor module for step-by-step plan execution
- Output limits module (tools/limits.py) for context-aware truncation
- Rich markup escape module (ui/escape.py)
- Model manager module for download and switching logic

## [0.1.8] - 2026-02-28

### Improved

- Test coverage sweep: 393 → 641+ tests across 28 test files
- Added tests for GPU detection, system context, widgets, slash commands, plugins, MCP, prompt cache, headless, sessions, git tool, backup

## [0.1.7] - 2026-02-28

### Improved

- Extend scaling tiers for 256K context windows (max_tokens→65536, max_steps→75, output→64K chars, read_file→4000 lines)

## [0.1.6] - 2026-02-28

### Improved

- Improve command timeout handling: default raised from 30s to 60s, auto-detection patterns for long-running commands (nmap, apt install, make, etc.), system prompt guidance for timeout usage

## [0.1.5] - 2026-02-28

### Improved

- Scale tool limits with context window size: shell output truncation and read_file line limits now auto-scale with n_ctx

## [0.1.4] - 2026-02-28

### Fixed

- Fix ruff lint: line length violation in plan execution prompt builder

## [0.1.3] - 2026-02-28

### Improved

- Agent step limit now auto-scales with model context window size: 15/25/35/50 steps for 4k/8k/16k/32k+ contexts
- Large and remote models get up to 50 steps per request, allowing complex multi-file tasks without hitting the ceiling
- Plan execution step limit increased from 25 to 35
- Explicit `max_steps` config overrides disable auto-scaling (user control preserved)

## [0.1.2] - 2026-02-28

### Fixed

- Publish workflow now attaches .whl and .tar.gz as GitHub release assets
- Publish workflow is idempotent on re-runs (skip-existing for PyPI, --clobber for assets)

## [0.1.1] - 2026-02-28

### Improved

- Plan execution now injects preamble context (tech stack, conventions, key interfaces) into each step prompt, preventing cross-step mistakes like ESM/CJS confusion
- Higher step budget for plan execution (25 tool calls per step, up from 15) to allow write-test-debug cycles
- Step prompts include budget awareness and cleanup guidance to reduce wasted tool calls and prevent cascading port conflicts
- Plan generation prompt includes execution-quality rules: explicit cross-file dependencies, test framework setup, and no "verify in browser" validation

## [0.1.0] - 2026-02-28

Initial public release.

### Added

- ReAct-style agent loop with configurable max steps, temperature, and token limits
- Local inference via bundled llama.cpp (llama-cpp-python) with auto model download (Qwen3-4B Q4_K_M)
- Remote inference via any OpenAI-compatible API (Ollama, vLLM, LM Studio, etc.)
- Automatic engine fallback — remote to local when server is unreachable
- Runtime model switching with `/model` commands (no restart required)
- 8 agent tools: execute_shell, read_file, write_file, edit_file, run_code, list_directory, search_files, natshell_help
- Code execution in 10 languages (Python, JavaScript, Bash, Ruby, Perl, PHP, C, C++, Rust, Go)
- Plan generation (`/plan`) and execution (`/exeplan`) for multi-step workflows
- Textual-based TUI with command palette, clipboard integration, input history
- GPU acceleration — auto-detection via vulkaninfo, nvidia-smi, lspci; Vulkan, Metal, and CPU backends
- Regex-based safety classifier with three modes (confirm, warn, danger)
- Command chaining detection — splits `&&`, `||`, `;`, `&`, `|` and classifies each sub-command
- Sensitive file path gating for read_file (SSH keys, /etc/shadow, .env, etc.)
- Environment variable filtering — strips API keys and tokens from subprocesses
- Sudo password caching with 5-minute expiry
- Rich markup escaping to prevent TUI injection
- Cross-platform support: Linux, macOS, WSL
- Cross-platform clipboard: wl-copy, xclip, xsel, pbcopy, clip.exe, OSC52 fallback
- Platform-aware system context gathering (CPU, RAM, disk, network, services, containers)
- Interactive installer with GPU detection, Ollama setup, and model download
- TOML configuration with sensible defaults and env var support (`NATSHELL_API_KEY`)
- 353 tests across 13 test files
