# Changelog

All notable changes to NatShell will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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
- Regex-based safety classifier with three modes (confirm, warn, yolo)
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
