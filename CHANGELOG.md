# Changelog

All notable changes to NatShell will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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
