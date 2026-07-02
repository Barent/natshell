Working memory (agents.md) — persistent scratchpad across sessions.

The agent can read and write a persistent markdown file to remember facts, decisions, and context across sessions and plan steps.

File location (searched in order):
  1. {project_root}/.natshell/agents.md   (project-local)
  2. ~/.config/natshell/agents.md          (global fallback)

The project root is detected by looking for .git, pyproject.toml, package.json, Cargo.toml, go.mod, Makefile, or CMakeLists.txt.

Commands:
  /memory          — Show current memory content and source path
  /memory reload   — Re-read from disk and update the system prompt
  /memory clear    — Truncate the memory file
  /memory path     — Show where the file is (or would be created)

How the agent uses it:
  - Memory content is injected into the system prompt at startup
  - The agent can update it via write_file or edit_file (auto-approved)
  - During plan execution, memory is re-read before each step
  - Memory is skipped when context window < 16384 tokens

Configuration ([memory] section in config.toml):
  enabled    — Enable/disable memory (default: true)
  max_chars  — Base character budget (default: 4000, ~1000 tokens).
               Auto-scales with context window: 4K/8K/12K/16K/24K/32K
               for <32K/32K/64K/128K/256K/512K+ contexts.
  min_ctx    — Minimum context window for injection (default: 16384)

Tips:
  - Ask the agent to 'remember' something and it will write to agents.md
  - Keep entries concise — the file is injected into every prompt
  - Do not store secrets or credentials in agents.md