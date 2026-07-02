Getting started with NatShell:

  1. First run — NatShell runs a setup wizard to pick a model tier:
     Light (4B), Standard (8B), or Enhanced (12B).
     You can also choose Remote only (Ollama/API) or Skip.

  2. Download models later — use /model download from within NatShell:
     /model download              — show tiers and download status
     /model download standard     — download the 8B model

  3. Connect to Ollama — set the URL in config.toml:
     [ollama]
     url = "http://localhost:11434"
     default_model = "qwen3:8b"

  4. Basic usage — type a request in plain English:
     "scan my local network for computers"
     "find all Python files larger than 1MB"
     "edit config.py and change the timeout to 30"

  5. Use /help to see all commands, or ask about any topic:
     getting_started, commands, tools, models, profiles, prompt_customization, memory, sessions, plans, plugins, headless, mcp, backup, keyboard_shortcuts, safety, config, troubleshooting