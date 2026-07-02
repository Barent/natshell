MCP (Model Context Protocol) server mode exposes NatShell's tools via JSON-RPC.

Usage:
  natshell --mcp

Protocol: JSON-RPC over stdin/stdout
All NatShell tools are available as MCP methods.

Safety configuration:
  [mcp] section in config.toml controls safety mode.
  The same safety classifier applies as in the TUI.

This allows external editors or AI agents to use NatShell's tools (execute_shell, read_file, edit_file, etc.) as an MCP tool provider.