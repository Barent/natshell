Session persistence lets you save and resume conversations.

Commands:
  /save [name]   — Save current conversation (optional name)
  /load [id]     — Load a saved session by ID or pick from list
  /sessions      — List all saved sessions

Details:
  Storage: ~/.local/share/natshell/sessions/
  Format: JSON with conversation history and metadata
  Size limit: 10 MB per session (configurable)
  Session IDs: 32-character hex (UUID format)
  Directory permissions: 0o700 (user-only access)