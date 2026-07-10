Headless mode runs NatShell non-interactively for scripting.

Usage:
  natshell --headless "your prompt here"

Options:
  --danger-fast     Auto-approve all confirmations (use with caution)
  --no-danger-fast  Force confirmations back on, overriding a persisted
                     [safety] danger_fast = true in config.toml

Persisting danger-fast mode:
  Set [safety] danger_fast = true in config.toml to always run as if
  --danger-fast were passed (e.g. on a disposable VM or CI runner).
  BLOCKED commands are still blocked either way.

Output:
  Response text → stdout
  Diagnostics   → stderr

Exit codes:
  0 — Success
  1 — Error (agent failure, tool error, etc.)

Examples:
  natshell --headless "list files in /tmp" > output.txt
  natshell --headless --danger-fast "update packages"
  echo $(natshell --headless "what is my IP address")