Headless mode runs NatShell non-interactively for scripting.

Usage:
  natshell --headless "your prompt here"

Options:
  --danger-fast    Auto-approve all confirmations (use with caution)

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