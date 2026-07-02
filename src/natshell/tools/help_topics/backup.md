NatShell automatically backs up files before edits.

How it works:
  - Before every edit_file or write_file, a timestamped copy is saved
  - Backups are stored in ~/.local/share/natshell/backups/
  - Directory permissions: 0o700 (user-only access)
  - Symlinks are refused (security measure)

Commands:
  /undo    — Restore the most recent backup

Configuration:
  Backup pruning keeps a limited number of backups per file.
  Old backups are automatically cleaned up.