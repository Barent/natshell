"""Pre-edit backup system — backs up files before edits and supports undo."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

BACKUP_DIR = Path.home() / ".local" / "share" / "natshell" / "backups"


class BackupManager:
    """Create timestamped backups of files before edits, with undo support."""

    def __init__(self, backup_dir: Path = BACKUP_DIR, max_per_file: int = 10) -> None:
        self._backup_dir = backup_dir
        self._max_per_file = max_per_file
        self._history: list[tuple[Path, Path]] = []  # (original, backup) pairs

    def backup(self, path: str | Path) -> Path | None:
        """Create a timestamped backup of a file before editing.

        Returns the backup path, or None if the file doesn't exist.
        """
        raw = Path(path).expanduser()
        if raw.is_symlink():
            logger.warning("Refusing to back up symlink: %s → %s", raw, raw.resolve())
            return None
        source = raw.resolve()
        if not source.exists() or not source.is_file():
            return None

        self._backup_dir.mkdir(parents=True, exist_ok=True)
        self._backup_dir.chmod(0o700)
        ts = time.time_ns()
        backup_name = f"{source.stem}.{ts}{source.suffix}.bak"
        backup_path = self._backup_dir / backup_name

        try:
            shutil.copy2(source, backup_path)
            self._history.append((source, backup_path))
            logger.debug("Backed up %s → %s", source, backup_path)
            self._prune(source)
            return backup_path
        except Exception:
            logger.exception("Failed to back up %s", source)
            return None

    def undo_last(self) -> tuple[bool, str]:
        """Restore the most recent backup.

        Returns (success, message).
        """
        if not self._history:
            return False, "No backups to undo."

        original, backup = self._history.pop()
        if not backup.exists():
            return False, f"Backup file missing: {backup}"

        try:
            shutil.copy2(backup, original)
            backup.unlink()
            return True, f"Restored {original} from backup."
        except Exception as e:
            return False, f"Failed to restore: {e}"

    def _prune(self, source: Path) -> None:
        """Keep only the most recent max_per_file backups for a given file."""
        stem = source.stem
        suffix = source.suffix
        pattern = f"{stem}.*{suffix}.bak"

        backups = sorted(self._backup_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        excess = len(backups) - self._max_per_file
        if excess <= 0:
            return
        for old in backups[:excess]:
            try:
                old.unlink()
            except OSError:
                pass

    def reset(self) -> None:
        """Clear history (for tests)."""
        self._history.clear()


# Module-level singleton
_instance = BackupManager()


def get_backup_manager() -> BackupManager:
    """Return the module-level BackupManager singleton."""
    return _instance


def reset_backup_manager() -> None:
    """Reset backup history (used by tests)."""
    _instance.reset()
