"""Track file read state to prevent editing partially-read files."""

from __future__ import annotations

from pathlib import Path

# Read states
_PARTIAL = "partial"
_FULL = "full"

_tracker_state: dict[str, str] = {}


class FileReadTracker:
    """Tracks whether files have been fully read before allowing edits.

    Uses a module-level dict so all tools share the same state within a session.
    """

    def record_read(self, path: str, truncated: bool) -> None:
        """Record a first read (offset=1) of a file.

        Sets state to "partial" if truncated (and not already "full"),
        or "full" if the entire file was returned.
        """
        key = self._resolve(path)
        if truncated:
            # Don't downgrade a full read back to partial
            if _tracker_state.get(key) != _FULL:
                _tracker_state[key] = _PARTIAL
        else:
            _tracker_state[key] = _FULL

    def record_continuation(self, path: str, truncated: bool) -> None:
        """Record a continuation read (offset>1) of a file.

        Promotes "partial" to "full" when the continuation is not truncated.
        """
        key = self._resolve(path)
        if not truncated:
            _tracker_state[key] = _FULL
        # If still truncated, leave state as-is (partial stays partial)

    def can_edit(self, path: str) -> tuple[bool, str]:
        """Check if a file can be edited.

        Returns (True, "") if allowed, (False, reason) if blocked.
        Files never read are allowed (agent may know content from other means).
        """
        key = self._resolve(path)
        state = _tracker_state.get(key)
        if state == _PARTIAL:
            return (
                False,
                f"Cannot edit {path}: file was only partially read (truncated). "
                "Read the remaining lines with read_file(offset=...) before editing, "
                "or use write_file to rewrite the entire file.",
            )
        return (True, "")

    def invalidate(self, path: str) -> None:
        """Remove tracking for a file after it has been written/edited."""
        key = self._resolve(path)
        _tracker_state.pop(key, None)

    def clear(self) -> None:
        """Reset all tracking state."""
        _tracker_state.clear()

    @staticmethod
    def _resolve(path: str) -> str:
        """Resolve path to a canonical key."""
        return str(Path(path).expanduser().resolve())


# Module-level singleton
_instance = FileReadTracker()


def get_tracker() -> FileReadTracker:
    """Return the module-level FileReadTracker singleton."""
    return _instance


def reset_tracker() -> None:
    """Reset all tracking state (used by agent loop on /clear and by tests)."""
    _instance.clear()
