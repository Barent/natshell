"""Resolving a caller-supplied path for writing, without following a symlink.

``Path.resolve()`` follows symlinks, so a tool that resolves and then writes
edits whatever the link points at.  ``BackupManager.backup()`` already refuses
symlinks — it returns ``None`` rather than copying through one — but its
callers discarded that return value, so the write went ahead with no backup and,
where the link's own path was treated as safe, with no confirmation either.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_write_target(path: str) -> tuple[Path | None, str]:
    """Resolve *path* for writing.

    Returns ``(target, "")`` when the path is safe to write, or
    ``(None, message)`` explaining why it is not.  A symlink is refused rather
    than followed: the caller asked to write one file, and following the link
    writes a different one.
    """
    try:
        raw = Path(path).expanduser()
    except (OSError, ValueError, RuntimeError) as e:
        return None, f"Invalid path {path!r}: {e}"

    if raw.is_symlink():
        try:
            points_at = os.readlink(raw)
        except OSError:
            points_at = "?"
        return None, (
            f"Refusing to write through the symlink {raw} (points at {points_at}). "
            f"Write to the real path directly if that is what you meant."
        )

    try:
        target = raw.resolve()
    except (OSError, ValueError, RuntimeError) as e:
        return None, f"Could not resolve path {path!r}: {e}"

    if target.exists() and not target.is_file():
        # Wording matches the tools' existing vocabulary for this case.
        return None, f"Refusing to write {target}: not a file."

    return target, ""
