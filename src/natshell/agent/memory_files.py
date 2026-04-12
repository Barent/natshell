"""Plain-file compaction memory.

When ``AgentLoop.compact_history`` drops old messages, their content is
written verbatim to plain ``.txt`` files under
``~/.local/share/natshell/memory/<session_id>/``.  The compaction summary
replaces each dropped chunk with the resolved absolute path of its file,
so the agent can use the existing ``read_file`` and ``search_files`` tools
to retrieve the content on demand — no new tool to learn, no new storage
abstraction.

This module is the lightweight replacement for the earlier SQLite-backed
``MemoryStore``.  Design constraints:

* zero dependencies beyond the stdlib
* zero new tool registrations
* every public function returns cleanly on OSError so the agent loop is
  never broken by a memory write failure
* after ``_MAX_FAILURES`` consecutive write failures the feature
  self-disables for the rest of the process lifetime
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
from pathlib import Path

from natshell.platform import data_dir as _data_dir

logger = logging.getLogger(__name__)

# Root directory for all sessions' chunk files.
MEMORY_ROOT = _data_dir() / "memory"

# Prefix length used for chunk filenames.  12 hex chars = 48 bits of
# collision resistance; collisions are resolved by overwrite.
SHORT_HASH_LEN = 12

# Per-chunk byte cap.  Oversize content is stored head+tail with an
# elision marker so very large tool outputs don't blow out disk usage.
CHUNK_MAX_BYTES = 64 * 1024

# After this many consecutive write failures the feature self-disables.
_MAX_FAILURES = 3

# Process-wide failure counter.  Flipped off explicitly by
# ``_record_success`` when any write succeeds.
_consecutive_failures = 0
_disabled = False


def healthy() -> bool:
    """Return True when chunk writes should still be attempted."""
    return not _disabled


def reset_health() -> None:
    """Test helper — clear the process-wide disable latch."""
    global _consecutive_failures, _disabled
    _consecutive_failures = 0
    _disabled = False


def session_memory_dir(session_id: str) -> Path:
    """Return the per-session chunk directory.  Does not create it."""
    return MEMORY_ROOT / session_id


def _ensure_session_dir(session_id: str) -> Path:
    """Create (if needed) and return the session's chunk directory."""
    path = session_memory_dir(session_id)
    path.mkdir(parents=True, exist_ok=True)
    # Best-effort — Windows and read-only mounts are non-fatal.
    try:
        path.chmod(0o700)
        if path.parent != path:
            path.parent.chmod(0o700)
    except OSError:
        pass
    return path


def _sha_prefix(content: str) -> str:
    """Return the SHA-12 filename prefix for a chunk's content."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return digest[:SHORT_HASH_LEN]


def _truncate_oversize(content: str) -> str:
    """Cap a chunk at ``CHUNK_MAX_BYTES`` with a head+tail elision marker."""
    raw = content.encode("utf-8")
    if len(raw) <= CHUNK_MAX_BYTES:
        return content
    half = CHUNK_MAX_BYTES // 2 - 64
    head = content[:half]
    tail = content[-half:]
    elided = len(raw) - CHUNK_MAX_BYTES
    return (
        f"{head}\n\n[... {elided} bytes elided — original was "
        f"{len(raw)} bytes; re-read the source file if still present "
        f"...]\n\n{tail}"
    )


def _record_failure(exc: Exception) -> None:
    global _consecutive_failures, _disabled
    _consecutive_failures += 1
    logger.warning(
        "memory_files write failed (%d/%d): %s",
        _consecutive_failures,
        _MAX_FAILURES,
        exc,
    )
    if _consecutive_failures >= _MAX_FAILURES:
        _disabled = True
        logger.error(
            "memory_files disabled after %d consecutive failures",
            _consecutive_failures,
        )


def _record_success() -> None:
    global _consecutive_failures
    if _consecutive_failures:
        _consecutive_failures = 0


def write_chunk(session_id: str, content: str) -> Path | None:
    """Write a chunk to the session's memory directory.

    Returns the resolved absolute path on success, or ``None`` when the
    write fails or the feature is disabled.  Oversize content is
    truncated with an elision marker before writing.

    Filenames are SHA-12 of the (possibly truncated) content — idempotent
    for identical content within a session.  Collision at the 48-bit
    prefix is resolved by overwrite.
    """
    if _disabled or not session_id or not content:
        return None
    try:
        body = _truncate_oversize(content)
        name = _sha_prefix(body) + ".txt"
        directory = _ensure_session_dir(session_id)
        path = directory / name
        # Idempotent: skip the write if the file already exists and
        # matches (common within a single session when the same command
        # produces identical output).
        if not path.exists():
            path.write_text(body, encoding="utf-8")
            try:
                path.chmod(0o600)
            except OSError:
                pass
        _record_success()
        return path.resolve()
    except OSError as exc:
        _record_failure(exc)
        return None


def cleanup_session(session_id: str) -> None:
    """Remove all chunk files for a session.  Best-effort; errors swallowed."""
    if not session_id:
        return
    path = session_memory_dir(session_id)
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except OSError as exc:
        logger.warning("Failed to clean session memory dir %s: %s", path, exc)


def prune_old(max_age_days: int = 30) -> int:
    """Remove chunk files older than ``max_age_days``.

    Iterates every session directory under ``MEMORY_ROOT`` and unlinks
    files whose mtime is past the cutoff.  Returns the number of files
    removed.  Safe to call repeatedly; intended to be invoked lazily
    from ``compact_history`` at most once per process.
    """
    if max_age_days <= 0 or not MEMORY_ROOT.exists():
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    try:
        for session_dir in MEMORY_ROOT.iterdir():
            if not session_dir.is_dir():
                continue
            for chunk in session_dir.glob("*.txt"):
                try:
                    if chunk.stat().st_mtime < cutoff:
                        chunk.unlink()
                        removed += 1
                except OSError:
                    continue
            # Drop empty session dirs.
            try:
                if not any(session_dir.iterdir()):
                    session_dir.rmdir()
            except OSError:
                pass
    except OSError as exc:
        logger.warning("prune_old failed to scan %s: %s", MEMORY_ROOT, exc)
    return removed


def total_size_bytes() -> int:
    """Estimate total disk usage across all session chunk directories."""
    if not MEMORY_ROOT.exists():
        return 0
    total = 0
    try:
        for session_dir in MEMORY_ROOT.iterdir():
            if not session_dir.is_dir():
                continue
            for chunk in session_dir.glob("*.txt"):
                try:
                    total += chunk.stat().st_size
                except OSError:
                    continue
    except OSError:
        pass
    return total
