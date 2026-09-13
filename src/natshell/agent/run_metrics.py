"""Run-metrics persistence (R2-6 — recording half).

Every completed ``AgentLoop`` run (success, error, max-steps, …) appends
one JSON line to an append-only JSONL file under
``data_dir()/metrics`` — the same ``0o700`` / ``os.replace`` contract as
:mod:`natshell.backup` and :mod:`natshell.session`.

This half is **recording only**; it does *not* adapt ``n_ctx``,
``max_tokens``, truncation or step budget.  The **feedback** half (R2-6
follow-up) will consume :func:`RunMetricsStore.load_recent` to
feed history into :mod:`natshell.scaling` decisions, but that is
deliberately out of scope for this unit — the recording layer must be
green, tested, and in production before any policy can safely build on
it.

Default location:  ``~/.local/share/natshell/metrics/natshell.jsonl``
(or ``%LOCALAPPDATA%\\natshell\\metrics`` on Windows).

The module mirrors the :mod:`natshell.backup` shape: singleton accessor
:func:`get_run_metrics_store`, a reset hook for tests
(:func:`reset_run_metrics_store`), and a per-test ``RunMetricsStore(
dir_path, enabled=...)`` constructor that overrides just the directory.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from natshell.platform import data_dir as _data_dir

logger = logging.getLogger(__name__)

#: Process-wide lock serializing appends + the truncation rewrite.  A module
#: global (rather than an instance attribute) so the default singleton and
#: any per-test stores sharing a directory serialize against each other.
_WRITE_LOCK = threading.Lock()

#: Cap on retained JSONL lines (per-file).  1000 lines × ~300 bytes ≈ 300 KB —
#: trivial to enumerate, and long enough for meaningful history analysis.
MAX_LINES_DEFAULT = 1000

#: The single file within the metrics directory; a JSONL file is one line
#: per run.  (A single file is easier to enumerate than a fan-out, mirrors
#: ``session.py`` / ``backup.py`` conventions, and keeps the "load last N"
#: primitive O(1) instead of a directory walk.)
FILENAME = "natshell.jsonl"


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class RunMetricsStore:
    """Append-only JSONL run-stats store.

    Thread-safe: appends are serialized by a module-level lock that also
    covers the truncation pass (which rewrites the whole file via
    tempfile + ``os.replace``, mirroring ``config.py::_write_config_atomically``).
    """

    def __init__(
        self,
        dir_path: Path | str | None = None,
        *,
        enabled: bool = True,
        max_lines: int = MAX_LINES_DEFAULT,
        filename: str = FILENAME,
    ) -> None:
        # Global kill switch: NATSHELL_DISABLE_METRICS turns recording off
        # for the *default* (shared data-dir) location.  The test suite sets
        # it in conftest.py so the default singleton never writes into the
        # real data dir, while explicit stores that pass a ``dir_path``
        # (e.g. a temp dir) are unaffected.
        if dir_path is None and enabled and os.environ.get("NATSHELL_DISABLE_METRICS"):
            enabled = False
        self._dir = Path(dir_path) if dir_path is not None else (_data_dir() / "metrics")
        self._enabled = bool(enabled)
        self._max_lines = max(1, max_lines)
        self._filename = filename
        if self._enabled:
            self._prepare_dir()

    def disable(self) -> None:
        """Turn recording off at runtime (idempotent)."""
        self._enabled = False

    # ── directory bootstrap ──────────────────────────────────────────

    def _prepare_dir(self) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._dir.chmod(0o700)
        except OSError:
            logger.warning(
                "Run metrics dir %s unavailable — metrics will be in-memory only",
                self._dir,
                exc_info=True,
            )
            self._enabled = False

    @property
    def path(self) -> Path:
        return self._dir / self._filename

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def max_lines(self) -> int:
        return self._max_lines

    # ── record() ─────────────────────────────────────────────────────

    def record(self, stats: dict[str, Any] | None, **context: Any) -> Path | None:
        """Append one run's stats + context to the JSONL file.

        ``stats`` is the dict produced by
        :meth:`natshell.agent.step_metrics.RunStats.run_stats`.  ``context``
        carries anything about the run we might want for the future
        feedback half (model name, engine type, n_ctx, config snapshot,
        …) — anything JSON-serializable.

        Returns the file path on success, ``None`` when disabled or on
        failure.  Never raises: a metrics write must not break a run.
        """
        if not self._enabled or stats is None:
            return None
        line: dict[str, Any] = {"ts": time.time(), "ts_iso": _utc_iso(time.time())}
        line.update(stats)
        line.update(context)
        line["schema"] = 1
        try:
            text = json.dumps(line, separators=(",", ":"), default=str)
        except TypeError:
            logger.warning("Run metrics not JSON-serializable — skipped", exc_info=True)
            return None

        # Serialize with the module-wide lock (also used by the truncation pass)
        with _WRITE_LOCK:
            try:
                # Ensure the file exists (cheap, idempotent)
                if not self.path.exists():
                    self.path.touch()
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(text + "\n")
                self._maybe_truncate_locked()
                return self.path
            except OSError:
                logger.warning("Failed to append run metrics", exc_info=True)
                return None

    # ── load_recent() — the future feedback half consumes this ──────

    def load_recent(self, n: int = 0) -> list[dict[str, Any]]:
        """Return the last ``n`` records, oldest first.

        ``n=0`` or ``None`` returns every record (in chronological order).
        Malformed lines are skipped, not raised.
        """
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        line = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(line, dict):
                        records.append(line)
        except OSError:
            logger.warning("Failed to read run metrics", exc_info=True)
            return []
        if n and n > 0:
            return records[-n:]
        return records

    def stats(self) -> dict[str, Any]:
        """Small summary: line count, first/last ts, total tokens (if any)."""
        all_recs = self.load_recent()
        if not all_recs:
            return {"lines": 0}
        total_prompt = sum(
            int(r.get("total_prompt_tokens") or 0) for r in all_recs
        )
        total_completion = sum(
            int(r.get("total_completion_tokens") or 0) for r in all_recs
        )
        return {
            "lines": len(all_recs),
            "first_ts": all_recs[0].get("ts"),
            "last_ts": all_recs[-1].get("ts"),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
        }

    # ── internal truncation ─────────────────────────────────────────

    def _maybe_truncate_locked(self) -> None:
        """Enforce the ``max_lines`` cap in-place (called under the lock)."""
        # Read the current tail to check if we're over budget
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            return
        if len(lines) <= self._max_lines:
            return
        keep = lines[-self._max_lines:]
        try:
            # Atomic rewrite via tempfile + os.replace, matching config.py
            fd, tmp = tempfile.mkstemp(
                dir=str(self._dir), prefix=f".{self._filename}.", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.writelines(keep)
                os.replace(tmp, str(self.path))
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError:
            logger.warning("Run-metrics truncation failed", exc_info=True)


# ── singleton ───────────────────────────────────────────────────────────────
_store: RunMetricsStore | None = None


def get_run_metrics_store() -> RunMetricsStore:
    """Return the process-wide :class:`RunMetricsStore`, creating it on first use."""
    global _store
    if _store is None:
        _store = RunMetricsStore()
    return _store


def reset_run_metrics_store() -> None:
    """Drop the process-wide singleton (primarily for tests)."""
    global _store
    _store = None
