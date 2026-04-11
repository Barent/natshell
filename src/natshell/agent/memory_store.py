"""Content-addressed SQLite chunk store for retrieval-augmented compaction.

When ``AgentLoop.compact_history`` drops old messages, their content is stored
here verbatim instead of being lost.  The short summary that replaces them
contains ``[mem:<hash>]`` references, and the agent can call the
``recall_memory`` tool to fetch full content on demand.

Storage layout:
    ~/.local/share/natshell/memory/store.db

Schema:
    chunks          : content-addressed rows (SHA256 PK, deduped across sessions)
    chunk_sessions  : join table scoping chunks to a session_id
    chunks_fts      : external-content FTS5 virtual table over chunks.content

The store is always-on-optional: every public method is wrapped so that a
SQLite failure cannot break the agent loop.  After ``_MAX_FAILURES`` consecutive
errors the store's ``healthy`` flag flips off and callers must fall back.

This module is unrelated to ``natshell.agent.working_memory``/``MemoryConfig``,
which is a separate agents.md scratchpad feature.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from natshell.platform import data_dir as _data_dir

logger = logging.getLogger(__name__)

# Default DB location (shared — per-session co-location is Phase 6).
DEFAULT_STORE_DIR = _data_dir() / "memory"
DEFAULT_STORE_PATH = DEFAULT_STORE_DIR / "store.db"

# Prefix length shown to the LLM.  12 hex chars = 48 bits ≈ birthday collision
# at ~16M chunks — comfortable, and still only ~4 tokens in most tokenizers.
SHORT_HASH_LEN = 12

# After this many consecutive failures we disable the feature until restart.
_MAX_FAILURES = 3

# FTS5 MATCH reserved characters that we strip from user queries.
_FTS5_RESERVED_RE = re.compile(r'[\"\*\(\)\:\^]')


def short_hash(full_hash: str) -> str:
    """Return the display prefix used in `[mem:...]` references."""
    return full_hash[:SHORT_HASH_LEN]


def _compute_hash(role: str, tool_name: str, content: str) -> str:
    """SHA256 of role + tool_name + content for dedup.

    Tool *arguments* are intentionally excluded so that repeated reads of the
    same file (same content) collapse to a single chunk.
    """
    h = hashlib.sha256()
    h.update(role.encode("utf-8"))
    h.update(b"\0")
    h.update(tool_name.encode("utf-8"))
    h.update(b"\0")
    h.update(content.encode("utf-8"))
    return h.hexdigest()


def _sanitize_fts_query(query: str) -> str:
    """Strip FTS5 MATCH special characters so untrusted input never throws."""
    cleaned = _FTS5_RESERVED_RE.sub(" ", query)
    return " ".join(cleaned.split()).strip()


class MemoryStore:
    """SQLite-backed content-addressed chunk store with FTS5 search."""

    def __init__(
        self,
        path: Path | None = None,
        max_size_bytes: int = 50 * 1024 * 1024,
        max_age_days: int = 30,
        chunk_max_bytes: int = 64 * 1024,
    ) -> None:
        self.path: Path = path or DEFAULT_STORE_PATH
        self.max_size_bytes = max_size_bytes
        self.max_age_days = max_age_days
        self.chunk_max_bytes = chunk_max_bytes
        self._failures = 0
        self.healthy = False
        self._fts5_available = False
        self._conn: sqlite3.Connection | None = None

        try:
            self._ensure_dir()
            self._conn = sqlite3.connect(
                str(self.path),
                timeout=5.0,
                check_same_thread=False,
                isolation_level=None,  # autocommit
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._fts5_available = self._probe_fts5()
            self._create_schema()
            self.healthy = True
            logger.info(
                "MemoryStore ready at %s (fts5=%s)",
                self.path,
                self._fts5_available,
            )
        except Exception as exc:
            logger.warning("MemoryStore init failed, disabling feature: %s", exc)
            self.healthy = False
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    # ── setup ──────────────────────────────────────────────────────────

    def _ensure_dir(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass  # Windows or read-only mount — non-fatal

    def _probe_fts5(self) -> bool:
        assert self._conn is not None
        try:
            self._conn.execute("CREATE VIRTUAL TABLE temp.fts5_probe USING fts5(x)")
            self._conn.execute("DROP TABLE temp.fts5_probe")
            return True
        except sqlite3.OperationalError as exc:
            logger.info("FTS5 not available (%s) — falling back to LIKE search", exc)
            return False

    def _create_schema(self) -> None:
        assert self._conn is not None
        cur = self._conn
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                hash TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                kind TEXT,
                tool_name TEXT,
                args_json TEXT,
                content TEXT NOT NULL,
                file_path TEXT,
                file_mtime REAL,
                token_estimate INTEGER,
                size_bytes INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_sessions (
                hash TEXT NOT NULL,
                session_id TEXT NOT NULL,
                first_seen REAL NOT NULL,
                last_accessed REAL NOT NULL,
                PRIMARY KEY (hash, session_id),
                FOREIGN KEY (hash) REFERENCES chunks(hash) ON DELETE CASCADE
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunk_sessions_session "
            "ON chunk_sessions(session_id, last_accessed)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_created "
            "ON chunks(created_at)"
        )

        if self._fts5_available:
            # External-content FTS5 tied to chunks.rowid.  The triggers keep
            # the index in sync with insert/update/delete.
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    content,
                    content='chunks',
                    content_rowid='rowid',
                    tokenize='porter unicode61'
                )
                """
            )
            cur.execute(
                """
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, content)
                    VALUES (new.rowid, new.content);
                END
                """
            )
            cur.execute(
                """
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
                END
                """
            )

    # ── failure isolation helpers ──────────────────────────────────────

    def _record_failure(self, exc: Exception) -> None:
        self._failures += 1
        logger.warning(
            "MemoryStore operation failed (%d/%d): %s",
            self._failures,
            _MAX_FAILURES,
            exc,
        )
        if self._failures >= _MAX_FAILURES:
            logger.error(
                "MemoryStore disabled after %d consecutive failures",
                self._failures,
            )
            self.healthy = False

    def _record_success(self) -> None:
        if self._failures:
            self._failures = 0

    # ── public API ─────────────────────────────────────────────────────

    def put(
        self,
        *,
        role: str,
        content: str,
        session_id: str,
        tool_name: str = "",
        kind: str = "",
        args_json: str = "",
        file_path: str | None = None,
        file_mtime: float | None = None,
        token_estimate: int = 0,
    ) -> str | None:
        """Store a chunk and return its full SHA256 hash (or None on failure).

        Idempotent by content hash: repeated calls with identical
        (role, tool_name, content) collapse to a single row.  The
        chunk_sessions row is upserted so the same chunk may be referenced
        by multiple sessions.
        """
        if not self.healthy or self._conn is None:
            return None
        if not content:
            return None
        try:
            # Truncate oversize content with a head/tail marker.
            raw = content
            size = len(raw.encode("utf-8"))
            if size > self.chunk_max_bytes:
                half = self.chunk_max_bytes // 2 - 64
                raw = (
                    raw[:half]
                    + f"\n\n[... {size - self.chunk_max_bytes} bytes elided — "
                    f"use recall_memory with a query to search, or re-read the "
                    f"original file if still present ...]\n\n"
                    + raw[-half:]
                )
                size = len(raw.encode("utf-8"))

            full_hash = _compute_hash(role, tool_name, raw)
            now = time.time()

            self._conn.execute(
                """
                INSERT OR IGNORE INTO chunks
                    (hash, role, kind, tool_name, args_json, content,
                     file_path, file_mtime, token_estimate, size_bytes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    full_hash,
                    role,
                    kind or None,
                    tool_name or None,
                    args_json or None,
                    raw,
                    file_path,
                    file_mtime,
                    token_estimate,
                    size,
                    now,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO chunk_sessions (hash, session_id, first_seen, last_accessed)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(hash, session_id) DO UPDATE SET
                    last_accessed = excluded.last_accessed
                """,
                (full_hash, session_id, now, now),
            )
            self._record_success()
            return full_hash
        except Exception as exc:
            self._record_failure(exc)
            return None

    def get(
        self,
        full_or_short_hash: str,
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch a chunk by full or short hash.

        When ``session_id`` is provided, the lookup is scoped to that session
        (prevents cross-session reads).  Pass ``session_id="*"`` for a global
        search.  Updates ``last_accessed`` on hit for LRU tracking.
        """
        if not self.healthy or self._conn is None:
            return None
        if not full_or_short_hash:
            return None
        try:
            ident = full_or_short_hash.strip().lower()
            # Accept either full or prefix.
            if len(ident) == 64:
                like_expr = ident
                exact = True
            else:
                like_expr = ident + "%"
                exact = False

            if session_id is None or session_id == "*":
                if exact:
                    row = self._conn.execute(
                        "SELECT * FROM chunks WHERE hash = ?", (ident,)
                    ).fetchone()
                else:
                    row = self._conn.execute(
                        "SELECT * FROM chunks WHERE hash LIKE ? LIMIT 2",
                        (like_expr,),
                    ).fetchone()
            else:
                if exact:
                    row = self._conn.execute(
                        """
                        SELECT c.* FROM chunks c
                        JOIN chunk_sessions s ON s.hash = c.hash
                        WHERE c.hash = ? AND s.session_id = ?
                        """,
                        (ident, session_id),
                    ).fetchone()
                else:
                    row = self._conn.execute(
                        """
                        SELECT c.* FROM chunks c
                        JOIN chunk_sessions s ON s.hash = c.hash
                        WHERE c.hash LIKE ? AND s.session_id = ?
                        LIMIT 2
                        """,
                        (like_expr, session_id),
                    ).fetchone()

            if row is None:
                self._record_success()
                return None

            if session_id and session_id != "*":
                self._conn.execute(
                    "UPDATE chunk_sessions SET last_accessed = ? "
                    "WHERE hash = ? AND session_id = ?",
                    (time.time(), row["hash"], session_id),
                )
            self._record_success()
            return dict(row)
        except Exception as exc:
            self._record_failure(exc)
            return None

    def search(
        self,
        query: str,
        session_id: str | None = None,
        kind: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Keyword search over stored chunks.

        Uses FTS5 when available, falls back to LIKE otherwise.
        Session-scoped by default.  Pass ``session_id="*"`` to search globally.
        """
        if not self.healthy or self._conn is None:
            return []
        cleaned = _sanitize_fts_query(query)
        if not cleaned:
            return []
        limit = max(1, min(20, int(limit)))
        try:
            params: list[Any]
            if self._fts5_available:
                sql = (
                    "SELECT c.* FROM chunks_fts f "
                    "JOIN chunks c ON c.rowid = f.rowid "
                )
                where: list[str] = ["chunks_fts MATCH ?"]
                params = [cleaned]
                if session_id and session_id != "*":
                    sql += "JOIN chunk_sessions s ON s.hash = c.hash "
                    where.append("s.session_id = ?")
                    params.append(session_id)
                if kind:
                    where.append("(c.kind = ? OR c.tool_name = ?)")
                    params.extend([kind, kind])
                sql += "WHERE " + " AND ".join(where)
                sql += " ORDER BY rank LIMIT ?"
                params.append(limit)
            else:
                like = f"%{cleaned}%"
                sql = "SELECT c.* FROM chunks c "
                where = ["c.content LIKE ?"]
                params = [like]
                if session_id and session_id != "*":
                    sql += "JOIN chunk_sessions s ON s.hash = c.hash "
                    where.append("s.session_id = ?")
                    params.append(session_id)
                if kind:
                    where.append("(c.kind = ? OR c.tool_name = ?)")
                    params.extend([kind, kind])
                sql += "WHERE " + " AND ".join(where)
                sql += " ORDER BY c.created_at DESC LIMIT ?"
                params.append(limit)

            rows = self._conn.execute(sql, params).fetchall()
            self._record_success()
            return [dict(r) for r in rows]
        except Exception as exc:
            self._record_failure(exc)
            return []

    def gc(
        self,
        max_bytes: int | None = None,
        max_age_days: int | None = None,
    ) -> int:
        """Garbage collect aged/oversized chunks.  Returns number deleted."""
        if not self.healthy or self._conn is None:
            return 0
        max_bytes = max_bytes if max_bytes is not None else self.max_size_bytes
        max_age_days = (
            max_age_days if max_age_days is not None else self.max_age_days
        )
        deleted = 0
        try:
            cutoff = time.time() - max_age_days * 86400
            cur = self._conn.execute(
                "DELETE FROM chunks WHERE created_at < ?", (cutoff,)
            )
            deleted += cur.rowcount or 0

            # Size-based LRU eviction: delete oldest-accessed until under budget.
            total_size = self._db_size_bytes()
            if total_size > max_bytes:
                # Evict chunks whose newest session reference is oldest first.
                rows = self._conn.execute(
                    """
                    SELECT c.hash, c.size_bytes,
                           COALESCE(MAX(s.last_accessed), c.created_at) AS last_used
                    FROM chunks c
                    LEFT JOIN chunk_sessions s ON s.hash = c.hash
                    GROUP BY c.hash
                    ORDER BY last_used ASC
                    """
                ).fetchall()
                freed = 0
                target = total_size - max_bytes
                for r in rows:
                    if freed >= target:
                        break
                    self._conn.execute(
                        "DELETE FROM chunks WHERE hash = ?", (r["hash"],)
                    )
                    freed += r["size_bytes"] or 0
                    deleted += 1
            self._record_success()
            return deleted
        except Exception as exc:
            self._record_failure(exc)
            return deleted

    def delete_session(self, session_id: str) -> int:
        """Remove all chunk references for a session.  Returns rows deleted.

        Content-addressed chunks are kept if referenced by other sessions;
        they are only removed once no session references remain.
        """
        if not self.healthy or self._conn is None:
            return 0
        try:
            cur = self._conn.execute(
                "DELETE FROM chunk_sessions WHERE session_id = ?", (session_id,)
            )
            removed = cur.rowcount or 0
            # Orphan sweep
            self._conn.execute(
                """
                DELETE FROM chunks WHERE hash NOT IN (
                    SELECT hash FROM chunk_sessions
                )
                """
            )
            self._record_success()
            return removed
        except Exception as exc:
            self._record_failure(exc)
            return 0

    def count(self, session_id: str | None = None) -> int:
        if not self.healthy or self._conn is None:
            return 0
        try:
            if session_id and session_id != "*":
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM chunk_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            else:
                row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            return int(row[0]) if row else 0
        except Exception as exc:
            self._record_failure(exc)
            return 0

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ── internals ──────────────────────────────────────────────────────

    def _db_size_bytes(self) -> int:
        """Estimate total content size from the chunks table."""
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM chunks"
        ).fetchone()
        return int(row[0]) if row else 0
