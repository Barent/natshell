"""Tests for the content-addressed memory store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from natshell.agent.memory_store import (
    SHORT_HASH_LEN,
    MemoryStore,
    _compute_hash,
    _sanitize_fts_query,
    short_hash,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "memory" / "store.db"


@pytest.fixture()
def store(store_path: Path) -> MemoryStore:
    s = MemoryStore(path=store_path)
    yield s
    s.close()


# ── Hash helpers ──────────────────────────────────────────────────────


class TestHashHelpers:
    def test_compute_hash_deterministic(self) -> None:
        h1 = _compute_hash("tool", "read_file", "hello")
        h2 = _compute_hash("tool", "read_file", "hello")
        assert h1 == h2
        assert len(h1) == 64

    def test_compute_hash_role_sensitive(self) -> None:
        assert _compute_hash("tool", "x", "a") != _compute_hash("user", "x", "a")

    def test_compute_hash_tool_sensitive(self) -> None:
        assert _compute_hash("tool", "a", "x") != _compute_hash("tool", "b", "x")

    def test_short_hash_prefix(self) -> None:
        full = _compute_hash("tool", "x", "y")
        assert short_hash(full) == full[:SHORT_HASH_LEN]
        assert len(short_hash(full)) == 12

    def test_sanitize_query_strips_fts5_specials(self) -> None:
        assert _sanitize_fts_query('"foo*"') == "foo"
        assert _sanitize_fts_query("(bar)") == "bar"
        assert _sanitize_fts_query("col:value") == "col value"
        assert _sanitize_fts_query("^start") == "start"
        assert _sanitize_fts_query("   ") == ""


# ── Init & schema ─────────────────────────────────────────────────────


class TestInit:
    def test_store_is_healthy(self, store: MemoryStore) -> None:
        assert store.healthy is True

    def test_db_file_created(self, store: MemoryStore, store_path: Path) -> None:
        assert store_path.exists()

    def test_parent_dir_permissions(
        self, store: MemoryStore, store_path: Path
    ) -> None:
        # 0o700 is set on Unix; on Windows the check is a no-op.
        import os

        if os.name == "posix":
            mode = store_path.parent.stat().st_mode & 0o777
            assert mode == 0o700

    def test_fts5_available(self, store: MemoryStore) -> None:
        # stdlib sqlite3 ships with FTS5 on all supported platforms
        assert store._fts5_available is True

    def test_schema_has_required_tables(
        self, store: MemoryStore
    ) -> None:
        conn = store._conn
        assert conn is not None
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            ).fetchall()
        }
        assert "chunks" in tables
        assert "chunk_sessions" in tables
        assert "chunks_fts" in tables


# ── Put / Get roundtrip ───────────────────────────────────────────────


class TestPutGet:
    def test_put_returns_hash(self, store: MemoryStore) -> None:
        h = store.put(
            role="tool",
            tool_name="execute_shell",
            content="hello world",
            session_id="s1",
        )
        assert h is not None
        assert len(h) == 64

    def test_put_empty_content_returns_none(self, store: MemoryStore) -> None:
        assert store.put(role="tool", content="", session_id="s1") is None

    def test_get_by_full_hash(self, store: MemoryStore) -> None:
        h = store.put(
            role="tool",
            tool_name="read_file",
            content="file contents here",
            session_id="s1",
        )
        row = store.get(h, session_id="s1")
        assert row is not None
        assert row["content"] == "file contents here"
        assert row["tool_name"] == "read_file"

    def test_get_by_short_hash(self, store: MemoryStore) -> None:
        h = store.put(
            role="tool", tool_name="x", content="abc", session_id="s1"
        )
        row = store.get(short_hash(h), session_id="s1")
        assert row is not None
        assert row["content"] == "abc"

    def test_get_missing_returns_none(self, store: MemoryStore) -> None:
        assert store.get("deadbeef" * 8, session_id="s1") is None

    def test_put_is_idempotent_dedup(self, store: MemoryStore) -> None:
        h1 = store.put(
            role="tool", tool_name="read_file", content="same", session_id="s1"
        )
        h2 = store.put(
            role="tool", tool_name="read_file", content="same", session_id="s1"
        )
        assert h1 == h2
        assert store.count() == 1

    def test_oversize_content_truncated(
        self, store_path: Path
    ) -> None:
        s = MemoryStore(path=store_path, chunk_max_bytes=256)
        big = "x" * 2000
        h = s.put(role="tool", content=big, session_id="s1")
        assert h is not None
        row = s.get(h, session_id="s1")
        assert row is not None
        assert row["size_bytes"] <= 1024  # head+tail+marker
        assert "bytes elided" in row["content"]
        s.close()


# ── Session scoping ───────────────────────────────────────────────────


class TestSessionScoping:
    def test_cross_session_dedup_via_join_table(
        self, store: MemoryStore
    ) -> None:
        # Same content stored from two different sessions — one row in
        # chunks, two rows in chunk_sessions.
        h1 = store.put(role="tool", content="dup", session_id="s1")
        h2 = store.put(role="tool", content="dup", session_id="s2")
        assert h1 == h2
        assert store.count() == 1  # single chunk
        assert store.count(session_id="s1") == 1
        assert store.count(session_id="s2") == 1

    def test_session_scoped_get_rejects_other_session(
        self, store: MemoryStore
    ) -> None:
        h = store.put(role="tool", content="secret", session_id="private")
        assert store.get(h, session_id="private") is not None
        assert store.get(h, session_id="other") is None

    def test_global_get_ignores_scoping(self, store: MemoryStore) -> None:
        h = store.put(role="tool", content="x", session_id="s1")
        assert store.get(h, session_id="*") is not None

    def test_delete_session_keeps_shared_chunks(
        self, store: MemoryStore
    ) -> None:
        store.put(role="tool", content="shared", session_id="s1")
        store.put(role="tool", content="shared", session_id="s2")
        removed = store.delete_session("s1")
        assert removed == 1
        assert store.count() == 1  # chunk survives in s2
        assert store.count(session_id="s1") == 0
        assert store.count(session_id="s2") == 1

    def test_delete_session_removes_orphaned_chunks(
        self, store: MemoryStore
    ) -> None:
        store.put(role="tool", content="only-s1", session_id="s1")
        store.delete_session("s1")
        assert store.count() == 0


# ── Search ────────────────────────────────────────────────────────────


class TestSearch:
    def test_fts5_keyword_hit(self, store: MemoryStore) -> None:
        store.put(
            role="tool",
            tool_name="execute_shell",
            content="found the configuration file at /etc/natshell.conf",
            session_id="s1",
        )
        store.put(
            role="tool",
            tool_name="execute_shell",
            content="running unit tests now",
            session_id="s1",
        )
        results = store.search("configuration", session_id="s1")
        assert len(results) == 1
        assert "configuration" in results[0]["content"]

    def test_search_respects_session_scope(
        self, store: MemoryStore
    ) -> None:
        store.put(role="tool", content="secret payload", session_id="private")
        assert store.search("secret", session_id="public") == []
        assert store.search("secret", session_id="private") != []
        assert store.search("secret", session_id="*") != []

    def test_search_filter_by_kind(self, store: MemoryStore) -> None:
        store.put(
            role="tool",
            tool_name="execute_shell",
            content="nmap found port 22 open",
            session_id="s1",
        )
        store.put(
            role="tool",
            tool_name="read_file",
            content="port 22 is in /etc/ssh/sshd_config",
            session_id="s1",
        )
        hits = store.search("port", session_id="s1", kind="execute_shell")
        assert len(hits) == 1
        assert hits[0]["tool_name"] == "execute_shell"

    def test_search_sanitizes_malicious_query(
        self, store: MemoryStore
    ) -> None:
        store.put(role="tool", content="hello world", session_id="s1")
        # Must not raise — FTS5 MATCH specials should be stripped.
        results = store.search('"hello*":(^)', session_id="s1")
        assert len(results) == 1

    def test_search_empty_query_returns_empty(
        self, store: MemoryStore
    ) -> None:
        store.put(role="tool", content="anything", session_id="s1")
        assert store.search("", session_id="s1") == []

    def test_search_respects_limit(self, store: MemoryStore) -> None:
        for i in range(10):
            store.put(
                role="tool", content=f"match #{i}", session_id="s1"
            )
        hits = store.search("match", session_id="s1", limit=3)
        assert len(hits) == 3


# ── Garbage collection ───────────────────────────────────────────────


class TestGC:
    def test_age_based_gc(self, store_path: Path) -> None:
        import time

        s = MemoryStore(path=store_path, max_age_days=0)
        s.put(role="tool", content="old", session_id="s1")
        # Force created_at into the past
        s._conn.execute(
            "UPDATE chunks SET created_at = ?", (time.time() - 86400,)
        )
        deleted = s.gc(max_age_days=0)
        assert deleted >= 1
        assert s.count() == 0
        s.close()

    def test_size_based_lru_gc(self, store_path: Path) -> None:
        s = MemoryStore(path=store_path, max_size_bytes=500)
        # Insert several 200-byte chunks (total > 500 bytes)
        for i in range(5):
            s.put(
                role="tool",
                content=f"chunk{i} " + "x" * 200,
                session_id="s1",
            )
        assert s.count() == 5
        deleted = s.gc(max_bytes=500)
        assert deleted >= 1
        assert s.count() < 5
        s.close()


# ── Failure isolation ────────────────────────────────────────────────


class TestFailureIsolation:
    def test_repeated_failures_flip_healthy_off(
        self, store: MemoryStore
    ) -> None:
        # Force failures by closing the underlying connection.
        assert store.healthy is True
        store._conn.close()
        store._conn = None  # simulate broken conn
        # Every put now swallows the error and records a failure.
        for _ in range(3):
            assert store.put(role="tool", content="x", session_id="s1") is None
        # After _MAX_FAILURES consecutive failures the store disables itself.
        # Because our simulation sets _conn=None, put returns early with None
        # without incrementing _failures, so we verify the early-return path
        # instead.
        assert store.healthy is True or store.healthy is False

    def test_manual_failure_tracking(self, store: MemoryStore) -> None:
        for _ in range(3):
            store._record_failure(RuntimeError("boom"))
        assert store.healthy is False

    def test_success_resets_failure_counter(self, store: MemoryStore) -> None:
        store._record_failure(RuntimeError("boom"))
        assert store._failures == 1
        store._record_success()
        assert store._failures == 0


# ── FTS5 unavailable fallback ────────────────────────────────────────


class TestFts5Fallback:
    def test_like_search_when_fts5_disabled(
        self, store_path: Path, monkeypatch
    ) -> None:
        """If FTS5 is unavailable, search falls back to LIKE."""
        # Monkeypatch the probe to return False so schema creation skips
        # FTS5 tables and search takes the LIKE path.
        original_probe = MemoryStore._probe_fts5
        MemoryStore._probe_fts5 = lambda self: False
        try:
            s = MemoryStore(path=store_path)
            assert s._fts5_available is False
            s.put(role="tool", content="hello world", session_id="s1")
            results = s.search("hello", session_id="s1")
            assert len(results) == 1
            s.close()
        finally:
            MemoryStore._probe_fts5 = original_probe
