"""Tests for the plain-file compaction memory helper."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from natshell.agent import memory_files
from natshell.agent.memory_files import (
    CHUNK_MAX_BYTES,
    SHORT_HASH_LEN,
    _sha_prefix,
    _truncate_oversize,
    cleanup_session,
    healthy,
    prune_old,
    reset_health,
    session_memory_dir,
    total_size_bytes,
    write_chunk,
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path: Path) -> None:
    """Redirect MEMORY_ROOT and reset health state for every test."""
    monkeypatch.setattr(memory_files, "MEMORY_ROOT", tmp_path / "memory")
    reset_health()
    yield
    reset_health()


# ── Hashing + truncation helpers ─────────────────────────────────────


class TestHashHelpers:
    def test_sha_prefix_length(self):
        assert len(_sha_prefix("hello")) == SHORT_HASH_LEN

    def test_sha_prefix_deterministic(self):
        assert _sha_prefix("same") == _sha_prefix("same")

    def test_sha_prefix_varies_with_content(self):
        assert _sha_prefix("a") != _sha_prefix("b")

    def test_truncate_small_content_passthrough(self):
        assert _truncate_oversize("short") == "short"

    def test_truncate_oversize_inserts_marker(self):
        big = "x" * (CHUNK_MAX_BYTES + 5000)
        result = _truncate_oversize(big)
        assert "bytes elided" in result
        # Head and tail both preserved.
        assert result.startswith("x")
        assert result.endswith("x")
        # Total size is bounded (head + tail + marker).
        assert len(result.encode("utf-8")) < CHUNK_MAX_BYTES + 1024


# ── write_chunk round-trip ───────────────────────────────────────────


class TestWriteChunk:
    def test_returns_path(self, tmp_path: Path):
        path = write_chunk("s1", "hello world")
        assert path is not None
        assert path.exists()
        assert path.suffix == ".txt"

    def test_filename_is_sha_prefix(self, tmp_path: Path):
        content = "deterministic content"
        path = write_chunk("s1", content)
        assert path is not None
        assert path.stem == _sha_prefix(content)

    def test_content_round_trip(self, tmp_path: Path):
        original = "line 1\nline 2\nline 3"
        path = write_chunk("s1", original)
        assert path is not None
        assert path.read_text() == original

    def test_idempotent_same_content(self, tmp_path: Path):
        p1 = write_chunk("s1", "same")
        p2 = write_chunk("s1", "same")
        assert p1 == p2

    def test_empty_content_returns_none(self, tmp_path: Path):
        assert write_chunk("s1", "") is None

    def test_empty_session_returns_none(self, tmp_path: Path):
        assert write_chunk("", "content") is None

    def test_per_session_scoping(self, tmp_path: Path):
        p1 = write_chunk("alpha", "shared content")
        p2 = write_chunk("beta", "shared content")
        assert p1 is not None and p2 is not None
        # Same filename (content-addressed), different session directories.
        assert p1.name == p2.name
        assert p1.parent != p2.parent

    def test_oversize_content_is_truncated_on_disk(self, tmp_path: Path):
        big = "x" * (CHUNK_MAX_BYTES * 2)
        path = write_chunk("s1", big)
        assert path is not None
        body = path.read_text()
        assert "bytes elided" in body
        assert len(body.encode("utf-8")) < CHUNK_MAX_BYTES + 1024

    @pytest.mark.skipif(os.name != "posix", reason="Unix-only perm check")
    def test_file_and_dir_permissions(self, tmp_path: Path):
        path = write_chunk("s1", "perms test")
        assert path is not None
        file_mode = path.stat().st_mode & 0o777
        dir_mode = path.parent.stat().st_mode & 0o777
        assert file_mode == 0o600
        assert dir_mode == 0o700


# ── Session directory helpers ────────────────────────────────────────


class TestSessionDir:
    def test_session_memory_dir_format(self, tmp_path: Path):
        path = session_memory_dir("abc123")
        assert path.name == "abc123"
        assert path.parent == memory_files.MEMORY_ROOT

    def test_session_dir_not_created_by_accessor(self, tmp_path: Path):
        # session_memory_dir is a pure path accessor.
        path = session_memory_dir("notyet")
        assert not path.exists()


# ── cleanup_session ─────────────────────────────────────────────────


class TestCleanupSession:
    def test_removes_directory(self, tmp_path: Path):
        write_chunk("s1", "a")
        write_chunk("s1", "b")
        dir_path = session_memory_dir("s1")
        assert dir_path.exists()
        cleanup_session("s1")
        assert not dir_path.exists()

    def test_missing_session_noop(self, tmp_path: Path):
        # Must not raise when the directory never existed.
        cleanup_session("never-existed")

    def test_empty_session_id_noop(self, tmp_path: Path):
        cleanup_session("")

    def test_does_not_touch_other_sessions(self, tmp_path: Path):
        write_chunk("alpha", "keep me")
        write_chunk("beta", "delete me")
        cleanup_session("beta")
        assert session_memory_dir("alpha").exists()
        assert not session_memory_dir("beta").exists()


# ── prune_old ───────────────────────────────────────────────────────


class TestPruneOld:
    def test_prunes_aged_chunks(self, tmp_path: Path):
        path = write_chunk("s1", "old")
        assert path is not None
        # Stamp the file far into the past.
        past = time.time() - 60 * 86400
        os.utime(path, (past, past))
        removed = prune_old(max_age_days=30)
        assert removed == 1
        assert not path.exists()

    def test_keeps_fresh_chunks(self, tmp_path: Path):
        path = write_chunk("s1", "fresh")
        assert path is not None
        removed = prune_old(max_age_days=30)
        assert removed == 0
        assert path.exists()

    def test_removes_empty_session_dirs(self, tmp_path: Path):
        path = write_chunk("s1", "only one")
        assert path is not None
        past = time.time() - 60 * 86400
        os.utime(path, (past, past))
        prune_old(max_age_days=30)
        assert not session_memory_dir("s1").exists()

    def test_missing_root_noop(self, tmp_path: Path):
        # No MEMORY_ROOT yet — must not raise.
        assert prune_old(max_age_days=30) == 0

    def test_zero_days_disables_pruning(self, tmp_path: Path):
        path = write_chunk("s1", "x")
        assert path is not None
        assert prune_old(max_age_days=0) == 0
        assert path.exists()


# ── Failure latch ───────────────────────────────────────────────────


class TestFailureLatch:
    def test_starts_healthy(self):
        assert healthy() is True

    def test_flips_off_after_three_failures(self):
        for _ in range(3):
            memory_files._record_failure(OSError("boom"))
        assert healthy() is False

    def test_success_resets_counter(self):
        memory_files._record_failure(OSError("boom"))
        assert memory_files._consecutive_failures == 1
        memory_files._record_success()
        assert memory_files._consecutive_failures == 0

    def test_disabled_feature_returns_none(self, tmp_path: Path):
        for _ in range(3):
            memory_files._record_failure(OSError("boom"))
        assert healthy() is False
        assert write_chunk("s1", "content") is None

    def test_write_failure_increments_counter(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # Force Path.write_text to raise by redirecting MEMORY_ROOT to a
        # file (not a directory) so mkdir fails.
        monkeypatch.setattr(
            memory_files, "MEMORY_ROOT", tmp_path / "not-a-dir"
        )
        (tmp_path / "not-a-dir").write_text("blocking file")
        result = write_chunk("s1", "content")
        assert result is None
        assert memory_files._consecutive_failures == 1


# ── total_size_bytes ────────────────────────────────────────────────


class TestTotalSize:
    def test_empty_root(self, tmp_path: Path):
        assert total_size_bytes() == 0

    def test_counts_all_sessions(self, tmp_path: Path):
        write_chunk("s1", "hello")
        write_chunk("s2", "world!")
        total = total_size_bytes()
        assert total == len("hello") + len("world!")
