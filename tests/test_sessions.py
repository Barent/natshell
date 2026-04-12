"""Tests for conversation session persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from natshell.session import AmbiguousSessionID, SessionManager

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def session_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for session storage."""
    return tmp_path / "sessions"


@pytest.fixture()
def mgr(session_dir: Path) -> SessionManager:
    """Return a SessionManager pointing at a temp directory."""
    return SessionManager(session_dir=session_dir)


SAMPLE_MESSAGES: list[dict] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, world!"},
    {"role": "assistant", "content": "Hi there!"},
]

SAMPLE_ENGINE_INFO: dict = {
    "engine_type": "remote",
    "model_name": "qwen3:4b",
    "base_url": "http://localhost:11434/v1",
    "n_ctx": 32768,
}


# ── Save / Load roundtrip ────────────────────────────────────────────


class TestSaveLoad:
    def test_save_returns_id(self, mgr: SessionManager) -> None:
        sid = mgr.save(SAMPLE_MESSAGES, engine_info=SAMPLE_ENGINE_INFO)
        assert isinstance(sid, str)
        assert len(sid) == 32  # uuid hex

    def test_roundtrip(self, mgr: SessionManager) -> None:
        sid = mgr.save(SAMPLE_MESSAGES, engine_info=SAMPLE_ENGINE_INFO, name="test chat")
        data = mgr.load(sid)
        assert data is not None
        assert data["id"] == sid
        assert data["name"] == "test chat"
        assert data["messages"] == SAMPLE_MESSAGES
        assert data["engine_info"] == SAMPLE_ENGINE_INFO
        assert "created" in data
        assert "updated" in data

    def test_save_creates_directory(self, session_dir: Path, mgr: SessionManager) -> None:
        """SESSION_DIR is created on first save if it doesn't exist."""
        assert not session_dir.exists()
        mgr.save(SAMPLE_MESSAGES)
        assert session_dir.is_dir()

    def test_overwrite_preserves_created(self, mgr: SessionManager) -> None:
        sid = mgr.save(SAMPLE_MESSAGES, name="v1")
        data_v1 = mgr.load(sid)
        assert data_v1 is not None
        created_v1 = data_v1["created"]

        # Overwrite with same ID
        mgr.save(SAMPLE_MESSAGES, name="v2", session_id=sid)
        data_v2 = mgr.load(sid)
        assert data_v2 is not None
        assert data_v2["created"] == created_v1
        assert data_v2["name"] == "v2"

    def test_save_writes_valid_json(self, mgr: SessionManager, session_dir: Path) -> None:
        sid = mgr.save(SAMPLE_MESSAGES)
        path = session_dir / f"{sid}.json"
        data = json.loads(path.read_text())
        assert data["id"] == sid


# ── Load nonexistent ──────────────────────────────────────────────────


class TestLoadEdgeCases:
    def test_load_nonexistent_returns_none(self, mgr: SessionManager) -> None:
        # Use a valid-format hex ID that doesn't exist on disk
        assert mgr.load("00000000000000000000000000000000") is None

    def test_load_corrupt_file_returns_none(
        self, mgr: SessionManager, session_dir: Path
    ) -> None:
        session_dir.mkdir(parents=True, exist_ok=True)
        sid = "00000000000000000000000000000bad"
        bad = session_dir / f"{sid}.json"
        bad.write_text("not valid json {{{")
        assert mgr.load(sid) is None


# ── list_sessions ─────────────────────────────────────────────────────


class TestListSessions:
    def test_list_empty(self, mgr: SessionManager) -> None:
        assert mgr.list_sessions() == []

    def test_list_returns_correct_info(self, mgr: SessionManager) -> None:
        sid = mgr.save(SAMPLE_MESSAGES, name="my session")
        sessions = mgr.list_sessions()
        assert len(sessions) == 1
        s = sessions[0]
        assert s["id"] == sid
        assert s["name"] == "my session"
        assert s["message_count"] == len(SAMPLE_MESSAGES)
        assert "created" in s
        assert "updated" in s

    def test_list_sorted_by_updated(self, mgr: SessionManager) -> None:
        sid1 = mgr.save([{"role": "user", "content": "first"}], name="first")
        sid2 = mgr.save([{"role": "user", "content": "second"}], name="second")
        sessions = mgr.list_sessions()
        # Most recently updated first
        assert sessions[0]["id"] == sid2
        assert sessions[1]["id"] == sid1

    def test_list_skips_corrupt_files(
        self, mgr: SessionManager, session_dir: Path
    ) -> None:
        mgr.save(SAMPLE_MESSAGES, name="good")
        session_dir.mkdir(parents=True, exist_ok=True)
        bad = session_dir / "bad.json"
        bad.write_text("corrupted")
        sessions = mgr.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["name"] == "good"


# ── delete ────────────────────────────────────────────────────────────


class TestDelete:
    def test_delete_removes_file(self, mgr: SessionManager, session_dir: Path) -> None:
        sid = mgr.save(SAMPLE_MESSAGES)
        assert (session_dir / f"{sid}.json").exists()
        result = mgr.delete(sid)
        assert result is True
        assert not (session_dir / f"{sid}.json").exists()

    def test_delete_nonexistent_returns_false(self, mgr: SessionManager) -> None:
        assert mgr.delete("00000000000000000000000000000000") is False

    def test_delete_then_load_returns_none(self, mgr: SessionManager) -> None:
        sid = mgr.save(SAMPLE_MESSAGES)
        mgr.delete(sid)
        assert mgr.load(sid) is None


# ── Auto-generated name ──────────────────────────────────────────────


class TestAutoName:
    def test_auto_name_from_first_user_message(self, mgr: SessionManager) -> None:
        sid = mgr.save(SAMPLE_MESSAGES)
        data = mgr.load(sid)
        assert data is not None
        assert data["name"] == "Hello, world!"

    def test_auto_name_truncates_long_message(self, mgr: SessionManager) -> None:
        long_msg = "x" * 100
        messages = [{"role": "user", "content": long_msg}]
        sid = mgr.save(messages)
        data = mgr.load(sid)
        assert data is not None
        assert len(data["name"]) == 60
        assert data["name"].endswith("...")

    def test_auto_name_fallback_when_no_user_message(self, mgr: SessionManager) -> None:
        messages = [{"role": "system", "content": "System only"}]
        sid = mgr.save(messages)
        data = mgr.load(sid)
        assert data is not None
        assert data["name"].startswith("session-")

    def test_custom_name_overrides_auto(self, mgr: SessionManager) -> None:
        sid = mgr.save(SAMPLE_MESSAGES, name="My Custom Name")
        data = mgr.load(sid)
        assert data is not None
        assert data["name"] == "My Custom Name"


# ── Security ──────────────────────────────────────────────────────────


class TestSessionSecurity:
    def test_path_traversal_rejected_save(self, mgr: SessionManager) -> None:
        with pytest.raises(ValueError, match="Invalid session ID"):
            mgr.save(SAMPLE_MESSAGES, session_id="../../etc/passwd")

    def test_path_traversal_rejected_load(self, mgr: SessionManager) -> None:
        with pytest.raises(ValueError, match="Invalid session ID"):
            mgr.load("../../etc/passwd")

    def test_path_traversal_rejected_delete(self, mgr: SessionManager) -> None:
        with pytest.raises(ValueError, match="Invalid session ID"):
            mgr.delete("../../etc/passwd")

    def test_non_hex_id_rejected(self, mgr: SessionManager) -> None:
        with pytest.raises(ValueError, match="Invalid session ID"):
            mgr.save(SAMPLE_MESSAGES, session_id="not-a-valid-session-id!!")

    def test_valid_uuid_hex_accepted(self, mgr: SessionManager) -> None:
        sid = mgr.save(SAMPLE_MESSAGES, session_id="abcdef0123456789abcdef0123456789")
        assert sid == "abcdef0123456789abcdef0123456789"
        assert mgr.load(sid) is not None

    def test_oversized_session_raises(self, session_dir: Path) -> None:
        small_mgr = SessionManager(session_dir=session_dir, max_size=100)
        big_messages = [{"role": "user", "content": "x" * 200}]
        with pytest.raises(RuntimeError, match="Session too large"):
            small_mgr.save(big_messages)

    def test_directory_permissions_are_0o700(
        self, mgr: SessionManager, session_dir: Path
    ) -> None:
        mgr.save(SAMPLE_MESSAGES)
        mode = session_dir.stat().st_mode & 0o777
        assert mode == 0o700


# ── Short-prefix loading ──────────────────────────────────────────────


class TestShortPrefixLoad:
    def test_load_by_12_char_prefix(self, mgr: SessionManager) -> None:
        """The 12-char prefix displayed by /sessions should resolve to the full ID."""
        sid = mgr.save(SAMPLE_MESSAGES, name="prefix test")
        data = mgr.load(sid[:12])
        assert data is not None
        assert data["id"] == sid
        assert data["name"] == "prefix test"

    def test_load_by_minimum_prefix(self, mgr: SessionManager) -> None:
        """A 4-char prefix is the minimum accepted length."""
        sid = mgr.save(SAMPLE_MESSAGES, session_id="abcd" + "0" * 28)
        data = mgr.load("abcd")
        assert data is not None
        assert data["id"] == sid

    def test_load_too_short_prefix_raises(self, mgr: SessionManager) -> None:
        """Prefixes shorter than 4 chars are rejected as invalid."""
        mgr.save(SAMPLE_MESSAGES)
        with pytest.raises(ValueError, match="Invalid session ID"):
            mgr.load("abc")

    def test_load_nonhex_prefix_raises(self, mgr: SessionManager) -> None:
        """Non-hex characters in a prefix are rejected."""
        mgr.save(SAMPLE_MESSAGES)
        with pytest.raises(ValueError, match="Invalid session ID"):
            mgr.load("xyz1")

    def test_load_prefix_no_match_returns_none(self, mgr: SessionManager) -> None:
        """Valid prefix that matches nothing on disk returns None, not a crash."""
        mgr.save(SAMPLE_MESSAGES)
        assert mgr.load("ffff") is None

    def test_load_ambiguous_prefix_raises(self, mgr: SessionManager) -> None:
        """When a prefix matches multiple sessions, raise AmbiguousSessionID."""
        sid1 = mgr.save(SAMPLE_MESSAGES, session_id="abcd1234" + "0" * 24)
        sid2 = mgr.save(SAMPLE_MESSAGES, session_id="abcd5678" + "0" * 24)
        with pytest.raises(AmbiguousSessionID) as excinfo:
            mgr.load("abcd")
        assert set(excinfo.value.candidates) == {sid1, sid2}
        assert excinfo.value.prefix == "abcd"

    def test_ambiguous_error_is_valueerror_subclass(
        self, mgr: SessionManager
    ) -> None:
        """AmbiguousSessionID inherits from ValueError for backwards compat."""
        mgr.save(SAMPLE_MESSAGES, session_id="abcd1234" + "0" * 24)
        mgr.save(SAMPLE_MESSAGES, session_id="abcd5678" + "0" * 24)
        with pytest.raises(ValueError):
            mgr.load("abcd")

    def test_load_full_id_does_not_scan(self, mgr: SessionManager) -> None:
        """Full 32-hex IDs take the fast path and skip prefix resolution.

        A valid-format ID with no matching file should return None, not
        attempt to glob-match the directory.
        """
        mgr.save(SAMPLE_MESSAGES)
        assert mgr.load("0" * 32) is None

    def test_load_ignores_non_session_files_in_dir(
        self, mgr: SessionManager, session_dir: Path
    ) -> None:
        """A stray file named `abcd_notes.json` must not satisfy an `abcd` lookup."""
        sid = mgr.save(SAMPLE_MESSAGES, session_id="abcd" + "0" * 28)
        # Drop a stray file with a matching prefix but non-hex stem
        stray = session_dir / "abcd_notes.json"
        stray.write_text('{"not": "a session"}')
        data = mgr.load("abcd")
        assert data is not None
        assert data["id"] == sid

    def test_save_still_requires_full_id(self, mgr: SessionManager) -> None:
        """save() and delete() remain strict — no prefix-matching for writes."""
        with pytest.raises(ValueError, match="Invalid session ID"):
            mgr.save(SAMPLE_MESSAGES, session_id="abcd")

    def test_delete_still_requires_full_id(self, mgr: SessionManager) -> None:
        mgr.save(SAMPLE_MESSAGES, session_id="abcd" + "0" * 28)
        with pytest.raises(ValueError, match="Invalid session ID"):
            mgr.delete("abcd")
