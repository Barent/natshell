"""Tests for the recall_memory tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from natshell.agent.memory_store import MemoryStore
from natshell.tools import recall_memory as recall_mod
from natshell.tools.recall_memory import recall_memory


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(path=tmp_path / "memory" / "store.db")
    yield s
    s.close()


@pytest.fixture(autouse=True)
def reset_state():
    yield
    recall_mod.reset()


# ── Hash lookup ─────────────────────────────────────────────────────


class TestHashLookup:
    @pytest.mark.asyncio
    async def test_returns_chunk_by_full_hash(self, store: MemoryStore) -> None:
        h = store.put(
            role="tool",
            tool_name="execute_shell",
            content="port 22 open",
            session_id="s1",
        )
        recall_mod.configure(store, "s1")
        result = await recall_memory(hash=h)
        assert result.exit_code == 0
        assert "port 22 open" in result.output
        assert "[mem:" in result.output

    @pytest.mark.asyncio
    async def test_returns_chunk_by_short_hash(self, store: MemoryStore) -> None:
        h = store.put(role="tool", content="hello world", session_id="s1")
        recall_mod.configure(store, "s1")
        result = await recall_memory(hash=h[:12])
        assert result.exit_code == 0
        assert "hello world" in result.output

    @pytest.mark.asyncio
    async def test_unknown_hash_returns_error(
        self, store: MemoryStore
    ) -> None:
        recall_mod.configure(store, "s1")
        result = await recall_memory(hash="deadbeef")
        assert result.exit_code == 1
        assert "No chunk found" in result.error

    @pytest.mark.asyncio
    async def test_hash_scoped_to_session(self, store: MemoryStore) -> None:
        h = store.put(role="tool", content="secret", session_id="private")
        recall_mod.configure(store, "other")
        result = await recall_memory(hash=h)
        assert result.exit_code == 1


# ── Search ──────────────────────────────────────────────────────────


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_returns_matches(self, store: MemoryStore) -> None:
        store.put(
            role="tool",
            tool_name="execute_shell",
            content="installed package nginx",
            session_id="s1",
        )
        store.put(
            role="tool",
            tool_name="execute_shell",
            content="systemctl status nginx",
            session_id="s1",
        )
        store.put(
            role="tool",
            tool_name="execute_shell",
            content="ls /tmp",
            session_id="s1",
        )
        recall_mod.configure(store, "s1")
        result = await recall_memory(query="nginx")
        assert result.exit_code == 0
        assert "Found 2 chunk(s)" in result.output
        assert "nginx" in result.output

    @pytest.mark.asyncio
    async def test_search_filters_by_kind(self, store: MemoryStore) -> None:
        store.put(
            role="tool",
            tool_name="execute_shell",
            content="grep result",
            session_id="s1",
        )
        store.put(
            role="tool",
            tool_name="read_file",
            content="grep configuration",
            session_id="s1",
        )
        recall_mod.configure(store, "s1")
        result = await recall_memory(query="grep", kind="execute_shell")
        assert result.exit_code == 0
        assert result.output.count("[mem:") == 1

    @pytest.mark.asyncio
    async def test_no_match_returns_message(self, store: MemoryStore) -> None:
        store.put(role="tool", content="anything", session_id="s1")
        recall_mod.configure(store, "s1")
        result = await recall_memory(query="nonexistent")
        assert result.exit_code == 0
        assert "No stored chunks matched" in result.output

    @pytest.mark.asyncio
    async def test_query_sanitization(self, store: MemoryStore) -> None:
        store.put(role="tool", content="hello world", session_id="s1")
        recall_mod.configure(store, "s1")
        # Must not raise — special chars stripped before FTS5
        result = await recall_memory(query='"hello*":(^)')
        assert result.exit_code == 0
        assert "hello world" in result.output


# ── Argument validation ────────────────────────────────────────────


class TestArgValidation:
    @pytest.mark.asyncio
    async def test_no_args_returns_error(self, store: MemoryStore) -> None:
        recall_mod.configure(store, "s1")
        result = await recall_memory()
        assert result.exit_code == 1
        assert "hash" in result.error and "query" in result.error

    @pytest.mark.asyncio
    async def test_unhealthy_store_returns_error(self) -> None:
        # Configure with no store at all
        recall_mod.configure(None, "s1")
        result = await recall_memory(query="anything")
        assert result.exit_code == 1
        assert "unavailable" in result.error

    @pytest.mark.asyncio
    async def test_limit_clamping(self, store: MemoryStore) -> None:
        for i in range(30):
            store.put(role="tool", content=f"match {i}", session_id="s1")
        recall_mod.configure(store, "s1")
        result = await recall_memory(query="match", limit=99)
        # Internal cap is 20
        assert result.output.count("[mem:") <= 20
