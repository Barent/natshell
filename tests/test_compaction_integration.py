"""Integration test for retrieval-augmented compaction.

End-to-end exercises the wiring between AgentLoop, ContextManager,
MemoryStore, and the recall_memory tool: populate history with tool
calls, trigger compact_history, verify [mem:<hash>] refs in the summary,
verify recall_memory round-trips the original content, and verify the
fallback path activates when the store is unhealthy.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from natshell.agent.context import SystemContext
from natshell.agent.loop import AgentLoop
from natshell.agent.memory_store import MemoryStore
from natshell.config import AgentConfig, MemoryStoreConfig, SafetyConfig
from natshell.safety.classifier import SafetyClassifier
from natshell.tools import recall_memory as recall_mod
from natshell.tools.recall_memory import recall_memory
from natshell.tools.registry import create_default_registry


def _make_agent_with_store(store_path: Path) -> AgentLoop:
    """Build an AgentLoop with a real MemoryStore at a temp path."""
    engine = AsyncMock()
    tools = create_default_registry()
    safety_config = SafetyConfig(mode="confirm", always_confirm=[], blocked=[])
    safety = SafetyClassifier(safety_config)
    agent_config = AgentConfig(max_steps=15, temperature=0.3, max_tokens=2048)
    store_cfg = MemoryStoreConfig(enabled=True, max_size_mb=10, max_age_days=7)
    agent = AgentLoop(
        engine=engine,
        tools=tools,
        safety=safety,
        config=agent_config,
        memory_store_config=store_cfg,
    )
    # Replace the auto-created store with one rooted at the test path so
    # the test never touches the user's real ~/.local/share/natshell.
    if agent.memory_store is not None:
        agent.memory_store.close()
    agent.memory_store = MemoryStore(path=store_path)
    recall_mod.configure(agent.memory_store, agent.session_id)

    agent.initialize(
        SystemContext(
            hostname="testhost",
            distro="Debian 13",
            kernel="6.12.0",
            username="testuser",
        )
    )
    return agent


def _tool_pair(name: str, args: dict, result: str, tc_id: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": tc_id, "content": result},
    ]


# ── Round-trip: compact then recall ──────────────────────────────────


class TestCompactThenRecall:
    @pytest.mark.asyncio
    async def test_summary_contains_mem_refs(self, tmp_path: Path) -> None:
        agent = _make_agent_with_store(tmp_path / "store.db")
        # 8 tool pairs of meaningful detail
        for i in range(8):
            agent.messages.extend(
                _tool_pair(
                    "execute_shell",
                    {"command": f"echo line-{i}"},
                    f"Exit code: 0\noutput line {i}",
                    f"tc{i}",
                )
            )
        # Add a final user message so last_2 has user-context
        agent.messages.append({"role": "user", "content": "what did you find?"})

        stats = agent.compact_history()
        assert stats["compacted"] is True
        assert stats["chunks_stored"] >= 7  # 8 pairs minus the last 2 kept
        assert stats["used_refs"] is True
        # The summary system message should contain at least one ref
        summary_msg = agent.messages[1]
        assert "[mem:" in summary_msg["content"]

    @pytest.mark.asyncio
    async def test_recall_round_trips_content(self, tmp_path: Path) -> None:
        agent = _make_agent_with_store(tmp_path / "store.db")
        for i in range(6):
            agent.messages.extend(
                _tool_pair(
                    "execute_shell",
                    {"command": f"nmap -sV host{i}"},
                    f"Exit code: 0\nfound port 22 open on host{i}",
                    f"tc{i}",
                )
            )
        agent.messages.append({"role": "user", "content": "summarize"})
        agent.compact_history()

        # Search via the tool — must round-trip the content verbatim.
        result = await recall_memory(query="port 22 open")
        assert result.exit_code == 0
        assert "port 22 open" in result.output

    @pytest.mark.asyncio
    async def test_legacy_substring_preserved(self, tmp_path: Path) -> None:
        """Other code may grep for '[Context compacted:' — must not break."""
        agent = _make_agent_with_store(tmp_path / "store.db")
        for i in range(5):
            agent.messages.extend(
                _tool_pair("execute_shell", {"command": f"c{i}"}, f"o{i}", f"t{i}")
            )
        agent.compact_history()
        assert "[Context compacted:" in agent.messages[1]["content"]
        assert "messages replaced with summary" in agent.messages[1]["content"]


# ── Auto-rehydration heuristic ───────────────────────────────────────


class TestAutoRehydration:
    @pytest.mark.asyncio
    async def test_rehydrates_when_user_mentions_filename(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent_with_store(tmp_path / "store.db")
        # Read multiple files
        agent.messages.extend(
            _tool_pair(
                "read_file",
                {"path": "/etc/nginx/nginx.conf"},
                "worker_processes 4;\nuser www-data;",
                "tc1",
            )
        )
        for i in range(5):
            agent.messages.extend(
                _tool_pair(
                    "execute_shell", {"command": f"ls {i}"}, f"file{i}", f"x{i}"
                )
            )
        # Newest user message references the nginx file by name
        agent.messages.append(
            {"role": "user", "content": "what was inside nginx.conf?"}
        )
        agent.compact_history()

        # The summary should now contain the inlined content of nginx.conf
        summary = agent.messages[1]["content"]
        assert "Auto-rehydrated chunks" in summary
        assert "worker_processes 4" in summary

    @pytest.mark.asyncio
    async def test_no_rehydration_when_no_match(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent_with_store(tmp_path / "store.db")
        for i in range(8):
            agent.messages.extend(
                _tool_pair(
                    "read_file",
                    {"path": f"/some/path/file{i}.txt"},
                    f"contents of file{i}",
                    f"tc{i}",
                )
            )
        agent.messages.append(
            {"role": "user", "content": "tell me a joke"}
        )
        agent.compact_history()
        summary = agent.messages[1]["content"]
        assert "Auto-rehydrated chunks" not in summary


# ── Failure injection ────────────────────────────────────────────────


class TestFallback:
    def test_fallback_when_store_put_raises(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        agent = _make_agent_with_store(tmp_path / "store.db")
        # Replace put with a function that always raises
        original_put = agent.memory_store.put

        def boom(**kwargs):
            raise RuntimeError("simulated SQLite failure")

        monkeypatch.setattr(agent.memory_store, "put", boom)

        for i in range(5):
            agent.messages.extend(
                _tool_pair("execute_shell", {"command": f"c{i}"}, f"o{i}", f"t{i}")
            )

        # Must complete without raising — falls through to legacy path
        stats = agent.compact_history()
        assert stats["compacted"] is True
        # No chunks stored when put always fails
        assert stats["chunks_stored"] == 0
        # The legacy summary substring should still appear
        assert "[Context compacted:" in agent.messages[1]["content"]

    def test_disabled_store_uses_legacy_path(self, tmp_path: Path) -> None:
        engine = AsyncMock()
        tools = create_default_registry()
        safety = SafetyClassifier(SafetyConfig(mode="confirm"))
        agent_config = AgentConfig(max_steps=15, temperature=0.3, max_tokens=2048)
        # memory_store disabled in config
        agent = AgentLoop(
            engine=engine,
            tools=tools,
            safety=safety,
            config=agent_config,
            memory_store_config=MemoryStoreConfig(enabled=False),
        )
        agent.initialize(
            SystemContext(
                hostname="h", distro="d", kernel="k", username="u"
            )
        )
        assert agent.memory_store is None
        for i in range(5):
            agent.messages.extend(
                _tool_pair("execute_shell", {"command": f"c{i}"}, f"o{i}", f"t{i}")
            )
        stats = agent.compact_history()
        assert stats["compacted"] is True
        assert stats["chunks_stored"] == 0
        assert stats["used_refs"] is False


# ── Stats display fields ─────────────────────────────────────────────


class TestStats:
    def test_stats_includes_new_fields(self, tmp_path: Path) -> None:
        agent = _make_agent_with_store(tmp_path / "store.db")
        for i in range(5):
            agent.messages.extend(
                _tool_pair(
                    "execute_shell", {"command": f"c{i}"}, f"long output {i}" * 5, f"t{i}"
                )
            )
        stats = agent.compact_history()
        assert "chunks_stored" in stats
        assert "bytes_stored" in stats
        assert "used_refs" in stats
        assert stats["bytes_stored"] > 0
