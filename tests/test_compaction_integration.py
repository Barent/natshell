"""Integration test for plain-file retrieval-augmented compaction.

End-to-end exercises the wiring between AgentLoop, ContextManager, and
the memory_files helper: populate history with tool calls, trigger
compact_history, verify the summary contains absolute .txt paths, verify
those paths round-trip the original content via plain file reads, and
verify the fallback path activates when chunk writes fail.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from natshell.agent import memory_files
from natshell.agent.context import SystemContext
from natshell.agent.loop import AgentLoop
from natshell.config import AgentConfig, MemoryStoreConfig, SafetyConfig
from natshell.safety.classifier import SafetyClassifier
from natshell.tools.registry import create_default_registry


@pytest.fixture(autouse=True)
def _isolate_memory_root(monkeypatch, tmp_path: Path) -> None:
    """Redirect memory_files.MEMORY_ROOT at the tmp tree for every test."""
    monkeypatch.setattr(memory_files, "MEMORY_ROOT", tmp_path / "natshell_mem")
    memory_files.reset_health()
    yield
    memory_files.reset_health()


def _make_agent(enabled: bool = True) -> AgentLoop:
    engine = AsyncMock()
    tools = create_default_registry()
    safety_config = SafetyConfig(mode="confirm", always_confirm=[], blocked=[])
    safety = SafetyClassifier(safety_config)
    agent_config = AgentConfig(max_steps=15, temperature=0.3, max_tokens=2048)
    store_cfg = MemoryStoreConfig(
        enabled=enabled, max_size_mb=10, max_age_days=7
    )
    agent = AgentLoop(
        engine=engine,
        tools=tools,
        safety=safety,
        config=agent_config,
        memory_store_config=store_cfg,
    )
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


# ── Round-trip: compact then read the chunk file ────────────────────


class TestCompactThenRead:
    def test_summary_contains_txt_paths(self) -> None:
        agent = _make_agent()
        for i in range(8):
            agent.messages.extend(
                _tool_pair(
                    "execute_shell",
                    {"command": f"echo line-{i}"},
                    f"Exit code: 0\noutput line {i}",
                    f"tc{i}",
                )
            )
        agent.messages.append({"role": "user", "content": "what did you find?"})

        stats = agent.compact_history()
        assert stats["compacted"] is True
        assert stats["chunks_stored"] >= 7  # 8 pairs minus last 2 kept
        assert stats["used_refs"] is True
        summary_msg = agent.messages[1]
        assert ".txt" in summary_msg["content"]
        assert "Stored chunks" in summary_msg["content"]

    def test_chunk_file_round_trips_content(self, tmp_path: Path) -> None:
        agent = _make_agent()
        original = "Exit code: 0\nfound port 22 open on host-42\nelapsed 0.014s"
        agent.messages.extend(
            _tool_pair(
                "execute_shell",
                {"command": "nmap -sV host-42"},
                original,
                "tc1",
            )
        )
        for i in range(6):
            agent.messages.extend(
                _tool_pair(
                    "execute_shell",
                    {"command": f"filler-{i}"},
                    f"filler output {i}",
                    f"f{i}",
                )
            )
        agent.messages.append({"role": "user", "content": "summarize"})
        agent.compact_history()

        # Each ref line ends with "→ /abs/.../<sha12>.txt".  Extract the
        # absolute path after the arrow and verify Path.read_text()
        # returns the exact original content.
        summary = agent.messages[1]["content"]
        found_original = False
        for line in summary.splitlines():
            if "→" not in line or ".txt" not in line:
                continue
            path_str = line.split("→", 1)[1].strip()
            path = Path(path_str)
            if path.exists() and path.read_text() == original:
                found_original = True
                break
        assert found_original, (
            f"Original chunk content not found on disk. Summary was:\n{summary}"
        )

    def test_legacy_substring_preserved(self) -> None:
        """Other code may grep for '[Context compacted:' — must not break."""
        agent = _make_agent()
        for i in range(5):
            agent.messages.extend(
                _tool_pair(
                    "execute_shell",
                    {"command": f"c{i}"},
                    f"o{i}",
                    f"t{i}",
                )
            )
        agent.compact_history()
        content = agent.messages[1]["content"]
        assert "[Context compacted:" in content
        assert "messages replaced with summary" in content


# ── Auto-rehydration heuristic ───────────────────────────────────────


class TestAutoRehydration:
    def test_rehydrates_when_user_mentions_filename(self) -> None:
        agent = _make_agent()
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
                    "execute_shell", {"command": f"ls {i}"}, f"filler {i}", f"x{i}"
                )
            )
        agent.messages.append(
            {"role": "user", "content": "what was inside nginx.conf?"}
        )
        agent.compact_history()

        summary = agent.messages[1]["content"]
        assert "Auto-rehydrated chunks" in summary
        assert "worker_processes 4" in summary

    def test_no_rehydration_when_no_match(self) -> None:
        agent = _make_agent()
        for i in range(8):
            agent.messages.extend(
                _tool_pair(
                    "read_file",
                    {"path": f"/some/path/file{i}.txt"},
                    f"contents of file{i}",
                    f"tc{i}",
                )
            )
        agent.messages.append({"role": "user", "content": "tell me a joke"})
        agent.compact_history()
        summary = agent.messages[1]["content"]
        assert "Auto-rehydrated chunks" not in summary


# ── Failure injection ────────────────────────────────────────────────


class TestFallback:
    def test_fallback_when_write_fails(self, monkeypatch) -> None:
        agent = _make_agent()

        def fake_write(session_id: str, content: str):
            return None  # simulate every write failing silently

        monkeypatch.setattr(memory_files, "write_chunk", fake_write)

        for i in range(5):
            agent.messages.extend(
                _tool_pair("execute_shell", {"command": f"c{i}"}, f"o{i}", f"t{i}")
            )
        # Must complete without raising — falls through to legacy path.
        stats = agent.compact_history()
        assert stats["compacted"] is True
        assert stats["chunks_stored"] == 0
        assert "[Context compacted:" in agent.messages[1]["content"]

    def test_fallback_when_write_raises(self, monkeypatch) -> None:
        agent = _make_agent()

        def boom(session_id: str, content: str):
            raise RuntimeError("simulated disk failure")

        monkeypatch.setattr(memory_files, "write_chunk", boom)

        for i in range(5):
            agent.messages.extend(
                _tool_pair("execute_shell", {"command": f"c{i}"}, f"o{i}", f"t{i}")
            )
        stats = agent.compact_history()
        assert stats["compacted"] is True
        assert stats["chunks_stored"] == 0
        assert "[Context compacted:" in agent.messages[1]["content"]

    def test_disabled_feature_uses_legacy_path(self) -> None:
        agent = _make_agent(enabled=False)
        assert agent._memory_files_enabled is False
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
    def test_stats_includes_new_fields(self) -> None:
        agent = _make_agent()
        for i in range(5):
            agent.messages.extend(
                _tool_pair(
                    "execute_shell",
                    {"command": f"c{i}"},
                    f"long output {i}" * 5,
                    f"t{i}",
                )
            )
        stats = agent.compact_history()
        assert "chunks_stored" in stats
        assert "bytes_stored" in stats
        assert "used_refs" in stats
        assert stats["bytes_stored"] > 0


# ── System prompt wiring ─────────────────────────────────────────────


class TestSystemPromptWiring:
    def test_system_prompt_mentions_chunk_dir_when_enabled(self) -> None:
        agent = _make_agent(enabled=True)
        prompt = agent.messages[0]["content"]
        expected_dir = str(
            memory_files.session_memory_dir(agent.session_id)
        )
        assert expected_dir in prompt
        assert "read_file" in prompt or "search_files" in prompt

    def test_system_prompt_omits_rule_when_disabled(self) -> None:
        agent = _make_agent(enabled=False)
        prompt = agent.messages[0]["content"]
        # The explicit chunk-dir substitution should not appear.
        expected_dir = str(
            memory_files.session_memory_dir(agent.session_id)
        )
        assert expected_dir not in prompt
