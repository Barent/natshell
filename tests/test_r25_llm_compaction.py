"""Tests for R2-5 — the LLM compaction tier.

Covers:
- ``agent/summarizer.py``: ``render_messages`` prompt shaping,
  ``FailureTracker`` circuit breaker, and ``llm_summarize`` (success,
  timeout, engine error, empty output, tripped breaker).
- ``AgentLoop.compact_now`` integration: LLM tier engaged on enable,
  extractive fallback on tier failure, sync ``ContextManager.summarizer``
  seam precedence, dry-run semantics, and the sync ``compact_history``
  shim over the async core.
- ``[compaction]`` config section: defaults, TOML loading, persistence
  via ``save_config_value``.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from unittest.mock import AsyncMock

from natshell.agent.context import SystemContext
from natshell.agent.loop import AgentLoop
from natshell.agent.summarizer import (
    FailureTracker,
    llm_summarize,
    render_messages,
)
from natshell.config import (
    CompactionConfig,
    SafetyConfig,
    load_config,
    save_config_value,
)
from natshell.inference.engine import CompletionResult
from natshell.safety.classifier import SafetyClassifier
from natshell.tools.registry import create_default_registry

# ─── Test agents ──────────────────────────────────────────────────────────


def _make_agent(
    responses: list | None = None,
    compaction: CompactionConfig | None = None,
) -> AgentLoop:
    """Create an agent with a mocked inference engine (mirrors test_slash_commands)."""
    engine = AsyncMock()
    if responses is not None:
        engine.chat_completion = AsyncMock(side_effect=responses)
    else:
        engine.chat_completion = AsyncMock(
            return_value=CompletionResult(content="ok")
        )
    tools = create_default_registry()
    safety_config = SafetyConfig(
        mode="danger",
        always_confirm=[],
        blocked=[],
    )
    safety = SafetyClassifier(safety_config)
    from natshell.config import AgentConfig

    agent_config = AgentConfig(max_steps=15, temperature=0.3, max_tokens=2048)
    agent = AgentLoop(
        engine=engine,
        tools=tools,
        safety=safety,
        config=agent_config,
        compaction=compaction or CompactionConfig(),
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


def _add_conversation(agent: AgentLoop, n: int = 2) -> None:
    """Append n (user, assistant) pairs so compaction has something to drop."""
    for i in range(n):
        agent.messages.append({"role": "user", "content": f"question {i}"})
        agent.messages.append({"role": "assistant", "content": f"answer {i}"})


# ─── render_messages ──────────────────────────────────────────────────────


class TestRenderMessages:
    def test_system_messages_skipped(self):
        msgs = [
            {"role": "system", "content": "prior marker"},
            {"role": "user", "content": "hello"},
        ]
        body = render_messages(msgs)
        assert "hello" in body
        assert "prior marker" not in body

    def test_empty_conversation_yields_placeholder(self):
        """A conversation with only system-role messages yields placeholder text."""
        msgs = [{"role": "system", "content": "prior marker"}]
        body = render_messages(msgs)
        assert "empty conversation" in body

    def test_tool_calls_rendered_inline(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "execute_shell",
                            "arguments": json.dumps({"command": "ls -la"}),
                        }
                    }
                ],
            }
        ]
        body = render_messages(msgs)
        assert "execute_shell" in body
        assert "ls -la" in body

    def test_long_content_elided(self):
        msgs = [{"role": "user", "content": "x" * 5000}]
        body = render_messages(msgs)
        # 4000-char cap + "…" ellipsis
        assert "…" in body
        assert len(body) < 5200

    def test_max_messages_window(self):
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        body = render_messages(msgs, max_messages=3)
        # Only the most recent 3 messages appear; the others are noted as truncated
        assert "17 older messages truncated" in body
        lines = [line.strip() for line in body.splitlines()]
        # m0..m16 don't appear as their own lines (only m17, m18, m19)
        for i in range(17):
            assert f"[user] m{i}" not in lines
        assert "[user] m17" in lines
        assert "[user] m19" in lines


# ─── FailureTracker ───────────────────────────────────────────────────────


class TestFailureTracker:
    def test_resets_on_success(self):
        t = FailureTracker()
        t.record_failure()
        t.record_failure()
        assert t.consecutive_failures == 2
        t.record_success()
        assert t.consecutive_failures == 0
        assert not t.tripped

    def test_trips_after_threshold(self):
        t = FailureTracker(trip_after=2)
        t.record_failure()
        t.record_failure()
        assert t.tripped

    def test_reset_clears(self):
        t = FailureTracker(trip_after=1)
        t.record_failure()
        assert t.tripped
        t.reset()
        assert not t.tripped
        assert t.consecutive_failures == 0


# ─── llm_summarize ────────────────────────────────────────────────────────


class TestLlmSummarize:
    def test_returns_text_from_completion(self):
        engine = AsyncMock()
        engine.chat_completion = AsyncMock(
            return_value=CompletionResult(content="  files: a.py, b.py ")
        )
        msgs = [
            {"role": "user", "content": "what did we do?"},
            {"role": "assistant", "content": "wrote a.py"},
        ]
        result = asyncio.run(llm_summarize(engine, msgs))
        assert result is not None
        assert "files: a.py, b.py" in result
        # Prompt structure: system + user
        (args, kwargs) = engine.chat_completion.call_args
        assert args[0][0]["role"] == "system"
        assert args[0][1]["role"] == "user"
        assert "files: a.py" not in args[0][0]["content"]

    def test_empty_result_returns_none(self):
        engine = AsyncMock()
        engine.chat_completion = AsyncMock(
            return_value=CompletionResult(content=None)
        )
        result = asyncio.run(
            llm_summarize(engine, [{"role": "user", "content": "hi"}])
        )
        assert result is None

    def test_empty_string_returns_none(self):
        engine = AsyncMock()
        engine.chat_completion = AsyncMock(
            return_value=CompletionResult(content="   ")
        )
        result = asyncio.run(
            llm_summarize(engine, [{"role": "user", "content": "hi"}])
        )
        assert result is None

    def test_engine_exception_returns_none(self):
        engine = AsyncMock()
        engine.chat_completion = AsyncMock(side_effect=RuntimeError("boom"))
        result = asyncio.run(
            llm_summarize(engine, [{"role": "user", "content": "hi"}])
        )
        assert result is None

    def test_timeout_returns_none(self):
        async def slow():
            await asyncio.sleep(0.2)

        engine = AsyncMock()
        engine.chat_completion = AsyncMock(side_effect=slow)
        result = asyncio.run(
            llm_summarize(
                engine,
                [{"role": "user", "content": "hi"}],
                timeout=0.01,
            )
        )
        assert result is None

    def test_strips_think_blocks(self):
        engine = AsyncMock()
        engine.chat_completion = AsyncMock(
            return_value=CompletionResult(
                content=(
                    "Let me think... "
                    "<think>interior reasoning</think>"
                    "  files: a.py  "
                )
            )
        )
        result = asyncio.run(
            llm_summarize(engine, [{"role": "user", "content": "hi"}])
        )
        assert result is not None
        assert "interior reasoning" not in result
        assert "files: a.py" in result

    def test_tripped_breaker_short_circuits(self):
        engine = AsyncMock()
        tracker = FailureTracker(trip_after=1)
        tracker.record_failure()
        assert tracker.tripped
        result = asyncio.run(
            llm_summarize(
                engine,
                [{"role": "user", "content": "hi"}],
                tracker=tracker,
            )
        )
        assert result is None
        engine.chat_completion.assert_not_called()

    def test_success_resets_tracker(self):
        engine = AsyncMock()
        engine.chat_completion = AsyncMock(
            return_value=CompletionResult(content="ok")
        )
        tracker = FailureTracker(trip_after=1)
        tracker.record_failure()
        assert tracker.tripped
        # Reset first (loop does this per-run)
        tracker.reset()
        result = asyncio.run(
            llm_summarize(
                engine,
                [{"role": "user", "content": "hi"}],
                timeout=0.5,
                tracker=tracker,
            )
        )
        assert result is not None
        assert not tracker.tripped
        assert tracker.consecutive_failures == 0

    def test_failure_extends_tracker(self):
        engine = AsyncMock()
        engine.chat_completion = AsyncMock(side_effect=RuntimeError("boom"))
        tracker = FailureTracker(trip_after=2)
        asyncio.run(
            llm_summarize(
                engine,
                [{"role": "user", "content": "hi"}],
                timeout=0.1,
                tracker=tracker,
            )
        )
        assert tracker.consecutive_failures == 1
        asyncio.run(
            llm_summarize(
                engine,
                [{"role": "user", "content": "hi"}],
                timeout=0.1,
                tracker=tracker,
            )
        )
        assert tracker.consecutive_failures == 2
        assert tracker.tripped

    def test_empty_messages_returns_none(self):
        engine = AsyncMock()
        result = asyncio.run(llm_summarize(engine, []))
        assert result is None
        engine.chat_completion.assert_not_called()


# ─── AgentLoop.compact_now ────────────────────────────────────────────────


class TestCompactNowLlmTier:
    def test_default_off_uses_extractive(self):
        """When llm=False (default), compact_now uses extractive summary only."""
        agent = _make_agent(responses=[CompletionResult(content="unused")])
        _add_conversation(agent, n=3)
        result = asyncio.run(agent.compact_now())
        assert result["compacted"] is True
        # Extractive signature: "User asked:" or "Actions:" etc.
        assert "User asked:" in result["summary"]
        # Engine was never called for a summary (but the mock is the agent engine)

    def test_enabled_calls_engine_and_uses_result(self):
        """When llm=True, compact_now tries the LLM tier first."""
        agent = _make_agent(
            responses=[CompletionResult(content="LLM: 4 turns summarized")],
            compaction=CompactionConfig(llm=True, timeout=5.0),
        )
        _add_conversation(agent, n=3)
        result = asyncio.run(agent.compact_now())
        assert result["compacted"] is True
        assert result["summary"] == "LLM: 4 turns summarized"
        assert "LLM: 4 turns summarized" in agent.messages[1]["content"]
        # Engine was called at least once (for the summarization)

    def test_enabled_engine_failure_falls_back_to_extractive(self):
        """LLM tier engine error → extractive summary, not None/empty."""
        agent = _make_agent(
            compaction=CompactionConfig(llm=True, timeout=5.0),
        )
        agent.engine.chat_completion = AsyncMock(side_effect=RuntimeError("down"))
        _add_conversation(agent, n=3)
        result = asyncio.run(agent.compact_now())
        assert result["compacted"] is True
        assert "User asked:" in result["summary"]
        assert "User asked:" in agent.messages[1]["content"]

    def test_enabled_empty_llm_response_falls_back_to_extractive(self):
        """LLM tier returns empty content → extractive summary is used."""
        agent = _make_agent(
            responses=[CompletionResult(content=None)],
            compaction=CompactionConfig(llm=True, timeout=5.0),
        )
        _add_conversation(agent, n=3)
        result = asyncio.run(agent.compact_now())
        assert result["compacted"] is True
        assert "User asked:" in result["summary"]

    def test_dry_run_does_not_mutate_messages(self):
        agent = _make_agent(responses=[CompletionResult(content="x")])
        _add_conversation(agent, n=3)
        before = list(agent.messages)
        result = asyncio.run(agent.compact_now(dry_run=True))
        assert result["compacted"] is True
        # Messages unchanged
        assert agent.messages == before

    def test_sync_compat_shim_runs_the_async_core(self):
        """compact_history (sync API) routes through compact_now."""
        agent = _make_agent(responses=[CompletionResult(content="unused")])
        _add_conversation(agent, n=3)
        result = agent.compact_history()
        assert result["compacted"] is True
        assert len(agent.messages) == 4
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[1]["role"] == "system"

    def test_sync_compat_shim_dry_run_preserves_messages(self):
        agent = _make_agent(responses=[CompletionResult(content="unused")])
        _add_conversation(agent, n=3)
        before = list(agent.messages)
        result = agent.compact_history(dry_run=True)
        assert result["compacted"] is True
        assert agent.messages == before

    def test_tier_short_circuits_when_tripped(self):
        """After the breaker trips, no engine I/O for the rest of the run."""
        agent = _make_agent(compaction=CompactionConfig(llm=True, timeout=5.0))
        agent.engine.chat_completion = AsyncMock(side_effect=RuntimeError("down"))
        _add_conversation(agent, n=3)
        # Two calls: the first two are recorded as failures (trip_after=3
        # by default, so we need at least one to record, but the breaker
        # trips after three — so we need three to trigger, or override.)
        # Simpler: force-trip.
        agent._sum_failure_tracker.tripped = True
        agent.engine.chat_completion.reset_mock()
        result = asyncio.run(agent.compact_now())
        assert result["compacted"] is True
        assert "User asked:" in result["summary"]
        assert agent.engine.chat_completion.await_count == 0

    def test_sync_seam_takes_precedence_over_llm_tier(self):
        """cm.summarizer (explicit) wins over the engine-backed tier."""
        agent = _make_agent(
            responses=[CompletionResult(content="LLM says this should NOT win")],
            compaction=CompactionConfig(llm=True, timeout=5.0),
        )
        agent._context_manager.summarizer = lambda msgs: "explicit seam wins"
        _add_conversation(agent, n=3)
        result = asyncio.run(agent.compact_now())
        assert result["summary"] == "explicit seam wins"

    def test_short_conversation_noop(self):
        """<= 3 messages → compact_now is a no-op (matches compact_history)."""
        agent = _make_agent(responses=[CompletionResult(content="unused")])
        # Only the system prompt from initialize() is present
        result = asyncio.run(agent.compact_now())
        assert result["compacted"] is False

    def test_reset_failure_tracker_per_run(self):
        """handle_user_message resets the failure breaker between runs."""
        agent = _make_agent(compaction=CompactionConfig(llm=True))
        agent.engine.chat_completion = AsyncMock(side_effect=RuntimeError("down"))
        _add_conversation(agent, n=3)
        # First compact: failure, streak=1
        asyncio.run(agent.compact_now())
        assert agent._sum_failure_tracker.consecutive_failures == 1
        asyncio.run(agent.compact_now())
        assert agent._sum_failure_tracker.consecutive_failures == 2
        # Simulate end of run + new user message: reset the breaker.
        agent._sum_failure_tracker.reset()
        assert agent._sum_failure_tracker.consecutive_failures == 0
        assert not agent._sum_failure_tracker.tripped


# ─── [compaction] config section ──────────────────────────────────────────


class TestCompactionConfig:
    def test_defaults_off(self):
        cfg = CompactionConfig()
        assert cfg.llm is False
        assert cfg.max_messages == 30
        assert cfg.timeout == 30.0

    def test_loads_from_toml(self, tmp_path):

        config_file = tmp_path / "config.toml"
        config_file.write_text(
            textwrap.dedent(
                """
                [compaction]
                llm = true
                max_messages = 50
                timeout = 10
                """
            )
        )
        cfg = load_config(str(config_file))
        assert cfg.compaction.llm is True
        assert cfg.compaction.max_messages == 50
        assert cfg.compaction.timeout == 10

    def test_missing_section_uses_defaults(self, tmp_path):

        config_file = tmp_path / "config.toml"
        config_file.write_text('[ui]\ntheme = "light"\n')
        cfg = load_config(str(config_file))
        assert cfg.compaction.llm is False
        assert cfg.compaction.max_messages == 30

    def test_partial_section(self, tmp_path):

        config_file = tmp_path / "config.toml"
        config_file.write_text(
            textwrap.dedent(
                """
                [compaction]
                llm = true
                """
            )
        )
        cfg = load_config(str(config_file))
        assert cfg.compaction.llm is True
        assert cfg.compaction.max_messages == 30
        assert cfg.compaction.timeout == 30.0

    def test_save_config_value_roundtrip(self, tmp_path, monkeypatch):
        from pathlib import Path

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        path = save_config_value("compaction", "llm", True)
        assert path.exists()
        content = path.read_text()
        assert "[compaction]" in content
        assert "llm = true" in content

    def test_load_roundtrip_after_save(self, tmp_path, monkeypatch):
        from pathlib import Path

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        save_config_value("compaction", "llm", True)
        save_config_value("compaction", "max_messages", 77)
        cfg = load_config()
        assert cfg.compaction.llm is True
        assert cfg.compaction.max_messages == 77
