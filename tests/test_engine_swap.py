"""Tests for engine swap and runtime fallback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from natshell.agent.context import SystemContext
from natshell.agent.loop import AgentLoop, EventType
from natshell.config import AgentConfig, ModelConfig, SafetyConfig
from natshell.inference.engine import EngineInfo
from natshell.inference.remote import RemoteEngine
from natshell.safety.classifier import SafetyClassifier
from natshell.tools.registry import create_default_registry


def _make_context() -> SystemContext:
    return SystemContext(
        hostname="testhost",
        distro="Debian 13",
        kernel="6.12.0",
        username="testuser",
    )


def _make_agent(
    engine=None,
    fallback_config: ModelConfig | None = None,
) -> AgentLoop:
    if engine is None:
        engine = AsyncMock()
        engine.engine_info = MagicMock(return_value=EngineInfo(engine_type="mock"))
    tools = create_default_registry()
    safety = SafetyClassifier(
        SafetyConfig(
            mode="confirm",
            always_confirm=[r"^rm\s"],
            blocked=[r"^rm\s+-[rR]f\s+/\s*$"],
        )
    )
    agent = AgentLoop(
        engine=engine,
        tools=tools,
        safety=safety,
        config=AgentConfig(max_steps=15, temperature=0.3, max_tokens=2048),
        fallback_config=fallback_config,
    )
    agent.initialize(_make_context())
    return agent


async def _collect_events(agent: AgentLoop, message: str):
    events = []
    async for event in agent.handle_user_message(message):
        events.append(event)
    return events


# ─── swap_engine ────────────────────────────────────────────────────────────


class TestSwapEngine:
    async def test_swap_clears_history(self):
        agent = _make_agent()
        # Add some messages
        agent.messages.append({"role": "user", "content": "hello"})
        agent.messages.append({"role": "assistant", "content": "hi"})
        assert len(agent.messages) == 3  # system + user + assistant

        new_engine = AsyncMock()
        new_engine.engine_info = MagicMock(return_value=EngineInfo(engine_type="new"))
        await agent.swap_engine(new_engine)

        # Should be re-initialized with just the system prompt
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"
        assert agent.engine is new_engine

    async def test_swap_reinitializes_system_prompt(self):
        agent = _make_agent()
        system_content = agent.messages[0]["content"]

        new_engine = AsyncMock()
        new_engine.engine_info = MagicMock(return_value=EngineInfo(engine_type="new"))
        await agent.swap_engine(new_engine)

        # System prompt should be regenerated (same content since same context)
        assert agent.messages[0]["content"] == system_content

    async def test_swap_closes_old_engine(self):
        old_engine = AsyncMock()
        old_engine.close = AsyncMock()
        old_engine.engine_info = MagicMock(return_value=EngineInfo(engine_type="old"))
        agent = _make_agent(engine=old_engine)

        new_engine = AsyncMock()
        new_engine.engine_info = MagicMock(return_value=EngineInfo(engine_type="new"))
        await agent.swap_engine(new_engine)

        old_engine.close.assert_awaited_once()


# ─── Runtime fallback ───────────────────────────────────────────────────────


class TestRuntimeFallback:
    async def test_fallback_on_connect_error(self):
        """Remote ConnectError triggers local fallback."""
        remote_engine = AsyncMock(spec=RemoteEngine)
        remote_engine.base_url = "http://localhost:11434/v1"
        remote_engine.model = "qwen3:4b"
        remote_engine.engine_info = MagicMock(
            return_value=EngineInfo(
                engine_type="remote",
                model_name="qwen3:4b",
            )
        )
        remote_engine.chat_completion = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )

        fallback = ModelConfig(
            path="/tmp/test-model.gguf",
            n_ctx=0,
            n_threads=0,
            n_gpu_layers=0,
        )
        agent = _make_agent(engine=remote_engine, fallback_config=fallback)

        with (
            patch.object(Path, "exists", return_value=True),
            patch("natshell.agent.loop.asyncio.to_thread") as mock_thread,
        ):
            mock_local = AsyncMock()
            mock_local.engine_info = MagicMock(return_value=EngineInfo(engine_type="local"))
            mock_thread.return_value = mock_local

            events = await _collect_events(agent, "hello")

        types = [e.type for e in events]
        assert EventType.ERROR in types
        error = next(e for e in events if e.type == EventType.ERROR)
        assert "local model" in error.data.lower()

    async def test_no_fallback_when_already_local(self):
        """Local engine errors don't trigger fallback."""
        local_engine = AsyncMock()
        local_engine.engine_info = MagicMock(return_value=EngineInfo(engine_type="local"))
        local_engine.chat_completion = AsyncMock(side_effect=httpx.ConnectError("some error"))

        agent = _make_agent(engine=local_engine, fallback_config=ModelConfig())
        events = await _collect_events(agent, "hello")

        types = [e.type for e in events]
        assert EventType.ERROR in types
        error = next(e for e in events if e.type == EventType.ERROR)
        # Should be a normal error, not a fallback message
        assert "inference error" in error.data.lower()

    async def test_fallback_fails_when_model_missing(self):
        """Fallback fails gracefully if local model not downloaded."""
        remote_engine = AsyncMock(spec=RemoteEngine)
        remote_engine.base_url = "http://localhost:11434/v1"
        remote_engine.model = "qwen3:4b"
        remote_engine.engine_info = MagicMock(
            return_value=EngineInfo(
                engine_type="remote",
                model_name="qwen3:4b",
            )
        )
        remote_engine.chat_completion = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )

        fallback = ModelConfig(path="/nonexistent/model.gguf")
        agent = _make_agent(engine=remote_engine, fallback_config=fallback)

        events = await _collect_events(agent, "hello")
        types = [e.type for e in events]
        assert EventType.ERROR in types
        error = next(e for e in events if e.type == EventType.ERROR)
        # Should surface the original inference error since fallback failed
        assert "inference error" in error.data.lower()

    async def test_no_fallback_without_config(self):
        """No fallback when fallback_config is None."""
        remote_engine = AsyncMock(spec=RemoteEngine)
        remote_engine.base_url = "http://localhost:11434/v1"
        remote_engine.engine_info = MagicMock(return_value=EngineInfo(engine_type="remote"))
        remote_engine.chat_completion = AsyncMock(side_effect=httpx.ConnectError("refused"))

        agent = _make_agent(engine=remote_engine, fallback_config=None)
        events = await _collect_events(agent, "hello")

        types = [e.type for e in events]
        assert EventType.ERROR in types
        error = next(e for e in events if e.type == EventType.ERROR)
        assert "inference error" in error.data.lower()


# ─── Compaction before fallback ────────────────────────────────────────────


def _make_remote_agent_with_history(
    side_effect,
    fallback_config=None,
    n_ctx=262144,
):
    """Create an agent with a remote engine and pre-populated conversation history."""
    remote_engine = AsyncMock(spec=RemoteEngine)
    remote_engine.base_url = "http://localhost:11434/v1"
    remote_engine.api_key = ""
    remote_engine.model = "gemma4:31b"
    remote_engine.engine_info = MagicMock(
        return_value=EngineInfo(
            engine_type="remote",
            model_name="gemma4:31b",
            n_ctx=n_ctx,
        )
    )
    remote_engine.chat_completion = AsyncMock(side_effect=side_effect)

    agent = _make_agent(engine=remote_engine, fallback_config=fallback_config)
    # Populate with >3 messages so compaction is possible
    agent.messages.append({"role": "user", "content": "list cron jobs"})
    agent.messages.append({"role": "assistant", "content": "Here are the cron jobs..."})
    agent.messages.append({"role": "user", "content": "check the backup log"})
    agent.messages.append({"role": "assistant", "content": "The backup log shows failures..."})
    agent.messages.append({"role": "user", "content": "fix the log path"})
    return agent


class TestCompactionBeforeFallback:
    async def test_timeout_compacts_when_server_alive(self):
        """ReadTimeout + server alive → compact and retry, no fallback."""
        from natshell.inference.engine import CompletionResult

        # First call: timeout. Second call (after compaction): success.
        side_effects = [
            httpx.ReadTimeout("read timed out"),
            CompletionResult(
                content="Done.",
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=100,
                completion_tokens=20,
            ),
        ]
        agent = _make_remote_agent_with_history(
            side_effect=side_effects,
            fallback_config=ModelConfig(
                path="/tmp/test-model.gguf", n_ctx=0, n_threads=0, n_gpu_layers=0,
            ),
        )

        with patch("natshell.inference.ollama.ping_server", new_callable=AsyncMock, return_value=True):
            events = await _collect_events(agent, "fix the log path")

        errors = [e for e in events if e.type == EventType.ERROR]
        assert any("compacted" in e.data.lower() and "retrying" in e.data.lower() for e in errors)
        assert not any("local model" in e.data.lower() for e in errors)
        assert agent._context_recovery_attempted is True

    async def test_timeout_falls_back_when_server_dead(self):
        """ReadTimeout + server dead → fall back to local model."""
        agent = _make_remote_agent_with_history(
            side_effect=httpx.ReadTimeout("read timed out"),
            fallback_config=ModelConfig(
                path="/tmp/test-model.gguf",
                n_ctx=0,
                n_threads=0,
                n_gpu_layers=0,
            ),
        )

        with (
            patch("natshell.inference.ollama.ping_server", new_callable=AsyncMock, return_value=False),
            patch.object(Path, "exists", return_value=True),
            patch("natshell.agent.loop.asyncio.to_thread") as mock_thread,
        ):
            mock_local = AsyncMock()
            mock_local.engine_info = MagicMock(
                return_value=EngineInfo(engine_type="local", n_ctx=32768)
            )
            mock_thread.return_value = mock_local

            events = await _collect_events(agent, "fix the log path")

        errors = [e for e in events if e.type == EventType.ERROR]
        assert any("local model" in e.data.lower() for e in errors)

    async def test_timeout_skips_ping_when_no_history(self):
        """ReadTimeout + short conversation → skip ping, fall back directly."""
        remote_engine = AsyncMock(spec=RemoteEngine)
        remote_engine.base_url = "http://localhost:11434/v1"
        remote_engine.api_key = ""
        remote_engine.model = "gemma4:31b"
        remote_engine.engine_info = MagicMock(
            return_value=EngineInfo(engine_type="remote", model_name="gemma4:31b", n_ctx=262144)
        )
        remote_engine.chat_completion = AsyncMock(
            side_effect=httpx.ReadTimeout("read timed out")
        )

        fallback = ModelConfig(
            path="/tmp/test-model.gguf", n_ctx=0, n_threads=0, n_gpu_layers=0,
        )
        agent = _make_agent(engine=remote_engine, fallback_config=fallback)
        # Only system + user message (<=3 messages) — no history to compact

        with (
            patch("natshell.inference.ollama.ping_server", new_callable=AsyncMock) as mock_ping,
            patch.object(Path, "exists", return_value=True),
            patch("natshell.agent.loop.asyncio.to_thread") as mock_thread,
        ):
            mock_local = AsyncMock()
            mock_local.engine_info = MagicMock(
                return_value=EngineInfo(engine_type="local", n_ctx=32768)
            )
            mock_thread.return_value = mock_local

            events = await _collect_events(agent, "hello")

        # No ping because conversation is too short for compaction-retry
        mock_ping.assert_not_called()
        errors = [e for e in events if e.type == EventType.ERROR]
        assert any("local model" in e.data.lower() for e in errors)

    async def test_timeout_falls_back_after_prior_compaction(self):
        """Two sequential timeouts: first compacts+retries, second falls back."""
        # Both calls timeout — first triggers compaction+retry, second hits
        # the _context_recovery_attempted guard and falls back.
        agent = _make_remote_agent_with_history(
            side_effect=httpx.ReadTimeout("read timed out"),
            fallback_config=ModelConfig(
                path="/tmp/test-model.gguf", n_ctx=0, n_threads=0, n_gpu_layers=0,
            ),
        )

        with (
            patch("natshell.inference.ollama.ping_server", new_callable=AsyncMock, return_value=True) as mock_ping,
            patch.object(Path, "exists", return_value=True),
            patch("natshell.agent.loop.asyncio.to_thread") as mock_thread,
        ):
            mock_local = AsyncMock()
            mock_local.engine_info = MagicMock(
                return_value=EngineInfo(engine_type="local", n_ctx=32768)
            )
            mock_thread.return_value = mock_local

            events = await _collect_events(agent, "fix the log path")

        # Ping called once for first timeout, skipped for second
        mock_ping.assert_called_once()
        errors = [e for e in events if e.type == EventType.ERROR]
        # First timeout: compaction message
        assert any("compacted" in e.data.lower() and "retrying" in e.data.lower() for e in errors)
        # Second timeout: fallback
        assert any("local model" in e.data.lower() for e in errors)

    async def test_fallback_preserves_compacted_context(self):
        """Fallback to local model preserves compacted context when it fits."""
        agent = _make_remote_agent_with_history(
            side_effect=httpx.ReadTimeout("read timed out"),
            fallback_config=ModelConfig(
                path="/tmp/test-model.gguf", n_ctx=0, n_threads=0, n_gpu_layers=0,
            ),
        )

        with (
            patch("natshell.inference.ollama.ping_server", new_callable=AsyncMock, return_value=False),
            patch.object(Path, "exists", return_value=True),
            patch("natshell.agent.loop.asyncio.to_thread") as mock_thread,
        ):
            mock_local = AsyncMock()
            mock_local.engine_info = MagicMock(
                return_value=EngineInfo(engine_type="local", n_ctx=32768)
            )
            mock_thread.return_value = mock_local

            events = await _collect_events(agent, "fix the log path")

        errors = [e for e in events if e.type == EventType.ERROR]
        assert any("context preserved" in e.data.lower() for e in errors)
        # More than just system prompt — preserved messages were injected
        assert len(agent.messages) > 1

    async def test_fallback_clears_history_when_context_too_large(self):
        """Fallback clears history when preserved context doesn't fit."""
        agent = _make_remote_agent_with_history(
            side_effect=httpx.ReadTimeout("read timed out"),
            fallback_config=ModelConfig(
                path="/tmp/test-model.gguf", n_ctx=0, n_threads=0, n_gpu_layers=0,
            ),
        )

        with (
            patch("natshell.inference.ollama.ping_server", new_callable=AsyncMock, return_value=False),
            patch.object(Path, "exists", return_value=True),
            patch("natshell.agent.loop.asyncio.to_thread") as mock_thread,
        ):
            mock_local = AsyncMock()
            # Tiny context — preserved messages won't fit
            mock_local.engine_info = MagicMock(
                return_value=EngineInfo(engine_type="local", n_ctx=512)
            )
            mock_thread.return_value = mock_local

            events = await _collect_events(agent, "fix the log path")

        errors = [e for e in events if e.type == EventType.ERROR]
        assert any("history cleared" in e.data.lower() for e in errors)
        assert not any("context preserved" in e.data.lower() for e in errors)
