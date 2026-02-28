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
