"""Unit tests for the ordered recovery ladder (RecoveryCoordinator).

The coordinator owns the context-overflow / connectivity-failure
recovery state machine extracted from ``AgentLoop.handle_user_message``
(R1-6).  Integration tests for the same behaviour live in
``tests/test_agent.py`` (TestContextOverflow) and
``tests/test_engine_swap.py`` (TestRuntimeFallback,
TestCompactionBeforeFallback); these tests exercise the state machine
in isolation so its outcome/flag semantics have a home of their own.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from natshell.agent.recovery import (
    RecoveryCoordinator,
    RecoveryOutcome,
    RecoveryStrategy,
)
from natshell.inference.remote import AuthenticationError, ContextOverflowError


def _make_coordinator(
    *,
    engine: Any,
    fallback_config: Any = None,
    compact_stats: dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
    context_manager: Any = None,
    max_tokens_fn: Any = None,
    reserve: int = 800,
    local_engine: Any = None,
):
    """Build a coordinator with recorded call counters.

    ``engine_box`` stands in for ``AgentLoop.engine``: the stub swap
    replaces what ``engine_ref`` sees, mirroring
    ``AgentLoop.swap_engine`` mutating ``self.engine``.
    ``local_engine`` is what the stub ``load_fallback`` returns
    (``None`` simulates a failed load).
    """
    calls: dict[str, Any] = {
        "compact": 0,
        "fallback_load": 0,
        "swaps": 0,
    }
    engine_box: dict[str, Any] = {"engine": engine}
    compact_stats = compact_stats or {"compacted": False}

    def compact() -> dict[str, Any]:
        calls["compact"] += 1
        return compact_stats

    async def load_fallback(config: Any) -> Any:
        calls["fallback_load"] += 1
        return local_engine

    async def swap(new_engine: Any) -> None:
        calls["swaps"] += 1
        engine_box["engine"] = new_engine

    coordinator = RecoveryCoordinator(
        engine_ref=lambda: engine_box["engine"],
        fallback_config=fallback_config,
        compact=compact,
        effective_max_tokens=max_tokens_fn or (lambda n_ctx: 1024),
        context_reserve=reserve,
        messages_ref=lambda: messages or [],
        load_fallback=load_fallback,
        swap_engine=swap,
        context_manager_ref=lambda: context_manager,
    )
    return coordinator, calls


class TestLadderOrdering:
    """The ladder's outcomes and ordering, in isolation."""

    async def test_overflow_compacts_and_retries(self):
        coordinator, calls = _make_coordinator(
            engine=MagicMock(name="engine"),
            compact_stats={"compacted": True},
            messages=[{"role": "system"}] * 6,
        )
        outcome, events = await coordinator.handle(
            ContextOverflowError("context length exceeded")
        )
        assert outcome is RecoveryOutcome.RETRY
        assert calls["compact"] == 1
        assert coordinator.attempted is True
        assert any("compacted" in e.data.lower() for e in events)
        assert any("Retrying" in e.data for e in events)

    async def test_overflow_second_time_gives_up_with_clear_hint(self):
        coordinator, _ = _make_coordinator(
            engine=MagicMock(name="engine"),
            compact_stats={"compacted": True},
        )
        coordinator.attempted = True  # compaction already tried this run
        outcome, events = await coordinator.handle(
            ContextOverflowError("context length exceeded")
        )
        assert outcome is RecoveryOutcome.STOP
        assert any("/clear" in e.data for e in events)

    async def test_overflow_too_short_to_compact(self):
        coordinator, calls = _make_coordinator(
            engine=MagicMock(name="engine"),
            compact_stats={"compacted": False},
            messages=[{"role": "system"}, {"role": "user", "content": "hi"}],
        )
        outcome, events = await coordinator.handle(
            ContextOverflowError("context length exceeded")
        )
        assert calls["compact"] == 1
        assert outcome is RecoveryOutcome.STOP
        assert any("/clear" in e.data for e in events)

    async def test_overflow_never_considers_local_fallback(self):
        """Overflow is a context problem, not a connectivity one."""
        coordinator, calls = _make_coordinator(
            engine=MagicMock(spec=object, **{}),
            fallback_config=object(),
            compact_stats={"compacted": False},
        )
        outcome, _ = await coordinator.handle(ContextOverflowError("full"))
        assert outcome is RecoveryOutcome.STOP
        assert calls["fallback_load"] == 0

    async def test_connection_error_no_config_surfaces_raw_error(self):
        coordinator, _ = _make_coordinator(
            engine=MagicMock(name="not_remote"),
            fallback_config=None,
        )
        outcome, events = await coordinator.handle(
            httpx.ConnectError("connection refused")
        )
        assert outcome is RecoveryOutcome.STOP
        assert "Inference error: connection refused" in events[0].data

    async def test_generic_exception_surfaces_raw_error(self):
        coordinator, calls = _make_coordinator(engine=MagicMock())
        outcome, events = await coordinator.handle(ValueError("boom"))
        assert outcome is RecoveryOutcome.STOP
        assert "Inference error: boom" in events[0].data
        assert calls["compact"] == 0
        assert calls["fallback_load"] == 0

    async def test_reset_clears_attempt_latch(self):
        coordinator, _ = _make_coordinator(engine=MagicMock())
        coordinator.attempted = True
        coordinator.reset()
        assert coordinator.attempted is False


class TestConnectivityLadder:
    """Phase 1 (ping/compact/retry) vs Phase 2 (local fallback)."""

    _PING = "natshell.inference.ollama.ping_server"

    def _remote_engine(self) -> Any:
        """A stand-in remote engine that passes can_fallback's checks."""
        from natshell.inference.remote import RemoteEngine

        engine = AsyncMock(spec=RemoteEngine)
        engine.base_url = "http://localhost:11434"
        return engine

    async def test_alive_server_compacts_and_retries(self):
        messages = [
            {"role": "system"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        coordinator, calls = _make_coordinator(
            engine=self._remote_engine(),
            fallback_config=object(),
            compact_stats={"compacted": True},
            messages=messages,
        )
        with patch(self._PING, new_callable=AsyncMock, return_value=True):
            outcome, events = await coordinator.handle(
                httpx.ReadTimeout("read timed out")
            )
        assert outcome is RecoveryOutcome.RETRY
        assert calls["compact"] == 1
        assert calls["fallback_load"] == 0
        assert coordinator.attempted is True
        assert any("timed out" in e.data.lower() for e in events)
        assert any("retrying" in e.data.lower() for e in events)

    async def test_dead_server_falls_back_to_local(self):
        messages = [
            {"role": "system"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        new_engine = AsyncMock()
        new_engine.engine_info = MagicMock(
            return_value=MagicMock(n_ctx=4096)
        )
        coordinator, calls = _make_coordinator(
            engine=self._remote_engine(),
            fallback_config=object(),
            compact_stats={"compacted": True},
            messages=messages,
            local_engine=new_engine,
        )
        with patch(self._PING, new_callable=AsyncMock, return_value=False):
            outcome, events = await coordinator.handle(
                httpx.ReadTimeout("read timed out")
            )

        assert outcome is RecoveryOutcome.STOP
        assert calls["fallback_load"] == 1
        assert calls["swaps"] == 1
        # No context manager → nothing can be verified to fit → cleared
        assert any("Switched to local model" in e.data for e in events)
        assert any("History cleared" in e.data for e in events)
        # The fallback engine is what engine_ref now reports
        assert coordinator._engine() is new_engine

    async def test_fallback_load_failure_surfaces_raw_error(self):
        coordinator, calls = _make_coordinator(
            engine=self._remote_engine(),
            fallback_config=object(),
            local_engine=None,  # stub load_fallback returns None → failure
            messages=[{"role": "system"}, {"role": "user", "content": "a"}],
        )
        with patch(self._PING, new_callable=AsyncMock, return_value=False):
            outcome, events = await coordinator.handle(
                httpx.ConnectError("refused")
            )
        assert outcome is RecoveryOutcome.STOP
        assert "Inference error" in events[0].data
        assert calls["fallback_load"] == 1
        assert calls["swaps"] == 0

    async def test_phase1_skipped_when_no_history(self):
        """≤3 messages → no ping, straight to the fallback phase."""
        local = AsyncMock()
        local.engine_info = MagicMock(return_value=MagicMock(n_ctx=4096))
        coordinator, calls = _make_coordinator(
            engine=self._remote_engine(),
            fallback_config=object(),
            messages=[{"role": "system"}, {"role": "user", "content": "a"}],
            local_engine=local,
        )
        with patch(
            self._PING, new_callable=AsyncMock,
            return_value=True
        ) as ping:
            outcome, _ = await coordinator.handle(
                httpx.ReadTimeout("read timed out")
            )
        ping.assert_not_called()
        assert outcome is RecoveryOutcome.STOP
        assert calls["fallback_load"] == 1
        assert calls["swaps"] == 1

    async def test_phase1_skipped_when_latch_set(self):
        """A prior compaction this run escalates straight to fallback."""
        messages = [
            {"role": "system"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        local = AsyncMock()
        local.engine_info = MagicMock(return_value=MagicMock(n_ctx=4096))
        coordinator, calls = _make_coordinator(
            engine=self._remote_engine(),
            fallback_config=object(),
            compact_stats={"compacted": True},
            messages=messages,
            local_engine=local,
        )
        coordinator.attempted = True
        with patch(
            self._PING, new_callable=AsyncMock,
            return_value=True
        ) as ping:
            outcome, _ = await coordinator.handle(
                httpx.ReadTimeout("read timed out")
            )
        ping.assert_not_called()
        assert outcome is RecoveryOutcome.STOP
        assert calls["fallback_load"] == 1


class TestContextPreservation:
    """Fallback context re-injection respects the local budget."""

    _PING = "natshell.inference.ollama.ping_server"

    def _remote_engine(self) -> Any:
        from natshell.inference.remote import RemoteEngine

        engine = AsyncMock(spec=RemoteEngine)
        engine.base_url = "http://localhost:11434"
        return engine

    def _context_manager(self, tokens_per_msg: int = 10) -> Any:
        cm = MagicMock()

        def estimate(messages: list) -> int:
            return len(messages) * tokens_per_msg

        cm.estimate_tokens = estimate
        return cm

    async def test_preserved_context_fits_budget(self):
        messages = [
            {"role": "system"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        new_engine = AsyncMock()
        new_engine.engine_info = MagicMock(
            return_value=MagicMock(n_ctx=65536)
        )
        cm = self._context_manager()
        coordinator, calls = _make_coordinator(
            engine=self._remote_engine(),
            fallback_config=object(),
            compact_stats={"compacted": True},
            messages=messages,
            context_manager=cm,
            max_tokens_fn=lambda n_ctx: 2048,
            local_engine=new_engine,
        )
        with patch(self._PING, new_callable=AsyncMock, return_value=False):
            outcome, events = await coordinator.handle(
                httpx.ReadTimeout("read timed out")
            )
        assert outcome is RecoveryOutcome.STOP
        # 3 + 3 tokens « 65536 - 2048 - 800 — everything fits
        assert any("context preserved" in e.data.lower() for e in events)
        # Preserved messages were re-extended onto the history
        assert len(messages) > 3

    async def test_preserved_context_too_large(self):
        messages = [
            {"role": "system"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        new_engine = AsyncMock()
        new_engine.engine_info = MagicMock(return_value=MagicMock(n_ctx=512))
        cm = self._context_manager(tokens_per_msg=1000)
        coordinator, _ = _make_coordinator(
            engine=self._remote_engine(),
            fallback_config=object(),
            compact_stats={"compacted": True},
            messages=messages,
            context_manager=cm,
            max_tokens_fn=lambda n_ctx: 2048,
            local_engine=new_engine,
        )
        with patch(self._PING, new_callable=AsyncMock, return_value=False):
            outcome, events = await coordinator.handle(
                httpx.ReadTimeout("read timed out")
            )
        assert outcome is RecoveryOutcome.STOP
        # 3000 + 3000 tokens ≫ 512 - 2048 - 800 (negative budget) → dropped
        assert any("history cleared" in e.data.lower() for e in events)
        assert not any("context preserved" in e.data.lower() for e in events)


class TestMisc:
    def test_recovery_strategy_alias(self):
        assert RecoveryStrategy is RecoveryCoordinator

    async def test_can_fallback_delegates_to_fallback_module(self):
        from natshell.inference.remote import RemoteEngine

        # Local engine → never a fallback candidate
        coordinator, _ = _make_coordinator(engine=MagicMock())
        assert not coordinator._can_fallback(ConnectionError("x"))

        # Remote engine + config → yes; auth errors are explicitly not
        coordinator, _ = _make_coordinator(
            engine=AsyncMock(spec=RemoteEngine), fallback_config=object()
        )
        assert coordinator._can_fallback(httpx.ConnectError("x"))
        assert not coordinator._can_fallback(AuthenticationError("bad key"))
        assert not coordinator._can_fallback(ContextOverflowError("full"))
