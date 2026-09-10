"""Tests for natshell.agent.tool_dispatch.

Unit tests for the single-tool-call dispatcher extracted from
``AgentLoop.handle_user_message`` (R1-7 final chunk): the step-budget hint
and the per-call lifecycle (normalize → classify → confirm → execute →
observe).  The loop-level integration (event interleaving, sudo-retry
ladder, guard stop breaking the batch) is pinned by ``tests/test_agent.py``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from natshell.agent.events import EventType
from natshell.agent.repetition_guard import RepetitionGuard
from natshell.agent.tool_dispatch import (
    PARALLEL_SAFE_TOOLS,
    DispatchOutcome,
    dispatch_tool_batch,
    dispatch_tool_call,
    step_budget_hint,
)
from natshell.config import SafetyConfig
from natshell.inference.engine import ToolCall
from natshell.safety.classifier import SafetyClassifier
from natshell.tools.registry import ToolResult


class FakeRegistry:
    """Minimal ToolRegistry stand-in for dispatch tests."""

    def __init__(self, result: ToolResult | None = None) -> None:
        self.result = result or ToolResult(output="out", exit_code=0)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def normalize_arguments(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        # mirror registry behaviour: "cmd" → "command" for execute_shell
        if name == "execute_shell" and "cmd" in args:
            out = dict(args)
            out["command"] = out.pop("cmd")
            return out
        return None

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append((name, arguments))
        return self.result


def _safety(mode: str = "confirm") -> SafetyClassifier:
    return SafetyClassifier(
        SafetyConfig(
            mode=mode,
            blocked=[r"^rm\s+-[rR]f\s+/"],
            always_confirm=[r"^echo\s+confirm-me"],
        )
    )


async def _dispatch(
    call: ToolCall,
    *,
    confirm=None,
    password=None,
    registry: FakeRegistry | None = None,
    safety: SafetyClassifier | None = None,
    steps_used: int = 2,
    max_steps: int = 10,
) -> tuple[DispatchOutcome, list[tuple[ToolCall, str]]]:
    exch: list[tuple[ToolCall, str]] = []
    registry = registry or FakeRegistry()
    safety = safety or _safety()
    outcome = await dispatch_tool_call(
        call,
        tools=registry,  # type: ignore[arg-type]
        safety=safety,
        guard=RepetitionGuard(),
        confirm_callback=confirm,
        password_callback=password,
        steps_used=steps_used,
        max_steps=max_steps,
        append_exchange=lambda tc, content: exch.append((tc, content)),
    )
    return outcome, exch


# ─────────────────────────────────────────────────────────────────────────────
# Step-budget hint (pure)
# ─────────────────────────────────────────────────────────────────────────────


class TestStepBudgetHint:
    def test_first_step_welcome(self):
        hint = step_budget_hint(1, 15)
        assert "Budget: 15 steps available" in hint
        assert "plan your approach" in hint

    def test_quarter_quarter_quarter_is_silent(self):
        # 2/10 = 20% — below the 50% tier, not step 1
        assert step_budget_hint(2, 10) == ""

    def test_halfway_marker(self):
        hint = step_budget_hint(5, 10)
        assert hint == "\n\n[5/10 steps used]"

    def test_75_percent_warns(self):
        hint = step_budget_hint(8, 10)
        assert "⚠" in hint
        assert "8/10 steps used" in hint
        assert "wrap up soon" in hint

    def test_90_percent_urgent(self):
        hint = step_budget_hint(9, 10)
        assert "URGENT" in hint
        assert "only 1 steps left" in hint
        assert "Finish NOW" in hint

    def test_exact_50_percent_tier(self):
        assert "8/10" not in step_budget_hint(5, 10)
        assert "5/10 steps used" in step_budget_hint(5, 10)


# ─────────────────────────────────────────────────────────────────────────────
# Happy path: SAFE tool executes, budget hint appended to the exchange
# ─────────────────────────────────────────────────────────────────────────────


class TestSafeExecution:
    async def test_executes_and_appends_exchange(self):
        call = ToolCall(id="1", name="list_directory", arguments={})
        registry = FakeRegistry()
        outcome, exch = await _dispatch(call, registry=registry)

        assert [e.type for e in outcome.events] == [
            EventType.THINKING,
            EventType.EXECUTING,
            EventType.TOOL_RESULT,
        ]
        assert registry.calls == [("list_directory", {})]
        assert len(exch) == 1
        content = exch[0][1]
        assert content.startswith("out")
        # 2/10 = 20% → silent tier, but the hint still exists (empty)
        assert "steps used" not in content

    async def test_budget_hint_lands_in_exchange(self):
        call = ToolCall(id="1", name="list_directory", arguments={})
        registry = FakeRegistry()
        outcome, exch = await _dispatch(
            call, registry=registry, steps_used=9, max_steps=10
        )
        assert "URGENT" in exch[0][1]
        assert "9/10 steps used" in exch[0][1]

    async def test_arguments_repaired_before_execution(self):
        call = ToolCall(id="1", name="execute_shell", arguments={"cmd": "ls"})
        registry = FakeRegistry()
        outcome, exch = await _dispatch(call, registry=registry)

        # The call was normalized in place…
        assert call.arguments == {"command": "ls"}
        # …and the executor saw the repaired arguments.
        assert registry.calls[0][1] == {"command": "ls"}

    async def test_result_attached_to_event(self):
        call = ToolCall(id="1", name="list_directory", arguments={})
        registry = FakeRegistry(result=ToolResult(output="payload", exit_code=1))
        outcome, _ = await _dispatch(call, registry=registry)
        tool_result_event = next(
            e for e in outcome.events if e.type == EventType.TOOL_RESULT
        )
        assert tool_result_event.tool_result is registry.result
        assert outcome.tool_result is registry.result


# ─────────────────────────────────────────────────────────────────────────────
# Safety gates: BLOCKED and DECLINED paths
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyGates:
    async def test_blocked_never_executes(self):
        call = ToolCall(id="1", name="execute_shell", arguments={"command": "rm -rf /"})
        registry = FakeRegistry()
        outcome, exch = await _dispatch(call, registry=registry)

        assert [e.type for e in outcome.events] == [EventType.BLOCKED]
        assert registry.calls == []
        assert len(exch) == 1
        assert exch[0][1].startswith("BLOCKED:")

    async def test_confirm_declined_never_executes(self):
        async def decline(tc: ToolCall) -> bool:
            return False

        call = ToolCall(
            id="1", name="execute_shell", arguments={"command": "echo confirm-me"}
        )
        registry = FakeRegistry()
        outcome, exch = await _dispatch(call, registry=registry, confirm=decline)

        assert [e.type for e in outcome.events] == [EventType.CONFIRM_NEEDED]
        assert registry.calls == []
        assert exch[0][1].startswith("DECLINED:")

    async def test_confirm_accepted_executes(self):
        async def accept(tc: ToolCall) -> bool:
            return True

        call = ToolCall(
            id="1", name="execute_shell", arguments={"command": "echo confirm-me"}
        )
        registry = FakeRegistry()
        outcome, _ = await _dispatch(call, registry=registry, confirm=accept)

        types = [e.type for e in outcome.events]
        assert types == [
            EventType.CONFIRM_NEEDED,
            EventType.THINKING,
            EventType.EXECUTING,
            EventType.TOOL_RESULT,
        ]
        assert registry.calls == [("execute_shell", {"command": "echo confirm-me"})]

    async def test_confirm_mode_safe_tool_still_executes(self):
        """The classifier's fails-closed rule: no rule → CONFIRM, so with a
        confirm callback the call is gated exactly like any other."""
        safe = SafetyClassifier(SafetyConfig(mode="confirm"))
        call = ToolCall(id="1", name="list_directory", arguments={})
        registry = FakeRegistry()

        # without a callback there is nothing to confirm against → it runs
        outcome, _ = await _dispatch(call, registry=registry, safety=safe)
        types = [e.type for e in outcome.events]
        assert EventType.TOOL_RESULT in types


# ─────────────────────────────────────────────────────────────────────────────
# Repetition guard stop → DispatchOutcome.stop
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardStop:
    async def test_duplicate_abort_sets_stop_flag(self):
        guard = RepetitionGuard()
        call = ToolCall(id="1", name="execute_shell", arguments={"command": "ping -c1 x"})

        stops = []
        for _ in range(5):  # 5 identical calls trigger the dupe-abort
            registry = FakeRegistry()
            exch: list = []
            outcome = await dispatch_tool_call(
                call,
                tools=registry,  # type: ignore[arg-type]
                safety=_safety(),
                guard=guard,
                confirm_callback=None,
                password_callback=None,
                steps_used=2,
                max_steps=10,
                append_exchange=lambda tc, content: exch.append((tc, content)),
            )
            stops.append(outcome.stop)

        # the last dispatch saw the guard's stop observation
        assert stops[-1] is True
        assert False in stops  # earlier dispatches did not already stop


# ─────────────────────────────────────────────────────────────────────────────
# Batch dispatch (R2-2)
#
# dispatch_tool_batch groups the calls in one batch so that consecutive
# PARALLEL_SAFE_TOOLS calls (list_directory, natshell_help, skill, fetch_url,
# kiwix_search) run concurrently via asyncio.gather while mutating / stateful
# calls (execute_shell, read_file, search_files, write_file, edit_file, …)
# keep the historical one-at-a-time dispatch.  Event and exchange order is
# the in-batch concatenation of each call's own events, so for every batch
# shape the tests below pin it matches the old serial loop exactly.
#
# Guard semantics under concurrency: RepetitionGuard.observe() is synchronous
# inside each coroutine, so a concurrent run of identical calls still
# increments the dupe counter to the abort threshold — the guard's stop fires
# on the last call, exactly the serial loop's end state, with all results
# delivered.  And once a stop fires, any later segments are skipped (the old
# loop broke out of the remaining calls), pinned below.
# ─────────────────────────────────────────────────────────────────────────────


def test_parallel_safe_set_is_the_read_only_allowlist_minus_none():
    """The allowlist must stay a subset of the classifier's read-only set —
    a tool that can mutate state must never take the concurrent path."""
    import natshell.safety.classifier as clf

    assert PARALLEL_SAFE_TOOLS <= clf._READ_ONLY_TOOLS


class FakeRegistryWithDelay(FakeRegistry):
    """Execute() sleeps a bit — start timestamps expose real concurrency."""

    def __init__(self, result=None, delay: float = 0.0) -> None:
        super().__init__(result)
        self.delay = delay
        self.start_times: list[float] = []

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        self.start_times.append(time.monotonic())
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        self.calls.append((name, arguments))
        return self.result


async def _dispatch_batch(
    calls: list[ToolCall],
    *,
    registry: FakeRegistry | None = None,
    safety: SafetyClassifier | None = None,
    guard: RepetitionGuard | None = None,
    confirm=None,
) -> tuple[DispatchOutcome, list[tuple[ToolCall, str]]]:
    """Test helper — runs the whole batch through dispatch_tool_batch."""
    exch: list[tuple[ToolCall, str]] = []
    registry = registry or FakeRegistry()
    outcome = await dispatch_tool_batch(
        calls,
        tools=registry,  # type: ignore[arg-type]
        safety=safety or _safety(),
        guard=guard or RepetitionGuard(),
        confirm_callback=confirm,
        password_callback=None,
        steps_used=2,
        max_steps=10,
        append_exchange=lambda tc, content: exch.append((tc, content)),
    )
    return outcome, exch


async def test_single_call_batch_fast_path():
    """One-call batch delegates to the per-call dispatcher — the exact
    event/exchange shape the existing suite pins on dispatch_tool_call."""
    call = ToolCall(id="1", name="list_directory", arguments={})
    registry = FakeRegistry()
    outcome, exch = await _dispatch_batch([call], registry=registry)

    assert [e.type for e in outcome.events] == [
        EventType.THINKING,
        EventType.EXECUTING,
        EventType.TOOL_RESULT,
    ]
    assert registry.calls == [("list_directory", {})]
    assert len(exch) == 1
    assert exch[0][1].startswith("out")
    assert outcome.stop is False


async def test_single_blocked_call_batch():
    """A single blocked call takes the same fast path — BLOCKED event,
    DECLINE/execute semantics byte-identical to the serial dispatcher."""
    call = ToolCall(id="1", name="execute_shell", arguments={"command": "rm -rf /"})
    registry = FakeRegistry()
    outcome, exch = await _dispatch_batch([call], registry=registry)

    assert [e.type for e in outcome.events] == [EventType.BLOCKED]
    assert registry.calls == []
    assert exch[0][1].startswith("BLOCKED:")
    assert outcome.stop is False


async def test_mixed_batch_runs_in_order_with_serial_events():
    """A parallel-safe call followed by a CONFIRM-gated mutating call:
    both execute, in batch order, with the confirm gate on the second."""
    safe_call = ToolCall(id="1", name="list_directory", arguments={"path": "/tmp"})
    serial_call = ToolCall(
        id="2", name="execute_shell", arguments={"command": "echo confirm-me"}
    )
    registry = FakeRegistry()

    async def _accept(tc: ToolCall) -> bool:
        return True

    outcome, exch = await _dispatch_batch(
        [safe_call, serial_call], registry=registry, confirm=_accept
    )

    # Events concatenated in batch order: call 1's trio, then call 2's
    # confirm-gated quadrant — exactly the old serial loop's interleaving.
    assert [e.type for e in outcome.events] == [
        EventType.THINKING,
        EventType.EXECUTING,
        EventType.TOOL_RESULT,
        EventType.CONFIRM_NEEDED,
        EventType.THINKING,
        EventType.EXECUTING,
        EventType.TOOL_RESULT,
    ]
    assert [n for n, _ in registry.calls] == ["list_directory", "execute_shell"]
    assert [tc.id for tc, _ in exch] == ["1", "2"]
    assert outcome.stop is False


async def test_mixed_batch_declined_serial_call_stops_execution_not_batch():
    """A declined CONFIRM call mid-batch appends its DECLINED exchange and
    the *subsequent* calls still run — matching the serial loop, which
    only broke on a guard ``stop``, never on a decline."""
    serial_call = ToolCall(
        id="1", name="execute_shell", arguments={"command": "echo confirm-me"}
    )
    safe_call = ToolCall(id="2", name="list_directory", arguments={"path": "/tmp"})
    registry = FakeRegistry()

    async def _decline(tc: ToolCall) -> bool:
        return False

    outcome, exch = await _dispatch_batch(
        [serial_call, safe_call], registry=registry, confirm=_decline
    )

    assert [tc.id for tc, _ in exch] == ["1", "2"]
    assert exch[0][1].startswith("DECLINED:")
    assert exch[1][1].startswith("out")
    assert [n for n, _ in registry.calls] == ["list_directory"]
    assert outcome.stop is False


async def test_parallel_safe_batch_runs_concurrently():
    """Two consecutive parallel-safe calls overlap in time: the 2nd
    execute() must start well before the 1st has finished."""
    registry = FakeRegistryWithDelay(delay=0.2)
    a = ToolCall(id="1", name="list_directory", arguments={"path": "/tmp"})
    b = ToolCall(id="2", name="list_directory", arguments={"path": "/var"})

    outcome, exch = await _dispatch_batch([a, b], registry=registry)

    assert len(registry.start_times) == 2
    gap = registry.start_times[1] - registry.start_times[0]
    # Serial would show a gap of ~200 ms; concurrent shows ≈ 0.
    assert gap < 0.10, f"expected overlapping starts, gap was {gap:.4f}s"
    # …and both results still land, in batch order.
    assert len(exch) == 2
    assert [tc.id for tc, _ in exch] == ["1", "2"]
    assert outcome.stop is False


async def test_five_identical_parallel_safe_calls_still_trigger_abort():
    """Concurrent does NOT bypass the guard: 5 identical parallel-safe
    calls in one batch still fire the duplicate-abort on the last
    observation — stop is True and the last exchange carries the CRITICAL
    suffix, while all five results were delivered (the serial loop's end
    state).  ``observe()`` is synchronous inside each coroutine, so the
    dupe counter still climbs to the threshold even when executions
    overlap in time."""
    batch = [
        ToolCall(id=str(i), name="list_directory", arguments={"path": "/tmp"})
        for i in range(5)
    ]
    registry = FakeRegistry()
    outcome, exch = await _dispatch_batch(batch, registry=registry)

    assert outcome.stop is True
    assert "CRITICAL" in exch[-1][1]
    assert len(registry.calls) == 5
    assert len(exch) == 5
    assert [tc.id for tc, _ in exch] == ["0", "1", "2", "3", "4"]


async def test_five_identical_serial_calls_do_trigger_abort():
    """Contrast pin: the same 5-identical batch on the *serial* path
    (execute_shell ∉ PARALLEL_SAFE_TOOLS) fires the duplicate-abort on the
    5th call — the old loop's break-on-stop, surfaced as outcome.stop."""
    batch = [
        ToolCall(
            id=str(i), name="execute_shell", arguments={"command": "ping -c1 x"}
        )
        for i in range(5)
    ]
    registry = FakeRegistry()
    outcome, exch = await _dispatch_batch(batch, registry=registry)

    assert outcome.stop is True
    assert "CRITICAL" in exch[-1][1]
    assert len(registry.calls) == 5
    assert len(exch) == 5


async def test_stop_halts_later_segments_in_batch():
    """Once a segment's guard observation sets ``stop`` (old loop's
    ``break``), dispatch must skip the remaining segments — a serial
    mutating call that fires the abort followed by a parallel-safe call
    must NOT execute; under the old inline loop it was never reached.
    Its events and exchange append are absent, matching serial history."""
    batch = [
        ToolCall(id=str(i), name="execute_shell", arguments={"command": "ping -c1 x"})
        for i in range(5)
    ]
    batch.append(ToolCall(id="9", name="list_directory", arguments={"path": "/tmp"}))
    registry = FakeRegistry()
    outcome, exch = await _dispatch_batch(batch, registry=registry)

    assert outcome.stop is True
    # only the 5 serial calls ran — the trailing list_directory was skipped
    assert len(registry.calls) == 5
    assert len(exch) == 5
    assert all(n == "execute_shell" for n, _ in registry.calls)
