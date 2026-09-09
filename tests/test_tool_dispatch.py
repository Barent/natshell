"""Tests for natshell.agent.tool_dispatch.

Unit tests for the single-tool-call dispatcher extracted from
``AgentLoop.handle_user_message`` (R1-7 final chunk): the step-budget hint
and the per-call lifecycle (normalize → classify → confirm → execute →
observe).  The loop-level integration (event interleaving, sudo-retry
ladder, guard stop breaking the batch) is pinned by ``tests/test_agent.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from natshell.agent.events import EventType
from natshell.agent.repetition_guard import RepetitionGuard
from natshell.agent.tool_dispatch import (
    DispatchOutcome,
    dispatch_tool_call,
    step_budget_hint,
)
from natshell.config import SafetyConfig
from natshell.inference.engine import ToolCall
from natshell.safety.classifier import SafetyClassifier
from natshell.tools.registry import ToolRegistry, ToolResult


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
