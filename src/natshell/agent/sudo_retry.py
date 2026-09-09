"""Sudo-password retry for shell tools.

Extracted from :mod:`natshell.agent.loop`, which previously inlined the whole
"prompt for password → cache it → maybe prepend sudo → re-classify →
confirm → re-execute" flow.  Behaviour (event order, re-classification,
double-sudo avoidance, the exact "continued"/"skipped" semantics) is
preserved exactly.

The helper is an async function (not an async generator) that collects
emitted :class:`AgentEvent` items via ``on_event`` and returns the terminal
:class:`SudoRetryOutcome`.  The caller is responsible for yielding the
collected events *in order* before reading the outcome — this is what
preserves the chronological position of the events in the loop's event
stream exactly as the previous inline version produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from natshell.agent.events import AgentEvent, EventType
from natshell.inference.engine import ToolCall
from natshell.safety.classifier import Risk, SafetyClassifier
from natshell.tools.execute_shell import (
    _has_sudo_invocation,
    needs_sudo_password,
    set_sudo_password,
)
from natshell.tools.registry import ToolRegistry, ToolResult


@dataclass
class SudoRetryOutcome:
    """Terminal state of a sudo-password retry attempt.

    Attributes:
        status: One of ``"retried"`` (a second execute call happened and
            its result is available), ``"blocked"`` (re-classification of
            the sudo-prefixed command came back blocked),
            ``"declined_password"`` (user cancelled the password prompt
            before any retry could happen), or ``"declined_confirm"``
            (user declined the re-classified command).
        tool_result: The re-executed result when ``status == "retried"``.
        retry_args: The arguments actually used for the retry.
    """

    status: str
    tool_result: ToolResult | None = None
    retry_args: dict[str, Any] | None = None


def should_attempt(
    tool_call: ToolCall, tool_result: ToolResult, has_password_cb: bool
) -> bool:
    """True when the loop should enter the sudo-password flow at all."""
    return (
        tool_call.name == "execute_shell"
        and has_password_cb
        and needs_sudo_password(tool_result)
    )


async def run(
    tool_call: ToolCall,
    *,
    tools: ToolRegistry,
    safety: SafetyClassifier,
    password_callback: Callable[[ToolCall], Any],
    confirm_callback: Callable[[ToolCall], Any] | None = None,
    on_event: Callable[[AgentEvent], None] | None = None,
) -> SudoRetryOutcome:
    """Prompt for the password and (possibly) re-execute.

    ``on_event`` is an optional collector invoked for each
    :class:`AgentEvent` in chronological order (CONFIRM_NEEDED, THINKING,
    BLOCKED).  When ``on_event`` is None, events are still produced
    internally but discarded; outcome-only callers (e.g. headless auto-approve)
    work fine without them.

    ``password_callback`` / ``confirm_callback`` are ``async (ToolCall) ->
    str | bool`` callables; the loop supplies whichever UI it has.
    """

    def emit(ev: AgentEvent) -> None:
        if on_event is not None:
            on_event(ev)

    password = await password_callback(tool_call)
    if not password:
        return SudoRetryOutcome(status="declined_password")

    # Cache the password so execute_shell can inject it via ``sudo -S``.
    set_sudo_password(password)

    # If the command does not already contain sudo at a command position
    # (e.g. "apt install" which internally invokes sudo), prepend it so the
    # password injection in execute_shell kicks in.
    retry_args = dict(tool_call.arguments)
    cmd = retry_args.get("command", "")
    if cmd and not _has_sudo_invocation(cmd):
        retry_args["command"] = f"sudo {cmd}"

    # Re-classify the modified command — prepending sudo may change the risk
    # level.
    retry_risk = safety.classify_tool_call(tool_call.name, retry_args)
    if retry_risk == Risk.BLOCKED:
        emit(AgentEvent(type=EventType.BLOCKED, tool_call=tool_call))
        return SudoRetryOutcome(status="blocked")

    if retry_risk == Risk.CONFIRM and confirm_callback:
        emit(AgentEvent(type=EventType.CONFIRM_NEEDED, tool_call=tool_call))
        confirmed = await confirm_callback(tool_call)
        if not confirmed:
            return SudoRetryOutcome(status="declined_confirm")

    # Restart thinking animation before execution (exactly as the inline
    # version did before the retry).
    emit(AgentEvent(type=EventType.THINKING))
    tool_result = await tools.execute(tool_call.name, retry_args)
    return SudoRetryOutcome(
        status="retried", tool_result=tool_result, retry_args=retry_args
    )
