"""Single tool-call dispatch — classify, confirm, execute, observe, budget-hint.

Extracted from ``AgentLoop.handle_user_message`` (R1-7, final chunk).  The
loop used to inline one whole tool call's lifecycle; :func:`dispatch_tool_call`
now owns it for exactly one :class:`ToolCall`, in the *same chronological
order* the inline code ran:

1. normalize arguments (mutates ``tool_call.arguments`` before anything sees
   the call — classifier, dialog and executor all judge the repaired call),
2. safety classification (BLOCKED → banner + exchange + next call),
3. confirmation (declined → exchange + next call),
4. THINKING + EXECUTING, execute,
5. the sudo-password retry ladder (``natshell.agent.sudo_retry``),
6. repetition-guard bookkeeping + TOOL_RESULT event,
7. the guard's observation suffix + step-budget hint,
8. append the exchange; ``stop`` (duplicate-abort / family hard-stop /
   similar-cmd abort) tells the loop to leave the tool-call batch.

The function *buffers* its :class:`AgentEvent` list and returns it along with
the ``stop`` flag; the loop yields the buffered events in order and breaks
the batch when ``stop`` is set.  That preserves the exact interleaving the
inline version produced (events, side effects and flow in the same order),
with message mutation still happening in the loop-supplied callback.

:func:`step_budget_hint` is the pure step-exhaustion suffix the model used
to get inline — now unit-testable on its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from natshell.agent.events import AgentEvent, EventType
from natshell.agent.repetition_guard import RepetitionGuard
from natshell.agent.sudo_retry import run as sudo_retry_run
from natshell.inference.engine import ToolCall
from natshell.safety.classifier import Risk, SafetyClassifier
from natshell.tools.execute_shell import needs_sudo_password
from natshell.tools.registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


def step_budget_hint(steps_used: int, max_steps: int) -> str:
    """Step-exhaustion suffix appended to tool results as the budget drains.

    Pure: no events, no state — the thresholds, wording and step-1 welcome
    line match the version ``handle_user_message`` used to build inline.
    Returns ``""`` when nothing should be appended.
    """
    pct_used = steps_used / max_steps
    if pct_used >= 0.90:
        remaining = max_steps - steps_used
        return (
            f"\n\n⚠ URGENT: [{steps_used}/{max_steps} steps used"
            f" — only {remaining} steps left. Finish NOW.]"
        )
    if pct_used >= 0.75:
        return (
            f"\n\n⚠ [{steps_used}/{max_steps} steps used"
            " — wrap up soon]"
        )
    if pct_used >= 0.50:
        return (
            f"\n\n[{steps_used}/{max_steps} steps used]"
        )
    if steps_used == 1:
        return (
            f"\n\n[Budget: {max_steps} steps available"
            " — plan your approach before diving in]"
        )
    return ""


@dataclass
class DispatchOutcome:
    """Result of dispatching one tool call.

    Attributes:
        events: AgentEvents to yield, in chronological order (the exact order
            the inline version produced them, including the sudo-retry
            ladder's events in their historical position).
        stop:  True when the repetition guard issued a stop observation —
            the loop leaves the tool-call batch (``break``), matching the
            inline version's flow exactly.
        tool_result: The (possibly sudo-retried) result, when execution
            happened; None on BLOCKED / DECLINED paths.  Exposed so callers
            that need the raw result (none today) don't have to re-execute.
    """

    events: list[AgentEvent] = field(default_factory=list)
    stop: bool = False
    tool_result: Any = None


async def dispatch_tool_call(
    tool_call: ToolCall,
    *,
    tools: ToolRegistry,
    safety: SafetyClassifier,
    guard: RepetitionGuard,
    confirm_callback: Callable[[ToolCall], Any] | None,
    password_callback: Callable[[ToolCall], Any] | None = None,
    steps_used: int = 1,
    max_steps: int = 15,
    append_exchange: Callable[[ToolCall, str], None],
) -> DispatchOutcome:
    """Run exactly one tool call through classify → confirm → execute → observe.

    Message mutation stays in the loop: the loop passes ``append_exchange``
    (bound to ``AgentLoop._append_tool_exchange``) and it is invoked in the
    same places the inline code called it.
    """
    events: list[AgentEvent] = []

    # Bind the arguments to their parameter names *before* classifying, so
    # the classifier judges the call that will actually run.  execute()
    # repairs misnamed parameters by position, which used to happen after
    # this point: a model that wrote "cmd" instead of "command" -- a mistake
    # the remap exists because small models make constantly -- got classified
    # against an absent key and ran unconfirmed.
    #
    # Assigning back onto the call also makes the confirmation dialog show
    # the arguments the tool will receive.
    normalized = tools.normalize_arguments(tool_call.name, tool_call.arguments)
    if normalized is not None:
        tool_call.arguments = normalized

    # Safety classification
    risk = safety.classify_tool_call(tool_call.name, tool_call.arguments)
    logger.debug("Tool %s classified as %s", tool_call.name, risk.name)

    if risk == Risk.BLOCKED:
        events.append(AgentEvent(type=EventType.BLOCKED, tool_call=tool_call))
        append_exchange(
            tool_call,
            "BLOCKED: This command was blocked by the safety classifier. "
            "Try an alternative approach.",
        )
        return DispatchOutcome(events=events)

    if risk == Risk.CONFIRM and confirm_callback:
        events.append(
            AgentEvent(type=EventType.CONFIRM_NEEDED, tool_call=tool_call)
        )
        confirmed = await confirm_callback(tool_call)
        if not confirmed:
            append_exchange(
                tool_call,
                "DECLINED: The user declined to execute this command.",
            )
            return DispatchOutcome(events=events)

    # Restart thinking animation before execution
    events.append(AgentEvent(type=EventType.THINKING))
    # Execute the tool
    events.append(AgentEvent(type=EventType.EXECUTING, tool_call=tool_call))

    tool_result: ToolResult = await tools.execute(tool_call.name, tool_call.arguments)
    logger.debug(
        "Tool %s → exit_code=%s, output_len=%d",
        tool_call.name,
        tool_result.exit_code,
        len(tool_result.output or ""),
    )

    # If sudo needs a password, prompt the user and retry.
    # (Event order + re-classification preserved via natshell.agent.sudo_retry
    #  — see that module.  Its events are appended to ours in their
    #  historical position, so the loop's yield order is unchanged.)
    if (
        tool_call.name == "execute_shell"
        and password_callback
        and needs_sudo_password(tool_result)
    ):
        outcome = await sudo_retry_run(
            tool_call,
            tools=tools,
            safety=safety,
            password_callback=password_callback,
            confirm_callback=confirm_callback,
            on_event=events.append,
        )
        if outcome.status == "retried" and outcome.tool_result is not None:
            tool_result = outcome.tool_result
        else:
            # blocked / declined_password / declined_confirm
            if outcome.status == "blocked":
                append_exchange(
                    tool_call,
                    "BLOCKED: The retried command with sudo was "
                    "blocked by the safety classifier.",
                )
            elif outcome.status == "declined_confirm":
                append_exchange(
                    tool_call,
                    "DECLINED: The user declined the retried "
                    "command with sudo.",
                )
            elif outcome.status == "declined_password":
                append_exchange(
                    tool_call,
                    "DECLINED: The user cancelled the sudo "
                    "password prompt.",
                )
            return DispatchOutcome(events=events, tool_result=tool_result)

    # Record the (possibly retried) result + ask the guard whether it should
    # warn the model.  The guard also owns the edit-succeed/fail bookkeeping
    # and the read-count reset, in the same order the inline version had.
    guard.register_outcome(tool_call.name, tool_result.exit_code)

    # Yield the tool result event.  (This matches the original order:
    # TOOL_RESULT fires *before* the guard checks so the TUI sees the raw
    # result either way.)
    events.append(
        AgentEvent(
            type=EventType.TOOL_RESULT,
            tool_call=tool_call,
            tool_result=tool_result,
        )
    )

    # Observations (repetition warnings + stop signal).
    # NOTE: we must skip the register_outcome path inside .observe() here —
    # already done above — otherwise the counters would double-count.
    observation = guard.observe(
        name=tool_call.name,
        arguments=tool_call.arguments,
        exit_code=tool_result.exit_code,
        record=False,
    )
    result_content = (
        tool_result.to_message_content() + observation.suffix
    )

    # Step budget awareness (pure suffix; thresholds & wording unit-tested
    # against step_budget_hint).
    result_content += step_budget_hint(steps_used, max_steps)

    # Append exchange to conversation history.  A ``stop`` observation
    # (duplicate-abort, command-family hard stop, similar-cmd abort) ends
    # the tool-dispatch batch the same way the inline version did.
    append_exchange(tool_call, result_content)
    return DispatchOutcome(events=events, stop=observation.stop, tool_result=tool_result)
