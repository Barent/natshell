"""Tool-call batch dispatch — classify, confirm, execute, observe, budget-hint.

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

:func:`dispatch_tool_batch` (R2-2) groups a batch of tool calls so that
consecutive :data:`PARALLEL_SAFE_TOOLS` calls — tools that are pure or free
of shared mutable state (``list_directory``, ``natshell_help``, ``skill``,
``fetch_url``, ``kiwix_search``) — run concurrently via ``asyncio.gather``
while any mutating or guard-stateful call (``execute_shell``, ``read_file``,
``search_files``, ``write_file``, ``edit_file``, ``run_code``, …) keeps the
serial dispatch path, one at a time.  Events and exchange ordering are the
in-batch concatenation of each call's own events in batch order — identical
to the serial version for every call; only execution concurrency changes.
A guard ``stop`` observation halts the remaining batch segments (the old
inline loop also broke out of the remaining calls), while the run itself
continues to its next LLM step so the model sees the CRITICAL suffix.
"""

from __future__ import annotations

import asyncio
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


# Tools safe to run concurrently within one dispatch batch (R2-2).
# Candidate set is ``SafetyClassifier``'s ``_READ_ONLY_TOOLS`` (the tools
# that cannot mutate state); from it we subtract the two whose guard
# bookkeeping is order-dependent — ``read_file`` (read-count / dupe
# counters reset on writes) and ``search_files`` (same dupe-key path) —
# because reordering their observation could mask the repetition guard's
# detectors.
#
# The five kept are verified concurrency-safe:
#   - ``list_directory`` / ``skill`` / ``natshell_help`` are pure — they
#     read from the filesystem / the in-memory skill registry and never
#     touch shared module state.
#   - ``fetch_url`` creates a per-call ``httpx.AsyncClient`` inside an
#     ``async with`` (no shared client) and only ever *reads*
#     ``_limits`` — no global mutation on the hot path.
#   - ``kiwix_search`` also creates a per-call client; its ``global
#     _kiwix_url`` writes are the benign re-discovery fallback (same
#     CPython string replacement a serial call would do) and
#     ``_known_books`` is only ever appended at startup, so concurrent
#     reads during a call are safe.
PARALLEL_SAFE_TOOLS: frozenset[str] = frozenset(
    {
        "list_directory",
        "natshell_help",
        "skill",
        "fetch_url",
        "kiwix_search",
    }
)


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
    stream_output: bool = False,
) -> DispatchOutcome:
    """Run exactly one tool call through classify → confirm → execute → observe.

    Message mutation stays in the loop: the loop passes ``append_exchange``
    (bound to ``AgentLoop._append_tool_exchange``) and it is invoked in the
    same places the inline code called it.

    ``stream_output`` (R2-4): when True *and* the tool has a streaming
    handler registered on the registry (only ``execute_shell`` does), the
    tool runs through its streaming executor and each stdout chunk is
    surface as a TOOL_OUTPUT event, in arrival order, between the EXECUTING
    and TOOL_RESULT events.  When False — or the tool isn't streaming-capable
    — the historical blocking ``tools.execute`` path is taken, and no
    TOOL_OUTPUT events are emitted.  That keeps the pre-R2-4 event sequence
    byte-identical for every caller that doesn't opt in.
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

    tool_result: ToolResult | None = None
    _streamer = getattr(tools, "execute_streaming", None)
    if stream_output and _streamer is not None:

        def _chunk(text: str) -> None:
            """Surface one live stdout chunk as a TOOL_OUTPUT event.

            The dispatcher is a synchronous collector from the streaming
            executor's point of view, so we append in arrival order — the
            loop's eventual ``for ev in outcome.events: yield ev`` preserves
            that ordering exactly, interleaved between EXECUTING and
            TOOL_RESULT.
            """
            events.append(
                AgentEvent(type=EventType.TOOL_OUTPUT, tool_call=tool_call, data=text)
            )

        streamed = await _streamer(
            tool_call.name, tool_call.arguments, on_chunk=_chunk
        )
        if streamed is not None:
            tool_result = streamed
    if tool_result is None:
        tool_result = await tools.execute(tool_call.name, tool_call.arguments)
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


# ─────────────────────────────────────────────────────────────────────────────
# Batch dispatch (R2-2)
# ─────────────────────────────────────────────────────────────────────────────


async def dispatch_tool_batch(
    tool_calls: list[ToolCall],
    *,
    tools: ToolRegistry,
    safety: SafetyClassifier,
    guard: RepetitionGuard,
    confirm_callback: Callable[[ToolCall], Any] | None,
    password_callback: Callable[[ToolCall], Any] | None = None,
    steps_used: int = 1,
    max_steps: int = 15,
    append_exchange: Callable[[ToolCall, str], None],
    stream_output: bool = False,
) -> DispatchOutcome:
    """Dispatch a batch of tool calls (R2-2).

    Runs consecutive :data:`PARALLEL_SAFE_TOOLS` calls concurrently via
    ``asyncio.gather``; mutating / stateful calls (``read_file``,
    ``search_files``, ``execute_shell``, ``write_file``, ``edit_file``,
    ``run_code``, and any tool not on the allowlist) each take the serial
    dispatch path — same lifecycle, same event order, same guard bookkeeping
    as before.

    The final :class:`DispatchOutcome` is the concatenation of each call's
    events in batch order; ``stop`` fires if any call's guard observation
    set it.  A ``stop`` also halts the dispatch of any later segments in
    the batch (the old inline loop also broke out of the remaining calls),
    while the run itself continues to its next LLM step so the model sees
    the CRITICAL suffix and can stop repeating.  Callers must NOT call
    :func:`dispatch_tool_call` separately on a batch — the loop uses
    :func:`dispatch_tool_batch` exclusively.

    For a batch of *one* call, this is byte-identical to one call to
    :func:`dispatch_tool_call`.  For a batch of only pure parallel-safe
    tools, the calls execute concurrently, but each call's guard
    observation still increments the dupe counter (``observe()`` is
    synchronous inside each coroutine, so a run of identical calls still
    reaches the abort threshold and fires ``stop`` on the last one —
    the serial loop's end state, with all results delivered).  For mixed
    batches the serial segments run one at a time in order after their
    neighbours, as the historical loop did.
    """
    if not tool_calls:
        return DispatchOutcome()
    if len(tool_calls) == 1:
        # Fast path: one call.  Identical to a single dispatch_tool_call and
        # preserves every behaviour the existing test suite pins.
        only = await dispatch_tool_call(
            tool_calls[0],
            tools=tools,
            safety=safety,
            guard=guard,
            confirm_callback=confirm_callback,
            password_callback=password_callback,
            steps_used=steps_used,
            max_steps=max_steps,
            append_exchange=append_exchange,
            stream_output=stream_output,
        )
        return only

    # Split the batch: consecutive PARALLEL_SAFE runs and everything else.
    # A "serial" item is any call NOT on the allowlist — mutating or
    # stateful tools (read_file, search_files, execute_shell, write_file,
    # edit_file, run_code, …) that must keep the historical one-at-a-time
    # dispatch; each also arrives as its own singleton segment.
    segments: list[list[ToolCall]] = []
    for tc in tool_calls:
        if tc.name in PARALLEL_SAFE_TOOLS:
            if segments and segments[-1][0].name in PARALLEL_SAFE_TOOLS:
                segments[-1].append(tc)
            else:
                segments.append([tc])
        else:
            segments.append([tc])

    async def _dispatch_one(tc: ToolCall) -> DispatchOutcome:
        return await dispatch_tool_call(
            tc,
            tools=tools,
            safety=safety,
            guard=guard,
            confirm_callback=confirm_callback,
            password_callback=password_callback,
            steps_used=steps_used,
            max_steps=max_steps,
            append_exchange=append_exchange,
            stream_output=stream_output,
        )

    outcomes: list[DispatchOutcome] = []
    stopped = False
    for segment in segments:
        # The old inline serial loop broke out of the remaining tool calls
        # when a guard observation set ``stop`` (the *run* continued to its
        # next LLM step — the model had to see the CRITICAL suffix to stop
        # repeating — but the rest of the current batch was skipped).  We
        # preserve that here: once any earlier segment's outcome reported
        # ``stop``, skip all later segments.  A single in-flight gather
        # cannot be cancelled from the outside, but the next segment is.
        if stopped:
            break
        if (
            len(segment) > 1
            and all(tc.name in PARALLEL_SAFE_TOOLS for tc in segment)
        ):
            # R2-2 fast path: pure read-only calls execute concurrently.
            # Events and exchange appends are buffered per-call by
            # dispatch_tool_call (so no interleaving hazard) and concatenated
            # in segment order, matching the serial result.
            outcome_group = await asyncio.gather(*(
                _dispatch_one(tc) for tc in segment
            ))
            for out in outcome_group:
                outcomes.append(out)
                if out.stop:
                    stopped = True
        else:
            # Serial path: mutating calls keep the historical one-at-a-time
            # dispatch.  A CONFIRM gate inside a gather would also be fine
            # but the historical event order is easiest to reason about
            # when the confirm dialog awaits are strictly sequential.
            out = await _dispatch_one(segment[0])
            outcomes.append(out)
            if out.stop:
                stopped = True

    all_events: list[AgentEvent] = []
    stop = False
    for outcome in outcomes:
        all_events.extend(outcome.events)
        if outcome.stop:
            stop = True
    return DispatchOutcome(events=all_events, stop=stop, tool_result=None)
