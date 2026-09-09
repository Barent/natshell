"""Per-step response handling — inference accounting & outcome banners.

Extracted from ``AgentLoop.handle_user_message`` (R1-7).  Three concerns the
loop used to inline now live in one place:

* :func:`build_metrics` / :func:`build_run_stats` — the dict builders for
  per-response metrics and end-of-run cumulative stats (re-exported from
  :mod:`natshell.agent.loop` so existing imports keep working).
* :class:`RunStats` — the per-run cumulative counters (inference ms,
  prompt/completion tokens) together with ``t0``.  The loop hands a single
  instance to its helpers and the max-steps epilogue instead of threading
  five locals through every call.
* :class:`StepOutcome` / :class:`StepControl` — the return type of the
  degenerate-output and token-limit handlers.  The loop only needs to know
  *what to do next* (retry the step vs. finish the run) plus the
  chronological :class:`AgentEvent` items to yield.  Event text is
  byte-identical to the pre-extraction inline code in ``loop.py``; the test
  suites (``TestDegenerateAgentLoop``, ``TestTruncatedResponse``) pin it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from natshell.agent.events import AgentEvent, EventType
from natshell.inference.engine import CompletionResult

# The think-block strip inlined by handle_user_message is byte-identical to
# the shared grammar regexes — reuse them instead of a third copy.  (Both
# match closed and trailing-unclosed blocks, DOTALL.)
from natshell.inference.grammars.common import (  # noqa: E402
    THINK_RE,
    THINK_UNCLOSED_RE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Metrics / stats builders (moved verbatim from loop.py, R1-7)
# ─────────────────────────────────────────────────────────────────────────────

def build_metrics(result: CompletionResult, elapsed_ms: int) -> dict[str, Any]:
    """Build a metrics dict from inference result and timing."""
    metrics: dict[str, Any] = {"response_time_ms": elapsed_ms}
    if result.completion_tokens:
        metrics["completion_tokens"] = result.completion_tokens
        if elapsed_ms > 0:
            metrics["tokens_per_sec"] = result.completion_tokens / (elapsed_ms / 1000)
    if result.prompt_tokens:
        metrics["prompt_tokens"] = result.prompt_tokens
    return metrics


def build_run_stats(
    steps: int,
    total_wall_ms: int,
    total_inference_ms: int,
    total_prompt_tokens: int,
    total_completion_tokens: int,
) -> dict[str, Any]:
    """Build cumulative stats for an entire agent run."""
    stats: dict[str, Any] = {
        "steps": steps,
        "total_wall_ms": total_wall_ms,
        "total_inference_ms": total_inference_ms,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
    }
    total_tokens = total_prompt_tokens + total_completion_tokens
    if total_tokens:
        stats["total_tokens"] = total_tokens
    if total_inference_ms > 0 and total_completion_tokens:
        stats["avg_tokens_per_sec"] = total_completion_tokens / (total_inference_ms / 1000)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Per-run cumulative counters (formerly locals in handle_user_message)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunStats:
    """Cumulative stats for one ``handle_user_message`` run.

    Mutable on purpose: the loop keeps a single instance across steps and
    the outcome handlers read it in place — exactly as the locals they
    replace did.
    """

    t0: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    inference_ms: int = 0
    # Wall-clock ms of the most recent inference (set by the loop so the
    # outcome handlers can attach metrics, as ``elapsed_ms`` did inline.)
    last_elapsed_ms: int = 0

    def accumulate(self, result: CompletionResult, elapsed_ms: int) -> None:
        """Fold one successful inference's accounting into the run totals."""
        self.last_elapsed_ms = elapsed_ms
        self.inference_ms += elapsed_ms
        self.prompt_tokens += result.prompt_tokens or 0
        self.completion_tokens += result.completion_tokens or 0

    def run_stats(self, steps_used: int, now: float) -> dict[str, Any]:
        """End-of-run cumulative stats (wall time measured against ``t0``)."""
        return build_run_stats(
            steps_used,
            int((now - self.t0) * 1000),
            self.inference_ms,
            self.prompt_tokens,
            self.completion_tokens,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Outcome handlers — degenerate output & token-limit truncation
# ─────────────────────────────────────────────────────────────────────────────

class StepControl(Enum):
    """What the agent loop should do after an outcome handler.

    ``RETRY``  — the current step should run again (context was compacted).
    ``STOP``   — the run is over; the handler already produced the events
                 for the epilogue (RESPONSE / ERROR / RUN_STATS as needed).
    """

    RETRY = "retry"
    STOP = "stop"


@dataclass
class StepOutcome:
    """Result of an outcome handler.

    Attributes:
        control: Whether the loop should retry the step or finish the run.
        events:  AgentEvents to yield, in chronological order.
        partial: When a truncated response was partially surfaced, the text
                 the loop must append to ``self.messages`` (message mutation
                 stays in the loop).
    """

    control: StepControl
    events: list[AgentEvent]
    partial: str | None = None


def strip_think_residue(raw: str) -> str:
    """Strip think blocks (closed and trailing unclosed) from ``raw``.

    Returns the leftover text, stripped.  Empty means the model spent its
    entire output budget thinking (or produced nothing).  Uses the shared
    grammar regexes — byte-identical to the two ``re.sub``s the loop used
    to run inline.
    """
    stripped = THINK_RE.sub("", raw)
    stripped = THINK_UNCLOSED_RE.sub("", stripped)
    return stripped.strip()


def handle_degenerate_output(
    result: CompletionResult,
    *,
    compact_stats: dict[str, Any],
) -> StepOutcome:
    """Handle degenerate output (repetitive garbage from local models).

    ``result`` is part of the handler API (and useful for logging) but only
    ``compact_stats`` drives the outcome: it is the result of
    ``AgentLoop.compact_history()``, passed in because message mutation
    belongs to the loop.
    """
    del result  # handler API surface; the outcome depends only on compact_stats
    if compact_stats.get("compacted"):
        return StepOutcome(
            control=StepControl.RETRY,
            events=[
                AgentEvent(
                    type=EventType.ERROR,
                    data=(
                        "Model produced degenerate output "
                        "(repeated characters). Automatically "
                        "compacted conversation — retrying."
                    ),
                )
            ],
        )
    return StepOutcome(
        control=StepControl.STOP,
        events=[
            AgentEvent(
                type=EventType.ERROR,
                data=(
                    "Model produced degenerate output "
                    "(repeated characters). The context window "
                    "may be full. Try /clear to reset."
                ),
            )
        ],
    )


def handle_token_limit(
    result: CompletionResult,
    *,
    steps_used: int,
    stats: RunStats,
    now: float,
) -> StepOutcome:
    """Handle a ``finish_reason == "length"`` response with no tool calls.

    Sets ``outcome.partial`` when a partial (think-stripped) response
    should be remembered in history; the loop appends it to
    ``self.messages`` (message mutation stays in the loop).
    """
    stripped = strip_think_residue(result.content or "")
    if stripped:
        # Partial response — show it but warn the user
        outcome = StepOutcome(
            control=StepControl.STOP,
            events=[
                AgentEvent(
                    type=EventType.RESPONSE,
                    data=stripped,
                    metrics=build_metrics(result, stats.last_elapsed_ms),
                ),
                AgentEvent(
                    type=EventType.ERROR,
                    data="Response was truncated (hit token limit). "
                    "The context window may be full. Try /clear to reset.",
                ),
            ],
            partial=stripped,
        )
        if steps_used > 1:
            outcome.events.append(
                AgentEvent(type=EventType.RUN_STATS, metrics=stats.run_stats(steps_used, now))
            )
        return outcome
    return StepOutcome(
        control=StepControl.STOP,
        events=[
            AgentEvent(
                type=EventType.ERROR,
                data="Response was truncated — the model used all"
                " available tokens without producing a complete"
                " response. Try a simpler request or /clear to reset.",
            )
        ],
    )
