"""LLM compaction tier (R2-5) — summarizing a dropped window via the engine.

When :mod:`natshell.config`'s ``[compaction] llm`` switch is enabled,
compaction first asks the local model for a one-shot prose summary of the
dropped messages (this module).  On timeout, engine failure, or after
enough consecutive failures, the caller transparently falls back to the
extractive :meth:`ContextManager.build_summary` — compaction can never be
broken, empty, or hanging because of the LLM tier.

``ContextManager.summarizer`` (the R2-5a seam) remains the synchronous
extension point: it is honoured by the same code path when it is set, and
the LLM tier here is the *engine-backed* implementation enabled by config.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Consecutive summarizer failures after which the LLM tier stops being
#: tried for the remainder of the run (extractive fallback continues).
CONSEC_FAILURE_TRIP = 3

#: Max characters of a single message's content included in the prompt.
_MAX_CONTENT_CHARS = 400

SYSTEM_PROMPT = (
    "You summarize elided conversation history for an agent whose context "
    "window is full. Write a compact, factual summary of at most ~6 lines: "
    "which files were created or edited, which commands were run and their "
    "outcomes, and every user request not yet fulfilled. Output only the "
    "summary text — no preamble, no questions, no tool calls.\n\n"
    "/no_think"
)


class FailureTracker:
    """Per-run circuit breaker for repeated summarizer failures.

    After ``trip_after`` consecutive failures the tier is marked tripped;
    :func:`llm_summarize` then returns ``None`` immediately (skipping any
    engine I/O) until the tracker is :meth:`reset` for the next run.
    """

    def __init__(self, trip_after: int = CONSEC_FAILURE_TRIP) -> None:
        self._trip_after = max(1, trip_after)
        self.consecutive_failures = 0
        self.tripped = False

    def record_success(self) -> None:
        """A successful summarization resets the failure streak."""
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """A timeout or engine error extends the failure streak."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self._trip_after:
            self.tripped = True

    def reset(self) -> None:
        """Clear the streak and the tripped state (call per run)."""
        self.consecutive_failures = 0
        self.tripped = False


def render_messages(
    messages: list[dict[str, Any]], max_messages: int = 0
) -> str:
    """Render dropped messages compactly as a prompt body.

    System-role messages (prior compaction markers) are skipped — the facts
    they carry are already compressed forms of earlier windows.  Long
    payloads are elided per message so a single giant tool result cannot
    consume the entire summarizer prompt.
    """
    window = messages if max_messages <= 0 else list(messages[-max_messages:])
    lines: list[str] = []
    if len(window) < len(messages):
        lines.append(f"[{len(messages) - len(window)} older messages truncated]")
    for msg in window:
        if msg.get("role") == "system":
            continue
        role = msg.get("role", "")
        body = (msg.get("content") or "").strip().replace("\n", " ")
        if len(body) > _MAX_CONTENT_CHARS:
            body = body[:_MAX_CONTENT_CHARS] + "…"
        calls: list[str] = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", "")
            if len(args) > 160:
                args = args[:160] + "…"
            if name:
                calls.append(f"{name}({args})")
        line = f"[{role}] {body}".rstrip()
        if calls:
            line += "  →  " + " ".join(calls)
        if line:
            lines.append(line)
    return "\n".join(lines) or "(empty conversation)"


async def llm_summarize(
    engine: Any,
    messages: list[dict[str, Any]],
    *,
    max_messages: int = 0,
    timeout: float = 30.0,
    tracker: FailureTracker | None = None,
) -> str | None:
    """Ask *engine* for a one-shot summary of *messages*.

    Returns the summary text, or ``None`` when the LLM tier is unavailable
    (tripped breaker, timeout, engine error, or empty output) — callers
    must then fall back to the extractive summary so compaction always
    produces marker text.

    A timed-out worker may continue briefly in the background (the local
    inference runs in a worker thread that cannot be killed mid-token);
    the :class:`FailureTracker` limits repeat cost within a run.
    """
    if tracker is not None and tracker.tripped:
        return None
    if not messages:
        return None

    body = render_messages(messages, max_messages)
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Summarize the elided conversation below.\n\n{body}",
        },
    ]
    try:
        result = await asyncio.wait_for(
            engine.chat_completion(
                prompt, temperature=0.2, max_tokens=768
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "LLM compaction timed out after %.1fs — using extractive summary",
            timeout,
        )
        if tracker is not None:
            tracker.record_failure()
        return None
    except Exception:
        logger.warning(
            "LLM compaction engine error — using extractive summary",
            exc_info=True,
        )
        if tracker is not None:
            tracker.record_failure()
        return None

    content = getattr(result, "content", None)
    text = str(content or "").strip()
    if text:
        # Strip any family think-blocks the model may have wrapped its
        # summary in (same regexes the response pipeline uses).
        from natshell.inference.grammars.common import THINK_RE, THINK_UNCLOSED_RE

        text = THINK_RE.sub("", text)
        text = THINK_UNCLOSED_RE.sub("", text)
        text = text.strip()
    if not text:
        if tracker is not None:
            tracker.record_failure()
        return None
    if tracker is not None:
        tracker.record_success()
    return text
