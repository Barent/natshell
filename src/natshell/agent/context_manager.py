"""Context window management — token counting, budget tracking, message trimming.

Keeps the conversation history within the model's context window by trimming
older messages when the token budget is exceeded. The system prompt and recent
messages are always preserved; dropped messages are replaced with a compact
extractive summary.

Note: If conversation logging/persistence is ever added, it should respect a
size cap to avoid unbounded disk growth.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Minimum number of recent messages (after system prompt) to always preserve.
# This keeps at least 3 user/assistant exchanges in context.
_MIN_RECENT = 6


class ContextManager:
    """Manages message trimming to fit within a token budget.

    Two token-counting modes:
    - Exact: uses the model's tokenizer via ``tokenizer_fn``
    - Approximate: ``len(text) // 3`` (conservative for Qwen/English text)
    """

    def __init__(
        self,
        context_budget: int,
        tokenizer_fn: Callable[[str], int] | None = None,
    ) -> None:
        self.context_budget = context_budget
        self._tokenizer_fn = tokenizer_fn
        self.trimmed_count: int = 0  # total messages trimmed across all calls

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Estimate total token count for a list of messages."""
        return sum(self._msg_tokens(m) for m in messages)

    def _msg_tokens(self, msg: dict[str, Any]) -> int:
        """Estimate tokens for a single message."""
        parts: list[str] = []
        content = msg.get("content")
        if content:
            parts.append(content)
        # Account for tool call arguments embedded in assistant messages
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            parts.append(func.get("name", ""))
            parts.append(func.get("arguments", ""))
        text = " ".join(parts)
        return self._count(text)

    def _count(self, text: str) -> int:
        if self._tokenizer_fn is not None:
            try:
                result = self._tokenizer_fn(text)
                if isinstance(result, int):
                    return result
            except Exception:
                pass
        # Fallback: ~3 chars per token (conservative for English / Qwen)
        return max(1, len(text) // 3)

    # ------------------------------------------------------------------
    # Trimming
    # ------------------------------------------------------------------

    def trim_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Trim *messages* to fit within :pyattr:`context_budget`.

        Strategy:
        1. Always keep ``messages[0]`` (system prompt).
        2. Always keep the last *_MIN_RECENT* non-system messages.
        3. Drop oldest non-system, non-recent messages first.
        4. Never split tool-call pairs (assistant+tool_calls → tool result).
        5. Insert a compact summary marker after the system prompt.
        """
        if len(messages) <= 1:
            return messages

        total = self.estimate_tokens(messages)
        if total <= self.context_budget:
            return messages

        # Separate system prompt from the rest
        system = messages[0]
        rest = messages[1:]

        # Identify the recent window we want to preserve
        recent_count = min(len(rest), _MIN_RECENT)
        recent = rest[-recent_count:]
        droppable = rest[:-recent_count] if recent_count < len(rest) else []

        if not droppable:
            # Nothing we can drop — return as-is (recent window is all we have)
            return messages

        # Drop from oldest first, respecting tool-call pairs
        kept: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []

        # Build a set of indices in droppable that form tool-call pairs
        # so we never split them.
        i = 0
        drop_groups: list[list[int]] = []  # groups of indices to drop together
        while i < len(droppable):
            msg = droppable[i]
            if (
                msg.get("tool_calls")
                and i + 1 < len(droppable)
                and droppable[i + 1].get("role") == "tool"
            ):
                drop_groups.append([i, i + 1])
                i += 2
            else:
                drop_groups.append([i])
                i += 1

        # Calculate how much we need to free
        system_tokens = self._msg_tokens(system)
        recent_tokens = sum(self._msg_tokens(m) for m in recent)
        target = self.context_budget - system_tokens - recent_tokens
        # Reserve space for the summary marker we'll insert
        summary_reserve = 200

        # Accumulate droppable groups from oldest, dropping until we fit
        droppable_tokens = sum(self._msg_tokens(m) for m in droppable)
        kept_tokens = droppable_tokens

        for group in drop_groups:
            if kept_tokens <= target - summary_reserve:
                # We've freed enough — keep the rest
                for idx in group:
                    kept.append(droppable[idx])
            else:
                # Drop this group
                group_tokens = sum(self._msg_tokens(droppable[idx]) for idx in group)
                kept_tokens -= group_tokens
                for idx in group:
                    dropped.append(droppable[idx])

        n_dropped = len(dropped)
        if n_dropped == 0:
            return messages

        self.trimmed_count += n_dropped
        logger.info("Context trimming: dropped %d messages to fit budget", n_dropped)

        # Build summary marker
        summary_text = self._build_summary(dropped)
        summary_msg: dict[str, Any] = {
            "role": "system",
            "content": (
                f"[Context note: {n_dropped} earlier messages were trimmed"
                " to fit the context window.\n"
                f"{summary_text}\n"
                "Recent context follows.]"
            ),
        }

        return [system, summary_msg] + kept + recent

    # ------------------------------------------------------------------
    # Extractive summary
    # ------------------------------------------------------------------

    def _build_summary(self, dropped_messages: list[dict[str, Any]]) -> str:
        """Build a compact extractive summary of dropped messages."""
        facts: list[str] = []
        for msg in dropped_messages:
            role = msg.get("role", "")
            if role == "user":
                content = msg.get("content", "")[:100]
                if content:
                    facts.append(f"User asked: {content}")
            elif msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args_str = func.get("arguments", "")
                    if name == "execute_shell":
                        try:
                            cmd = json.loads(args_str).get("command", "")[:80]
                            facts.append(f"Ran: {cmd}")
                        except (json.JSONDecodeError, AttributeError):
                            facts.append(f"Called: {name}")
                    else:
                        facts.append(f"Called: {name}")
            elif role == "tool":
                content = msg.get("content", "")
                for line in content.split("\n"):
                    if line.startswith("Exit code:"):
                        facts.append(line)
                        break

        summary = "\n".join(f"- {f}" for f in facts[:15])
        if len(summary) > 500:
            summary = summary[:500] + "..."
        return summary
