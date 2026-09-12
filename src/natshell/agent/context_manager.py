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
    - Approximate: ``len(text) // 4`` (BPE average for English text)
    """

    def __init__(
        self,
        context_budget: int,
        tokenizer_fn: Callable[[str], int] | None = None,
        summarizer: Callable[[list[dict[str, Any]]], str] | None = None,
    ) -> None:
        self.context_budget = context_budget
        self._initial_budget = context_budget
        self._tokenizer_fn = tokenizer_fn
        # Optional compact summarizer (R2-5 LLM tier).  Called with the
        # dropped messages; a non-empty return replaces the extractive
        # summary.  Defaults to None → pure extractive (historical).
        self.summarizer: Callable[[list[dict[str, Any]]], str] | None = summarizer
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
        # Fallback: ~4 chars per token (BPE average for English text)
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # Budget calibration from actual API token counts
    # ------------------------------------------------------------------

    def calibrate_from_actual(self, estimated_tokens: int, actual_tokens: int) -> None:
        """Shrink context_budget if our estimate significantly underestimates reality.

        Called after each successful inference with the estimated token count
        (from our ``len(text)//4`` heuristic) and the actual prompt_tokens
        reported by the API.  When the actual count exceeds the estimate by
        more than 15%, the budget is shrunk proportionally (floor 1024).
        """
        if estimated_tokens <= 0 or actual_tokens <= 0:
            return
        ratio = actual_tokens / estimated_tokens
        if ratio > 1.15:
            new_budget = int(self.context_budget / ratio)
            floor = max(1024, int(self._initial_budget * 0.4))
            new_budget = max(new_budget, floor)
            if new_budget != self.context_budget:
                logger.info(
                    "Calibrating context budget %d → %d "
                    "(estimate %d vs actual %d tokens, ratio %.2f)",
                    self.context_budget,
                    new_budget,
                    estimated_tokens,
                    actual_tokens,
                    ratio,
                )
                self.context_budget = new_budget

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

        # Build summary marker (shared marker shape — see context_marker)
        summary_msg: dict[str, Any] = self.context_marker(
            dropped,
            f"Context note: {n_dropped} earlier messages were trimmed to fit the context window.",
        )

        return [system, summary_msg] + kept + recent

    # ------------------------------------------------------------------
    # Compaction summary marker (shared shape)
    # ------------------------------------------------------------------

    def context_marker(
        self,
        dropped_messages: list[dict[str, Any]],
        preamble: str = "Context compacted.",
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Build the system-role summary message replacing dropped messages.

        One shared shape for both compaction paths (budget trimming and
        manual/forced compaction): ``[<preamble>\\n<summary>\\nRecent
        context follows.]``.  ``summary`` defaults to :meth:`summarize`;
        pass it when the caller already resolved it (e.g. to reuse the
        same text in stats) so the summarizer is invoked exactly once.
        """
        if summary is None:
            summary = self.summarize(dropped_messages)
        return {
            "role": "system",
            "content": f"[{preamble}\n{summary}\nRecent context follows.]",
        }

    def summarize(self, dropped_messages: list[dict[str, Any]]) -> str:
        """Choose the summary text for a compaction marker (R2-5 seam).

        ``self.summarizer`` (if configured) is tried first; a failed or
        empty result falls back to :meth:`build_summary` — exactly the
        historical extractive behaviour.
        """
        if self.summarizer is not None:
            try:
                custom = self.summarizer(dropped_messages)
            except Exception:
                logger.warning(
                    "Compaction summarizer raised — using extractive summary",
                    exc_info=True,
                )
                custom = None
            if custom and str(custom).strip():
                return str(custom)
        return self.build_summary(dropped_messages)

    # ------------------------------------------------------------------
    # Extractive summary
    # ------------------------------------------------------------------

    def build_summary(self, dropped_messages: list[dict[str, Any]]) -> str:
        """Build a compact extractive summary of dropped messages.

        Prioritizes file change tracking so the model knows what it
        already created/modified, even after trimming.
        """
        files_changed: list[str] = []
        actions: list[str] = []
        for msg in dropped_messages:
            role = msg.get("role", "")
            if role == "user":
                content = msg.get("content", "")[:100]
                if content:
                    actions.append(f"User asked: {content}")
            elif msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args_str = func.get("arguments", "")
                    if name in ("write_file", "edit_file"):
                        try:
                            path = json.loads(args_str).get("path", "")
                            if path:
                                action = "created" if name == "write_file" else "edited"
                                entry = f"{action}: {path}"
                                if entry not in files_changed:
                                    files_changed.append(entry)
                        except (json.JSONDecodeError, AttributeError):
                            actions.append(f"Called: {name}")
                    elif name == "execute_shell":
                        try:
                            cmd = json.loads(args_str).get("command", "")[:80]
                            actions.append(f"Ran: {cmd}")
                        except (json.JSONDecodeError, AttributeError):
                            actions.append(f"Called: {name}")
                    elif name == "read_file":
                        try:
                            path = json.loads(args_str).get("path", "")
                            if path:
                                actions.append(f"Read: {path}")
                        except (json.JSONDecodeError, AttributeError):
                            actions.append(f"Called: {name}")
                    else:
                        actions.append(f"Called: {name}")
            elif role == "tool":
                content = msg.get("content", "")
                for line in content.split("\n"):
                    if line.startswith("Exit code:"):
                        actions.append(line)
                        break

        parts: list[str] = []
        if files_changed:
            parts.append("Files changed:\n" + "\n".join(f"- {f}" for f in files_changed))
        action_text = "\n".join(f"- {a}" for a in actions[:12])
        if action_text:
            parts.append(f"Actions:\n{action_text}")
        summary = "\n".join(parts)
        if len(summary) > 800:
            summary = summary[:800] + "..."
        return summary

    # ------------------------------------------------------------------
    # Artifact elision (cheap in-place compression, no summarizer)
    # ------------------------------------------------------------------

    #: Most-recent non-system messages to leave untouched by
    #: :meth:`compress_artifacts`. Mirrors the "3 tool exchanges" intuition
    #: that used to live in ``agent.loop._compress_old_messages``.
    COMPRESS_PRESERVE_RECENT = 6

    def compress_artifacts(
        self, messages: list[dict[str, Any]]
    ) -> bool:
        """Elide big write_file contents and truncate long tool results
        for the *older* half of the conversation.

        Unlike :meth:`trim_messages`, this does **not** drop messages or
        insert a summary — it just rewrites the most expensive bytes in
        place.  Safe (and cheap) to run frequently because the model has
        already processed the affected messages and the elided content is
        still available if the model re-reads the file.

        Returns True if any message was modified, False otherwise.
        """
        import json as _json

        if len(messages) <= self.COMPRESS_PRESERVE_RECENT + 1:
            return False

        cutoff = len(messages) - self.COMPRESS_PRESERVE_RECENT
        changed = False

        for i in range(1, cutoff):  # skip system prompt
            msg = messages[i]

            # Compress write_file arguments (elide full file content)
            if msg.get("tool_calls"):
                new_tool_calls = []
                modified_in_msg = False
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args_str = func.get("arguments", "")
                    if name == "write_file" and len(args_str) > 300:
                        try:
                            args = _json.loads(args_str)
                            content = args.get("content", "")
                            if len(content) > 100:
                                args["content"] = f"[{len(content)} chars elided]"
                                func["arguments"] = _json.dumps(args)
                                modified_in_msg = True
                        except (_json.JSONDecodeError, TypeError):
                            pass
                    new_tool_calls.append(tc)
                if modified_in_msg:
                    msg["tool_calls"] = new_tool_calls
                    changed = True

            # Compress long tool results
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if len(content) > 800:
                    lines = content.split("\n")
                    if len(lines) > 8:
                        head = "\n".join(lines[:4])
                        tail = "\n".join(lines[-3:])
                        msg["content"] = (
                            f"{head}\n"
                            f"... [{len(lines) - 7} lines elided] ...\n"
                            f"{tail}"
                        )
                        changed = True

        return changed
