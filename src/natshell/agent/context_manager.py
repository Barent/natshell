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

# Per-ref char cap for the inline ref line emitted in the summary.
# Paths add ~60-100 chars, so the cap is generous enough to keep the
# full absolute path visible without truncating the middle of it.
_REF_MAX_CHARS = 260

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
    ) -> None:
        self.context_budget = context_budget
        self._initial_budget = context_budget
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

        # Build summary marker
        summary_text = self.build_summary(dropped)
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
    # Retrieval-augmented summary with plain-file chunk references
    # ------------------------------------------------------------------

    def build_summary_with_refs(
        self,
        dropped_messages: list[dict[str, Any]],
        write_fn: Callable[[str, str], Any],
        session_id: str,
    ) -> tuple[str, list[Any]]:
        """Build a summary that writes bulky tool results to plain files and
        emits their absolute paths in place of the content.

        ``write_fn`` has the signature ``(session_id, content) -> Path | None``
        matching ``natshell.agent.memory_files.write_chunk``.  Dependency
        injection lets ContextManager stay testable without touching disk:
        pass any callable returning a Path-like or None.

        Returns ``(summary_text, list_of_paths)``.  Paths are whatever
        ``write_fn`` returned (typically ``pathlib.Path``).

        Pair-aware: assistant ``tool_calls`` messages are processed
        together with their following ``role="tool"`` result so the path is
        attributed to the originating tool call, not the bare result.
        """
        files_changed: list[str] = []
        ref_lines: list[str] = []
        plain_lines: list[str] = []
        stored_paths: list[Any] = []

        i = 0
        while i < len(dropped_messages):
            msg = dropped_messages[i]
            role = msg.get("role", "")

            if role == "user":
                content = (msg.get("content") or "").strip()
                if content:
                    snippet = content[:100]
                    plain_lines.append(f"User asked: {snippet}")
                # Long user messages also get captured to disk so the
                # agent can re-read them verbatim if needed.
                if len(content) > 100:
                    path = write_fn(session_id, content)
                    if path is not None:
                        stored_paths.append(path)
                        ref_lines.append(
                            self._cap_ref(
                                f"user_message ({len(content)} chars) → {path}"
                            )
                        )
                i += 1
                continue

            tool_calls = msg.get("tool_calls") or []
            has_paired_result = (
                tool_calls
                and i + 1 < len(dropped_messages)
                and dropped_messages[i + 1].get("role") == "tool"
            )
            if has_paired_result:
                # Pair: assistant(tool_calls=[...]) → tool result.
                tc = tool_calls[0]
                func = tc.get("function", {})
                name = func.get("name", "")
                args_str = func.get("arguments", "") or ""
                tool_msg = dropped_messages[i + 1]
                result_content = tool_msg.get("content") or ""

                # Track file changes for the legacy section.
                if name in ("write_file", "edit_file"):
                    try:
                        path_str = json.loads(args_str).get("path", "")
                        if path_str:
                            action = "created" if name == "write_file" else "edited"
                            entry = f"{action}: {path_str}"
                            if entry not in files_changed:
                                files_changed.append(entry)
                    except (json.JSONDecodeError, AttributeError):
                        pass

                # Write the bulky tool result to a plain file and emit
                # its absolute path.
                path = write_fn(session_id, result_content)
                if path is not None:
                    stored_paths.append(path)
                    preview = self._tool_call_preview(name, args_str)
                    n_lines = result_content.count("\n") + (1 if result_content else 0)
                    exit_code = self._extract_exit_code(result_content)
                    suffix = f" exit={exit_code}" if exit_code is not None else ""
                    ref_lines.append(
                        self._cap_ref(
                            f"{name} {preview}{suffix} {n_lines}L → {path}"
                        )
                    )
                else:
                    # Fall back to a plain action line if the write failed.
                    plain_lines.append(f"Called: {name}")
                i += 2
                continue

            if tool_calls:
                # Lone tool_calls without a paired result (rare).
                tc = tool_calls[0]
                func = tc.get("function", {})
                name = func.get("name", "")
                plain_lines.append(f"Called: {name}")
                i += 1
                continue

            if role == "tool":
                # Orphan tool result (no preceding assistant message).
                content = msg.get("content") or ""
                path = write_fn(session_id, content)
                if path is not None:
                    stored_paths.append(path)
                    n_lines = content.count("\n") + (1 if content else 0)
                    ref_lines.append(
                        self._cap_ref(f"tool_result {n_lines}L → {path}")
                    )
                i += 1
                continue

            # assistant text content (model reasoning) — store + reference.
            if role == "assistant":
                content = msg.get("content") or ""
                if content.strip():
                    path = write_fn(session_id, content)
                    if path is not None:
                        stored_paths.append(path)
                        ref_lines.append(
                            self._cap_ref(
                                f"assistant_text ({len(content)} chars) → {path}"
                            )
                        )
            i += 1

        parts: list[str] = []
        if files_changed:
            parts.append(
                "Files changed:\n" + "\n".join(f"- {f}" for f in files_changed)
            )
        if ref_lines:
            parts.append(
                "Stored chunks (use read_file on the paths to recall):\n"
                + "\n".join(f"- {r}" for r in ref_lines)
            )
        if plain_lines:
            parts.append(
                "Notes:\n" + "\n".join(f"- {p}" for p in plain_lines[:8])
            )
        if len(stored_paths) > 3:
            parts.append(
                "To inspect any chunk above, call read_file(path=<path>). "
                "To search across all chunks, call "
                "search_files(path=<session memory dir>, pattern=<keyword>)."
            )

        summary = "\n".join(parts)
        return summary, stored_paths

    @staticmethod
    def _cap_ref(line: str) -> str:
        if len(line) <= _REF_MAX_CHARS:
            return line
        return line[: _REF_MAX_CHARS - 3] + "..."

    @staticmethod
    def _tool_call_preview(name: str, args_str: str) -> str:
        """Render a short, safe preview of a tool call for the summary line."""
        if not args_str:
            return ""
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(args, dict):
            return ""
        if name == "execute_shell" and "command" in args:
            cmd = str(args["command"])[:60]
            return f'"{cmd}"'
        if name in ("read_file", "write_file", "edit_file") and "path" in args:
            return f'"{args["path"]}"'
        if name == "search_files" and "pattern" in args:
            return f'"{args["pattern"]}"'
        if name == "fetch_url" and "url" in args:
            return f'"{str(args["url"])[:60]}"'
        # Generic: first non-empty arg value, truncated
        for v in args.values():
            if isinstance(v, (str, int, float)) and str(v):
                return f'"{str(v)[:60]}"'
        return ""

    @staticmethod
    def _extract_exit_code(content: str) -> int | None:
        for line in content.splitlines()[:5]:
            if line.startswith("Exit code:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    return None
        return None
