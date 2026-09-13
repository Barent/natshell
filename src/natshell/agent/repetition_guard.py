"""Repetition and edit-failure guarding for the agent's tool dispatch loop.

Extracted from :mod:`natshell.agent.loop` (which previously carried six
separate inline detectors plus edit-failure escalation as a wall of state and
string-building inside ``handle_user_message``).  Behaviour is preserved
exactly — same thresholds, same warning texts, same stop points; the loop
simply asks ``observe()`` for a result.

Design: one owner of the counter state, one entry point, one output shape.

* :meth:`RepetitionGuard.observe` returns an :class:`Observation` whose
  ``suffix`` is appended to the tool result given back to the model and whose
  ``stop`` flag tells the loop to end the current tool-dispatch loop (the
  "hard stop" paths that previously ``break``'d mid-method).
* Edit-failure / success bookkeeping lives here too, because the completion
  guard and the escalating edit warnings depend on the same counters.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DUP_WARN_THRESHOLD = 3
_DUP_ABORT_THRESHOLD = 5

_FAM_WARN = 4
_FAM_CRITICAL = 7
_FAM_HARD_STOP = 12

_SIM_WARN = 3
_SIM_ABORT = 5


@dataclass
class Observation:
    """Result of feeding one executed tool call to the guard.

    Attributes:
        suffix: Text (possibly empty) to append to the tool result content
            before it is handed back to the model.
        stop: If True, the dispatch loop must end immediately (the exchange
            has already been accounted for by the caller).
    """

    suffix: str = ""
    stop: bool = False


class RepetitionGuard:
    """Detects and escalates model repetition patterns across a run.

    All per-run state is reset by :meth:`reset` at the start of a run (and by
    ``clear_history``), mirroring the reset that previously lived at the top
    of ``handle_user_message``.
    """

    def __init__(self) -> None:
        self._edit_failures = 0
        self._edit_successes = 0
        self._completion_warning_sent = False
        # Repetitive read detection
        self._read_counts: dict[str, int] = {}
        # Repetitive URL fetch detection (cumulative, not just consecutive)
        self._fetch_url_counts: dict[str, int] = {}
        # Command-family repetition detection (e.g., 10+ `du` calls)
        self._cmd_family_counts: dict[str, int] = {}
        # Duplicate tool call detection
        self._last_tool_key: str = ""
        self._consecutive_dupes = 0
        # Similar-command detection (strips flags/pipes to find semantic dupes)
        self._similar_cmd_key: str = ""
        self._consecutive_similar = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all per-run counters (start of a new run / cleared history)."""
        self._edit_failures = 0
        self._edit_successes = 0
        self._completion_warning_sent = False
        self._read_counts = {}
        self._fetch_url_counts = {}
        self._cmd_family_counts = {}
        self._last_tool_key = ""
        self._consecutive_dupes = 0
        self._similar_cmd_key = ""
        self._consecutive_similar = 0

    # ------------------------------------------------------------------
    # Bookkeeping (called by the loop around successful/mutating calls)
    # ------------------------------------------------------------------

    def register_outcome(self, tool_name: str, exit_code: int) -> None:
        """Record an edit_file/write_file outcome for the completion guard."""
        if tool_name == "edit_file":
            if exit_code != 0:
                self._edit_failures += 1
            else:
                self._edit_successes += 1
        elif tool_name == "write_file" and exit_code == 0:
            self._edit_successes += 1

    def note_successful_write(self, path: str) -> None:
        """Clear the read-repeat counter when a write/edit succeeded on a path."""
        resolved = self._resolve_path(path)
        self._read_counts.pop(resolved, None)

    @property
    def completion_guard_due(self) -> bool:
        """True once when all edits failed and not a single one succeeded."""
        return (
            self._edit_failures > 0
            and self._edit_successes == 0
            and not self._completion_warning_sent
        )

    def mark_completion_guard_sent(self) -> None:
        self._completion_warning_sent = True

    @property
    def edit_failure_count(self) -> int:
        return self._edit_failures

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_path(path: str) -> str:
        """Canonicalize a path for tracking (mirrors file_tracker)."""
        return str(Path(path).expanduser().resolve())

    @staticmethod
    def _normalize_shell_cmd(cmd: str) -> str:
        """Strip flags and pipes to detect semantically duplicate commands.

        ``grep -o "Check" file | head -1`` and ``grep "Check" file``
        both normalise to ``grep Check file``, so the similar-command
        detector can spot the pattern even when the model tweaks flags.
        """
        base = cmd.split("|")[0].strip()
        try:
            tokens = shlex.split(base)
        except ValueError:
            tokens = base.split()
        core = [t for t in tokens if not t.startswith("-")]
        return " ".join(core)

    def observe(
        self,
        name: str,
        arguments: dict[str, Any],
        exit_code: int,
        *,
        record: bool = True,
    ) -> Observation:
        """Feed one executed tool call; return the suffix + stop signal.

        ``record=False`` skips the failure/success counters and the
        read-count reset entirely — used for a redundant re-evaluation of a
        result that was already recorded once.
        """
        import json

        # Edit failure/success bookkeeping must happen BEFORE the edit
        # warning check below, so the count reflects this call (identical
        # to the ordering the loop previously had inline).
        if record:
            self.register_outcome(name, exit_code)

        suffixes: list[str] = []
        stop = False

        # Duplicate tool call detection — catch infinite retry loops
        tool_key = (
            f"{name}:{json.dumps(arguments, sort_keys=True)}"
        )
        if tool_key == self._last_tool_key:
            self._consecutive_dupes += 1
        else:
            self._last_tool_key = tool_key
            self._consecutive_dupes = 1

        if self._consecutive_dupes >= _DUP_ABORT_THRESHOLD:
            suffixes.append(
                f"\n\n\u26a0 CRITICAL: You have called {name} "
                f"with identical arguments {self._consecutive_dupes} "
                "times in a row. The output will not change. "
                "STOP making this tool call. Complete the task using "
                "your existing knowledge and information already gathered."
            )
            # Reset counter so the LLM can use other tools freely
            self._last_tool_key = ""
            self._consecutive_dupes = 0
            stop = True
        elif self._consecutive_dupes >= _DUP_WARN_THRESHOLD:
            suffixes.append(
                f"\n\n\u26a0 You have called {name} with "
                f"identical arguments {self._consecutive_dupes} times "
                "in a row and gotten the same result each time. "
                "Try a DIFFERENT approach — change the arguments, "
                "use a different tool, or fix the underlying issue "
                "before retrying."
            )

        if stop:
            return Observation(suffix="".join(suffixes), stop=True)

        # Repetitive read detection
        if name == "read_file":
            read_path = arguments.get("path", "")
            if read_path:
                resolved = self._resolve_path(read_path)
                self._read_counts[resolved] = (
                    self._read_counts.get(resolved, 0) + 1
                )
                count = self._read_counts[resolved]
                if count >= 3:
                    suffixes.append(
                        f"\n\n\u26a0 You have read this file {count} times "
                        "without modifying it. Stop re-reading and use the "
                        "information you already have to make changes. "
                        "If edit_file is failing, use write_file instead."
                    )

        # Repetitive URL fetch detection
        # (cumulative across all calls, not just consecutive)
        if name == "fetch_url":
            url = arguments.get("url", "")
            if url:
                self._fetch_url_counts[url] = (
                    self._fetch_url_counts.get(url, 0) + 1
                )
                count = self._fetch_url_counts[url]
                if count == 2:
                    suffixes.append(
                        f"\n\n\u26a0 You have fetched this URL {count} times. "
                        "The content will not change. Do NOT fetch it again — "
                        "use the information already in your context."
                    )
                elif count >= 3:
                    suffixes.append(
                        f"\n\n\u26a0 CRITICAL: You have fetched this URL "
                        f"{count} times. "
                        "STOP fetching it. The result is already in your"
                        " conversation history. Use your existing knowledge"
                        " to complete the task."
                    )

        # Command-family repetition detection for execute_shell
        if name == "execute_shell":
            cmd = arguments.get("command", "")
            # Extract the first token as the command family
            words = cmd.strip().split()
            family = words[0] if words else ""
            # Normalise common prefixes (sudo X → X)
            if family == "sudo" and len(words) > 1:
                family = words[1]
            if family:
                self._cmd_family_counts[family] = (
                    self._cmd_family_counts.get(family, 0) + 1
                )
                fam_count = self._cmd_family_counts[family]
                if fam_count >= _FAM_HARD_STOP:
                    suffixes.append(
                        f"\n\n\u26a0 HARD STOP: You have run"
                        f" `{family}` {fam_count} times."
                        " You MUST use a completely"
                        " different approach or tool."
                        " Further `{family}` calls"
                        " are blocked."
                    )
                    stop = True
                elif fam_count >= _FAM_CRITICAL:
                    suffixes.append(
                        f"\n\n\u26a0 CRITICAL: You have run"
                        f" `{family}` {fam_count} times in"
                        " this session. STOP running more"
                        f" `{family}` commands. Synthesize"
                        " your findings from the output"
                        " already gathered and give the"
                        " user a complete answer NOW."
                    )
                elif fam_count >= _FAM_WARN:
                    suffixes.append(
                        f"\n\n\u26a0 You have run `{family}`"
                        f" {fam_count} times. Consolidate"
                        " your findings and answer with"
                        " what you have. Avoid further"
                        f" `{family}` calls unless"
                        " absolutely necessary."
                    )

            # Similar-command detection — catches near-duplicate
            # commands that differ only in flags or pipes
            # (e.g. grep -o vs grep -n on the same pattern/file)
            norm_key = self._normalize_shell_cmd(cmd)
            if norm_key and len(norm_key.split()) > 1:
                if norm_key == self._similar_cmd_key:
                    self._consecutive_similar += 1
                else:
                    self._similar_cmd_key = norm_key
                    self._consecutive_similar = 1

                if self._consecutive_similar >= _SIM_ABORT:
                    suffixes.append(
                        f"\n\n\u26a0 CRITICAL: You have run"
                        f" {self._consecutive_similar}"
                        " near-identical commands in a"
                        " row (same target, different"
                        " flags). The result will not"
                        " change. STOP and try a"
                        " completely different approach."
                    )
                    self._similar_cmd_key = ""
                    self._consecutive_similar = 0
                    stop = True
                elif self._consecutive_similar >= _SIM_WARN:
                    suffixes.append(
                        f"\n\n\u26a0 You have run"
                        f" {self._consecutive_similar}"
                        " near-identical commands."
                        " Changing flags or adding"
                        " pipes will not produce"
                        " different results. Try a"
                        " different approach."
                    )

        if stop:
            return Observation(suffix="".join(suffixes), stop=True)

        # Reset read count when write/edit succeeds on a path
        if name in ("edit_file", "write_file") and exit_code == 0:
            write_path = arguments.get("path", "")
            if write_path:
                self.note_successful_write(write_path)

        # Escalating warnings on repeated edit failures
        if name == "edit_file" and exit_code != 0:
            if self._edit_failures >= 3:
                suffixes.append(
                    "\n\n\u26a0 REPEATED EDIT FAILURES (3+). "
                    "STOP using edit_file for this file. "
                    "Use write_file to rewrite the entire file instead."
                )
            elif self._edit_failures >= 2:
                suffixes.append(
                    "\n\n\u26a0 Multiple edit failures. Try: "
                    "(1) use the closest match shown above as your old_text, or "
                    "(2) use write_file to rewrite the entire file instead."
                )

        return Observation(suffix="".join(suffixes), stop=False)
