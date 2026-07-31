"""Command risk classification — pattern-based, deterministic, fast."""

from __future__ import annotations

import logging
import re
from enum import Enum

from natshell.config import SafetyConfig
from natshell.safety.command_split import split_commands

logger = logging.getLogger(__name__)


class Risk(Enum):
    SAFE = "safe"
    CONFIRM = "confirm"
    BLOCKED = "blocked"

# Paths that should require user confirmation before read_file accesses them
_SENSITIVE_PATH_PATTERNS = [
    "/.ssh/",
    "/id_rsa",
    "/id_ed25519",
    "/etc/shadow",
    "/etc/sudoers",
    "/proc/self/environ",
    ".env",
    "/.aws/credentials",
    "/.kube/config",
    "/.docker/config.json",
]


_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_NUMERIC_ARG_RE = re.compile(r"^-?\d+(?:\.\d+)?[smhd]?$")

# Commands whose own argument is another command, so the binary that matters is
# not the first token.
_WRAPPER_COMMANDS = frozenset(
    {
        "builtin",
        "busybox",
        "command",
        "doas",
        "env",
        "eval",
        "exec",
        "ionice",
        "nice",
        "nohup",
        "setsid",
        "stdbuf",
        "sudo",
        "time",
        "timeout",
        "xargs",
    }
)

# Wrappers that take a bare number of their own (``timeout 5 …``, ``nice 10 …``)
_NUMERIC_ARG_WRAPPERS = frozenset({"ionice", "nice", "time", "timeout"})


def _basename(token: str) -> str:
    """Strip any directory prefix, POSIX or Windows."""
    return token.rpartition("/")[2].rpartition("\\")[2]


def _normalize_invocation(command: str) -> str:
    """Return *command* reduced to the binary it actually runs, plus its arguments.

    Every shipped pattern is anchored with ``^`` on a bare command name, so
    anything that pushes the binary off the front of the string evades all of
    them at once.  ``/bin/rm -rf x``, ``LC_ALL=C rm -rf x``, ``command rm -rf x``
    and ``nohup rm -rf x`` all normalize to ``rm -rf x``.

    Returns the input unchanged when there is nothing to strip.  Callers match
    against both forms, so an imperfect normalization can only add coverage,
    never remove it.
    """
    tokens = command.split()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if _ENV_ASSIGNMENT_RE.match(token):
            index += 1
            continue
        name = _basename(token)
        if name in _WRAPPER_COMMANDS:
            index += 1
            # Skip the wrapper's own options so the wrapped binary lands first.
            while index < len(tokens) and (
                tokens[index].startswith("-")
                or (name in _NUMERIC_ARG_WRAPPERS and _NUMERIC_ARG_RE.match(tokens[index]))
            ):
                index += 1
            continue
        break

    if index >= len(tokens):
        return command
    remaining = tokens[index:]
    return " ".join([_basename(remaining[0]), *remaining[1:]])


def _is_agents_md(path: str) -> bool:
    """Return True if *path* is a working memory agents.md file."""
    return (
        path.endswith(".natshell/agents.md")
        or path.endswith(".config/natshell/agents.md")
    )


class SafetyClassifier:
    """Classify tool calls by risk level using regex patterns."""

    def __init__(self, config: SafetyConfig) -> None:
        self.mode = config.mode
        # MULTILINE so that the '^' every shipped pattern starts with anchors to
        # each line of a multi-line command, not just the first.  Sub-commands are
        # split out below as well; this is the belt to that pair of braces.
        self._confirm_patterns = [re.compile(p, re.MULTILINE) for p in config.always_confirm]
        self._blocked_patterns = [re.compile(p, re.MULTILINE) for p in config.blocked]

    def classify_command(self, command: str) -> Risk:
        """Classify a shell command string by risk level.

        Splits on shell operators (&&, ||, ;, &, |, (, ) and newlines) and
        returns the highest risk found across the whole command and each
        sub-command independently.
        Also flags subshells and backtick expansions as CONFIRM.

        Every blocked check runs before any CONFIRM can be returned.  Returning
        the first CONFIRM match instead meant a blocked sub-command later in the
        string was never examined -- ``sudo apt update && <blocked>`` reported
        CONFIRM, and CONFIRM is downgraded to SAFE in danger mode while BLOCKED
        is not.
        """
        sub_commands = split_commands(command)

        # The full command is checked as well as its parts, because some
        # patterns (fork bombs, pipe-based patterns) span operators.
        if self._matches(self._blocked_patterns, command):
            logger.warning(f"BLOCKED command: {command}")
            return Risk.BLOCKED
        for sub in sub_commands:
            if self._matches(self._blocked_patterns, sub):
                logger.warning(f"BLOCKED command: {command}")
                return Risk.BLOCKED

        if self._matches(self._confirm_patterns, command):
            return Risk.CONFIRM

        # Flag commands using subshells or backtick expansion
        if re.search(r"`[^`]+`|\$\([^)]+\)", command):
            return Risk.CONFIRM

        for sub in sub_commands:
            risk = self._classify_single(sub)
            if risk is not Risk.SAFE:
                return risk
        return Risk.SAFE

    @staticmethod
    def _matches(patterns: list[re.Pattern[str]], text: str) -> bool:
        """True if any pattern matches *text* as written, or once normalized.

        Both forms are tried so that normalization can only widen coverage: a
        command the patterns already catch stays caught even if
        _normalize_invocation mishandles it.
        """
        if any(pattern.search(text) for pattern in patterns):
            return True
        normalized = _normalize_invocation(text)
        if normalized == text:
            return False
        return any(pattern.search(normalized) for pattern in patterns)

    def _classify_single(self, command: str) -> Risk:
        """Classify a single command (no chaining operators)."""
        # Check blocked first
        if self._matches(self._blocked_patterns, command):
            logger.warning(f"BLOCKED command: {command}")
            return Risk.BLOCKED

        # Check confirmation-required patterns
        if self._matches(self._confirm_patterns, command):
            return Risk.CONFIRM

        # Heuristic: sudo always requires confirmation.  Checked on the
        # normalized form too, so /usr/bin/sudo and `env sudo` are caught.
        if _normalize_invocation(command).strip().startswith("sudo "):
            return Risk.CONFIRM

        # Heuristic: redirecting to system paths
        if re.search(r">\s*/(?:etc|boot|usr|var/lib)/", command):
            return Risk.CONFIRM

        return Risk.SAFE

    def classify_tool_call(self, tool_name: str, arguments: dict) -> Risk:
        """Classify any tool call by risk level."""
        if tool_name == "execute_shell":
            command = arguments.get("command", "")
            risk = self.classify_command(command)
            # In danger mode, downgrade CONFIRM to SAFE (but not BLOCKED)
            if self.mode == "danger" and risk == Risk.CONFIRM:
                return Risk.SAFE
            return risk

        if tool_name == "write_file":
            path = arguments.get("path", "")
            if _is_agents_md(path):
                return Risk.SAFE
            for pattern in _SENSITIVE_PATH_PATTERNS:
                if pattern in path:
                    return Risk.CONFIRM
            if self.mode == "danger":
                return Risk.SAFE
            return Risk.CONFIRM

        if tool_name == "edit_file":
            # Always confirm edits; also check sensitive paths
            path = arguments.get("path", "")
            if _is_agents_md(path):
                return Risk.SAFE
            for pattern in _SENSITIVE_PATH_PATTERNS:
                if pattern in path:
                    return Risk.CONFIRM
            if self.mode == "danger":
                return Risk.SAFE
            return Risk.CONFIRM

        if tool_name == "run_code":
            if self.mode == "danger":
                return Risk.SAFE
            return Risk.CONFIRM

        if tool_name == "read_file":
            path = arguments.get("path", "")
            for pattern in _SENSITIVE_PATH_PATTERNS:
                if pattern in path:
                    return Risk.CONFIRM
            return Risk.SAFE

        if tool_name == "git_tool":
            operation = arguments.get("operation", "")
            # Read-only operations are safe
            if operation in ("status", "diff", "log", "branch"):
                return Risk.SAFE
            # Mutating operations require confirmation
            if operation in ("commit", "stash"):
                if self.mode == "danger":
                    return Risk.SAFE
                return Risk.CONFIRM
            # Unknown operation — let the tool handle the error, but confirm
            return Risk.CONFIRM

        if tool_name == "fetch_url":
            return Risk.SAFE

        # list_directory, search_files are always safe
        return Risk.SAFE
