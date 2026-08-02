"""Command risk classification — pattern-based, deterministic, fast."""

from __future__ import annotations

import logging
import os
import re
from enum import Enum
from pathlib import Path

from natshell.config import SafetyConfig
from natshell.safety.command_split import split_commands

logger = logging.getLogger(__name__)


class Risk(Enum):
    SAFE = "safe"
    CONFIRM = "confirm"
    BLOCKED = "blocked"


# Paths that should require user confirmation before a tool reads or searches them
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

# Tools that cannot change state and so do not need a confirmation dialog.
# Everything not listed here — and not given an explicit branch in
# classify_tool_call — falls through to CONFIRM rather than SAFE, so a newly
# registered tool is not auto-approved simply because nobody classified it.
_READ_ONLY_TOOLS = frozenset(
    {
        "list_directory",
        "natshell_help",
        "skill",
        "fetch_url",
        "kiwix_search",
    }
)

# Wrappers that take a command as their argument.  Stripping them exposes the
# real binary to the patterns.  Matching is additive (raw *and* normalized are
# both tested), so an over-eager strip can only ever add a confirmation — it
# can never cause a command to be classified more leniently.
_WRAPPER_COMMANDS = frozenset(
    {
        "command",
        "builtin",
        "exec",
        "nohup",
        "env",
        "time",
        "stdbuf",
        "nice",
        "ionice",
        "setsid",
        "eval",
    }
)

# Leading VAR=value assignments, e.g. `LC_ALL=C rm -rf /`
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|[^\s]*)\s+")


def _normalize_invocation(command: str) -> str:
    """Strip env assignments, wrapper commands and the executable's path.

    ``/bin/rm -rf /`` and ``LC_ALL=C command rm -rf /`` both normalize to
    ``rm -rf /``, so a pattern anchored on the binary name still matches.
    Without this a single path prefix defeats every pattern at once,
    including the BLOCKED tier that is supposed to survive danger mode.
    """
    text = command.strip()
    # Alternate between stripping env assignments and wrapper commands until
    # neither applies -- `LC_ALL=C nohup env FOO=1 rm ...` needs several passes.
    for _ in range(10):
        stripped = _ENV_ASSIGN_RE.sub("", text, count=1)
        if stripped != text:
            text = stripped.lstrip()
            continue

        parts = text.split(None, 1)
        if len(parts) == 2 and os.path.basename(parts[0]) in _WRAPPER_COMMANDS:
            rest = parts[1].lstrip()
            # Drop the wrapper's own flags (`nice -n 5 rm ...`, `stdbuf -o0 ...`).
            # A detached flag value (`-n 5`) is dropped with its flag; since
            # matching is additive, over-stripping can only add a confirmation.
            while rest.startswith("-"):
                flag_parts = rest.split(None, 1)
                if len(flag_parts) < 2:
                    break
                rest = flag_parts[1].lstrip()
                value_parts = rest.split(None, 1)
                if len(value_parts) == 2 and value_parts[0].isdigit():
                    rest = value_parts[1].lstrip()
            if rest:
                text = rest
                continue
        break

    # Reduce the executable to its basename: /usr/bin/rm -> rm
    parts = text.split(None, 1)
    if parts and "/" in parts[0]:
        base = os.path.basename(parts[0])
        if base:
            text = base if len(parts) == 1 else f"{base} {parts[1]}"

    return text


def _is_sensitive_path(path: str) -> bool:
    """True if ``path`` points at credentials or other sensitive material.

    The raw string is matched *and* the fully expanded absolute path, so a
    relative path (``../../.ssh/id_rsa``) or a ``~``-relative one cannot slip
    past a check written in terms of absolute path fragments.
    """
    candidates = [path]
    try:
        expanded = os.path.expanduser(path)
        # Resolve without requiring the path to exist; strict=False is the
        # default but is spelled out because the file usually will not exist.
        candidates.append(str(Path(expanded).resolve(strict=False)))
        candidates.append(os.path.abspath(expanded))
    except (OSError, ValueError, RuntimeError):
        # A malformed path (embedded NUL, symlink loop) is treated as
        # sensitive rather than waved through.
        return True

    # Several patterns are written as directory fragments ("/.ssh/"), which a
    # path naming the directory itself ("~/.ssh") would otherwise miss.
    candidates.extend([c.rstrip("/") + "/" for c in list(candidates)])

    return any(
        pattern in candidate for candidate in candidates for pattern in _SENSITIVE_PATH_PATTERNS
    )


class SafetyClassifier:
    """Classify tool calls by risk level using regex patterns."""

    def __init__(
        self, config: SafetyConfig, confirm_required: set[str] | None = None
    ) -> None:
        self.mode = config.mode
        self._confirm_patterns = [re.compile(p) for p in config.always_confirm]
        self._blocked_patterns = [re.compile(p) for p in config.blocked]
        # Tool names declaring requires_confirmation=True in their definition.
        # Used for escalation only — never to downgrade a computed risk.
        self._confirm_required = set(confirm_required or ())

    def classify_command(self, command: str) -> Risk:
        """Classify a shell command string by risk level.

        Checks the full command against blocked patterns first (to catch
        multi-token patterns like fork bombs), then splits on shell operators
        (&&, ||, ;, &, |, (, ) and newlines) using the shared tokenizer from
        natshell.safety.command_split — the same splitter execute_shell uses,
        so classification and execution always see the same sub-command
        boundaries.  Each sub-command is classified independently, returning
        the highest risk found.
        Also flags subshells and backtick expansions as CONFIRM.
        """
        # Check blocked patterns against the full command first
        # (some patterns like fork bombs or pipe-based patterns span operators)
        full_variants = (command, _normalize_invocation(command))
        for pattern in self._blocked_patterns:
            for variant in full_variants:
                if pattern.search(variant):
                    logger.warning(f"BLOCKED command: {command}")
                    return Risk.BLOCKED

        # Use the shared tokenizer (same as execute_shell) so that classifier
        # and executor split on identical boundaries — including newlines and
        # parentheses that were previously missed by one or the other.
        sub_commands = [sub for sub in split_commands(command) if sub]

        # Every blocked check completes before any CONFIRM is returned.  A
        # whole-command CONFIRM match used to return here, so a BLOCKED
        # sub-command later in the chain was never reached.
        worst_risk = Risk.SAFE
        for sub in sub_commands:
            risk = self._classify_single(sub)
            if risk == Risk.BLOCKED:
                return Risk.BLOCKED
            if risk == Risk.CONFIRM:
                worst_risk = Risk.CONFIRM

        if worst_risk == Risk.CONFIRM:
            return Risk.CONFIRM

        # Whole-command confirm patterns (spanning operators, e.g. a fetch
        # piped into an interpreter) are checked once the per-command pass has
        # ruled out anything blocked.
        for pattern in self._confirm_patterns:
            for variant in full_variants:
                if pattern.search(variant):
                    return Risk.CONFIRM

        # Flag commands using subshells or backtick expansion
        if re.search(r"`[^`]+`|\$\([^)]+\)", command):
            return Risk.CONFIRM

        return Risk.SAFE

    def _classify_single(self, command: str) -> Risk:
        """Classify a single command (no chaining operators)."""
        # Test the command as written and with wrappers, env assignments and
        # the executable's path stripped.  Checking both is additive: it can
        # only widen pattern coverage, never narrow it.
        normalized = _normalize_invocation(command)
        variants = (command, normalized) if normalized != command else (command,)

        # Check blocked first — every blocked check completes before any
        # CONFIRM can be returned, so a whole-command CONFIRM match cannot
        # mask a BLOCKED sub-command.
        for pattern in self._blocked_patterns:
            for variant in variants:
                if pattern.search(variant):
                    logger.warning(f"BLOCKED command: {command}")
                    return Risk.BLOCKED

        # Check confirmation-required patterns
        for pattern in self._confirm_patterns:
            for variant in variants:
                if pattern.search(variant):
                    return Risk.CONFIRM

        # Heuristic: sudo always requires confirmation
        if any(v.strip().startswith("sudo ") for v in variants):
            return Risk.CONFIRM

        # Heuristic: redirecting to system paths
        if re.search(r">\s*/(?:etc|boot|usr|var/lib)/", command):
            return Risk.CONFIRM

        return Risk.SAFE

    def classify_tool_call(self, tool_name: str, arguments: dict) -> Risk:
        """Classify any tool call by risk level.

        Tools without an explicit branch below fall through to CONFIRM unless
        they are on the read-only allowlist, and a tool declaring
        requires_confirmation=True is escalated to CONFIRM if the rules would
        otherwise have returned SAFE.
        """
        risk = self._classify_tool_call_inner(tool_name, arguments)

        # requires_confirmation is an escalation only — it never downgrades a
        # BLOCKED result, and it does not override danger mode, which is an
        # explicit user choice to skip confirmations.
        if (
            risk == Risk.SAFE
            and self.mode != "danger"
            and tool_name in self._confirm_required
        ):
            return Risk.CONFIRM
        return risk

    def _classify_tool_call_inner(self, tool_name: str, arguments: dict) -> Risk:
        if tool_name == "execute_shell":
            command = arguments.get("command", "")
            risk = self.classify_command(command)
            # In danger mode, downgrade CONFIRM to SAFE (but not BLOCKED)
            if self.mode == "danger" and risk == Risk.CONFIRM:
                return Risk.SAFE
            return risk

        if tool_name == "write_file":
            if _is_sensitive_path(arguments.get("path", "")):
                return Risk.CONFIRM
            if self.mode == "danger":
                return Risk.SAFE
            return Risk.CONFIRM

        if tool_name == "edit_file":
            # Always confirm edits; also check sensitive paths
            if _is_sensitive_path(arguments.get("path", "")):
                return Risk.CONFIRM
            if self.mode == "danger":
                return Risk.SAFE
            return Risk.CONFIRM

        if tool_name == "run_code":
            if self.mode == "danger":
                return Risk.SAFE
            return Risk.CONFIRM

        if tool_name == "read_file":
            if _is_sensitive_path(arguments.get("path", "")):
                return Risk.CONFIRM
            return Risk.SAFE

        if tool_name == "search_files":
            # Searching a sensitive directory exfiltrates its contents just as
            # readily as reading a file in it.
            if _is_sensitive_path(arguments.get("path", "")):
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

        # Fail closed: a tool with no rule above is only SAFE if it is on the
        # read-only allowlist.  The old fallthrough returned SAFE for anything
        # unlisted, so any newly registered tool — update_config among them —
        # ran without a confirmation dialog purely because nobody had
        # classified it yet.
        if tool_name in _READ_ONLY_TOOLS:
            return Risk.SAFE

        logger.debug("Tool %s has no classification rule — defaulting to CONFIRM", tool_name)
        return Risk.CONFIRM
