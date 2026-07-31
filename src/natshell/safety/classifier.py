"""Command risk classification — pattern-based, deterministic, fast."""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any

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


# Interpreters a fetched script can be piped into, and credential paths worth
# confirming before they reach the network.  Shared by the patterns below.
_INTERPRETERS = r"sh|bash|zsh|ksh|dash|fish|python3?|perl|ruby|node|php"
_NET_CLIENTS = r"curl|wget|nc|ncat|netcat|ssh|scp|sftp|ftp|rsync"
# \b rather than a trailing slash so that the directory itself matches:
# "tar czf - ~/.ssh | curl ..." exfiltrates the keys without naming one.
_CREDENTIAL_PATHS = (
    r"\.ssh\b|/id_rsa|/id_ed25519|\.aws/credentials|\.kube/config|"
    r"/etc/shadow|\.docker/config\.json|\.gnupg\b"
)

# Confirmation patterns that apply regardless of the user's config.
#
# The shipped always_confirm list is a denylist of first tokens, which cannot
# express a command that is dangerous because of its *shape* -- a download piped
# into a shell, an interpreter one-liner, a write into a credential path.  These
# live in Python rather than config.default.toml so that neither an edited
# config nor a missing default file can remove them; user config extends this
# list, it does not replace it.
BASELINE_CONFIRM_PATTERNS: tuple[str, ...] = (
    # Fetch piped straight into an interpreter — the classic install one-liner.
    rf"\b(?:curl|wget|fetch)\b[^|]*\|\s*(?:sudo\s+)?(?:\S*/)?(?:{_INTERPRETERS})\b",
    # Interpreter one-liners: the code never touches disk, so nothing else sees it.
    rf"\b(?:{_INTERPRETERS})\s+(?:-\S+\s+)*-[ce]\s",
    # Credentials handed to something that can put them on the network.
    rf"\b(?:{_NET_CLIENTS})\b[^|;&]*(?:{_CREDENTIAL_PATHS})",
    rf"(?:{_CREDENTIAL_PATHS})[^|]*\|\s*(?:{_NET_CLIENTS})\b",
    # Writes into paths that get sourced or trusted on the next login.
    r">>?\s*[^\s>|;&]*\.(?:ssh|aws|kube|docker|gnupg)/",
    r">>?\s*[^\s>|;&]*\.(?:bashrc|bash_profile|profile|zshrc|zprofile|zshenv)\b",
    r">>?\s*[^\s>|;&]*/(?:authorized_keys|known_hosts)\b",
    # find that mutates rather than lists.
    r"\bfind\b[^|;&]*\s-(?:delete|exec|execdir|ok|okdir)\b",
    # Recursive and setuid permission changes.
    r"\bchmod\b[^|;&]*\s-[a-zA-Z]*R",
    r"\bchmod\b\s+[ugoa]*\+[rwxt]*s\b",
    # Irreversible or untracked-file-destroying operations.
    r"^shred\b",
    r"^truncate\b",
    r"\bgit\s+clean\b",
)


# Block devices, spelled to include the ones a machine built after ~2012 has.
# The original class, [sh]d[a-z], covered neither NVMe nor SD cards nor virtio.
_BLOCK_DEVICES = r"(?:sd|hd|nvme|mmcblk|vd|xvd)[a-z0-9]*"

# Commands that are never executed, whatever the mode.
#
# These are in Python for the same reason as the confirm baseline, and one more:
# BLOCKED is the only tier that survives mode = "danger", danger_fast, and
# --danger-fast, so it is the tier that most needs to not be editable by a tool
# call.  User config extends this list.
#
# Kept deliberately narrow.  Because config can no longer remove an entry, a
# pattern here is unoverridable, so this covers only what is unrecoverable and
# never intentional -- the filesystem root, a raw block device.  Destructive but
# legitimate administration ("rm -rf /home/old-user") stays at CONFIRM.
BASELINE_BLOCKED_PATTERNS: tuple[str, ...] = (
    # Fork bomb by shape: NAME() { NAME | NAME & }; NAME, whatever NAME is.
    # The shipped literal was compiled as a regex, so its unescaped '|' became
    # an alternation and only the canonical spelling ever matched.
    r"([\w:]+)\s*\(\s*\)\s*\{[^}]*\|[^}]*&[^}]*\}\s*;\s*\1",
    re.escape(":(){ :|:& };:"),
    # rm targeting the filesystem root, with the recursive flag written any of
    # the ways it can be written, and --no-preserve-root on either side of it.
    r"^rm\s+(?:-\S+\s+)*-[a-zA-Z]*[rR][a-zA-Z]*\s+(?:-\S+\s+)*/\s*$",
    r"^rm\s+(?:-\S+\s+)*-[a-zA-Z]*[rR][a-zA-Z]*\s+(?:-\S+\s+)*/\*",
    r"^mv\s+/\s",
    # Writing over a block device.  No '$' anchor: trailing arguments such as
    # "bs=1M" put the device in the middle of the string, not at the end.
    rf"^dd\s+.*\bof=/dev/{_BLOCK_DEVICES}",
    rf"^mkfs(?:\.\w+)?\s+(?:\S+\s+)*/dev/{_BLOCK_DEVICES}",
    rf">\s*/dev/{_BLOCK_DEVICES}",
    r"^diskutil\s+eraseDisk",
    # Windows
    r"^format\s+[Cc]:",
    r"^rd\s+/[sS]\s+/[qQ]\s+[Cc]:\\",
    r"Remove-Item\s+-Recurse\s+-Force\s+[Cc]:\\",
)


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


# Tools that only read.  Everything not named here requires confirmation,
# including tools added in future — the default has to be that an unclassified
# capability is dangerous, not that it is safe.
_READ_ONLY_TOOLS = frozenset(
    {
        "fetch_url",
        "kiwix_search",
        "list_directory",
        "natshell_help",
        "search_files",
        "skill",
    }
)


# NOTE: writes to the working-memory file (agents.md) used to return SAFE via an
# endswith() test on the model-supplied path.  That test ran before the
# sensitive-path loop and before the mode checks, so it was SAFE in every mode,
# for any path ending in those characters — including a symlink pointing
# somewhere else entirely, and including a .natshell/agents.md in a directory
# chosen by the caller.
#
# The file is read from cwd on every run and spliced verbatim into the system
# prompt, so an unconfirmed write to it is a persistent instruction change that
# survives /clear and outlives the session.  There is no path test that makes
# that safe, because the risk is in the content rather than the location, so the
# special case is gone: working-memory writes confirm like any other write.


class SafetyClassifier:
    """Classify tool calls by risk level using regex patterns."""

    def __init__(self, config: SafetyConfig, registry: Any = None) -> None:
        self.mode = config.mode
        # Optional ToolRegistry, consulted only to honour a tool definition's
        # requires_confirmation flag.  Optional so that constructing a
        # classifier for pattern checks alone stays a one-argument call.
        self._registry = registry
        # MULTILINE so that the '^' every shipped pattern starts with anchors to
        # each line of a multi-line command, not just the first.  Sub-commands are
        # split out below as well; this is the belt to that pair of braces.
        # User config extends the baseline patterns rather than replacing them.
        self._confirm_patterns = [
            re.compile(p, re.MULTILINE)
            for p in (*BASELINE_CONFIRM_PATTERNS, *config.always_confirm)
        ]
        self._blocked_patterns = [
            re.compile(p, re.MULTILINE)
            for p in (*BASELINE_BLOCKED_PATTERNS, *config.blocked)
        ]

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

    def _declares_confirmation(self, tool_name: str) -> bool:
        """True if the tool's own definition asks for confirmation.

        ToolDefinition.requires_confirmation was previously read by nothing at
        all, so a tool declaring True — as update_config did — was still
        auto-approved.  It is honoured here, but only upward: a definition can
        ask for a dialog it would not otherwise get, never waive one.
        """
        if self._registry is None:
            return False
        definition = self._registry.get_definition(tool_name)
        return bool(definition is not None and definition.requires_confirmation)

    def classify_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        *,
        honor_danger_mode: bool = True,
    ) -> Risk:
        """Classify any tool call by risk level.

        Unrecognized tools return CONFIRM.  The previous fallthrough was SAFE,
        which meant any tool without an explicit branch below ran with no
        dialog — update_config among them, which can persist
        safety.mode = "danger", danger_fast, mcp.safety_mode = "permissive",
        and a remote inference URL.  Reaching that took no user interaction:
        fetch_url is SAFE, so injected page content could ask for it directly.

        Pass ``honor_danger_mode=False`` to get the risk before the danger-mode
        downgrade is applied.  The MCP transport needs that: mode = "danger" is
        a local setting about this operator's own session, and it must not
        quietly grant a *remote* client the unconfirmed execution that
        mcp.safety_mode = "strict" was set to deny.
        """
        mode = self.mode if honor_danger_mode else "confirm"

        if self._declares_confirmation(tool_name) and mode != "danger":
            return Risk.CONFIRM

        if tool_name == "execute_shell":
            command = arguments.get("command", "")
            risk = self.classify_command(command)
            # In danger mode, downgrade CONFIRM to SAFE (but not BLOCKED)
            if mode == "danger" and risk == Risk.CONFIRM:
                return Risk.SAFE
            return risk

        if tool_name == "write_file":
            path = arguments.get("path", "")
            for pattern in _SENSITIVE_PATH_PATTERNS:
                if pattern in path:
                    return Risk.CONFIRM
            if mode == "danger":
                return Risk.SAFE
            return Risk.CONFIRM

        if tool_name == "edit_file":
            # Always confirm edits; also check sensitive paths
            path = arguments.get("path", "")
            for pattern in _SENSITIVE_PATH_PATTERNS:
                if pattern in path:
                    return Risk.CONFIRM
            if mode == "danger":
                return Risk.SAFE
            return Risk.CONFIRM

        if tool_name == "run_code":
            if mode == "danger":
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
                if mode == "danger":
                    return Risk.SAFE
                return Risk.CONFIRM
            # Unknown operation — let the tool handle the error, but confirm
            return Risk.CONFIRM

        if tool_name in _READ_ONLY_TOOLS:
            return Risk.SAFE

        # Fail closed.  Anything with no branch above — update_config, a tool
        # registered by a skill, a tool added next year — confirms.
        logger.debug(
            "Tool %s has no classification rule; requiring confirmation", tool_name
        )
        if mode == "danger":
            return Risk.SAFE
        return Risk.CONFIRM
