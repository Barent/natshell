"""Unified sudo plumbing for shell tools (extracted ``execute_shell.py``).

The four free functions that were scattered inside ``execute_shell.py`` now
live in one place:

- ``inject_dash_s`` — rewrite ``sudo`` → ``sudo -S`` at invocation positions.
- ``has_invocation`` — ``True`` if the command calls sudo.
- ``get_password`` / ``set_password`` / ``clear_password`` — the 5-minute
  cached sudo password, shared by blocking and streaming paths.
- ``needs_password`` — detect "sudo asked for a password but couldn't get
  one" in a :class:`ToolResult`.
- ``prepare_for_run`` — the (command, stdin) payload for a sudo-aware run,
  including the package-manager ``y\\n`` x3 tail.
- ``scrub_prompt`` — strip sudo's ``[sudo] password for …`` line from stderr.
- ``SudoHandler`` — a thin convenience class over the same state for code
  that prefers a single object; the module-level functions remain the
  primary API (they share state with the class instance).

All behaviour is byte-identical to the pre-extraction functions; see
``tests/test_tools.py`` and ``tests/test_stream_execute_shell.py``.
"""

from __future__ import annotations

import re
import time

from natshell.platform import is_windows
from natshell.safety.command_split import split_with_delimiters
from natshell.tools.registry import ToolResult

__all__ = [
    "SudoHandler",
    "inject_dash_s",
    "has_invocation",
    "get_password",
    "set_password",
    "clear_password",
    "needs_password",
    "prepare_for_run",
    "scrub_prompt",
    # constants, kept stable for tests / external references
    "_SUDO_PW_TIMEOUT",
    "_SUDO_RE",
    "_PKG_MANAGER_RE",
    "_SUDO_NEEDS_PW",
]

# ── State & constants ──────────────────────────────────────────────────────

password: str | None = None
password_time: float = 0.0

# How long (seconds) a cached sudo password stays valid.
_SUDO_PW_TIMEOUT = 300  # 5 minutes

# A bare word-anchored ``sudo`` — used where a full tokenizer is unnecessary.
_SUDO_RE = re.compile(r"\bsudo\b")

# Package manager commands that may prompt "Do you want to continue? [Y/n]".
_PKG_MANAGER_RE = re.compile(
    r"\b(?:apt|apt-get|dnf|yum|pacman|zypper|apk|emerge)\b"
)

# stderr patterns that mean "sudo wanted a password but couldn't get one".
_SUDO_NEEDS_PW = [
    "sudo: a terminal is required to read the password",
    "sudo: a password is required",
    "sudo: no tty present and no askpass program specified",
    "sudo: no password was provided",
]


# ── Function API ───────────────────────────────────────────────────────────


def inject_dash_s(command: str) -> tuple[str, int]:
    """Replace ``sudo`` with ``sudo -S`` only at command-invocation positions.

    Returns ``(modified_command, replacement_count)``.  Uses the shared
    tokenizer from ``natshell.safety.command_split`` (same as the classifier)
    so splitting logic is identical between classify and execute paths.
    """
    parts = split_with_delimiters(command)
    count = 0
    result_parts: list[str] = []
    for i_part, part in enumerate(parts):
        # Odd indices are delimiters — pass through unchanged
        if i_part % 2 == 1:
            result_parts.append(part)
            continue
        # Even indices are tokens — check for sudo at start of position
        stripped = part.lstrip()
        if re.match(r"sudo(?:\s|$)", stripped):
            idx = part.index("sudo")
            part = part[:idx] + "sudo -S" + part[idx + 4:]
            count += 1
        result_parts.append(part)
    return "".join(result_parts), count


def has_invocation(command: str) -> bool:
    """Return True if the command calls ``sudo`` at an invocation position."""
    _, count = inject_dash_s(command)
    return count > 0


def get_password() -> str | None:
    """Return the cached sudo password (or ``None`` if expired/absent)."""
    global password, password_time
    if password and (time.monotonic() - password_time) > _SUDO_PW_TIMEOUT:
        password = None
    return password


def set_password(new_password: str) -> None:
    """Cache a sudo password for subsequent shell executions."""
    global password, password_time
    password = new_password
    password_time = time.monotonic()


def clear_password() -> None:
    """Clear any cached sudo password."""
    global password
    password = None


def needs_password(result: ToolResult) -> bool:
    """Detect "sudo asked for a password but didn't get one".

    Scans both stderr and stdout (not just exit code) — models often hide
    the failure with ``2>&1`` or a trailing command.  The signatures in
    ``_SUDO_NEEDS_PW`` are specific enough that a simple substring match
    is reliable.
    """
    haystack = f"{result.error}\n{result.output}"
    return any(msg in haystack for msg in _SUDO_NEEDS_PW)


def prepare_for_run(command: str, sudo_pw: str | None = None) -> tuple[str, str | None]:
    """Return ``(final_command, stdin_text_or_None)`` for a sudo-aware run.

    The password used is ``sudo_pw`` when supplied, otherwise the
    module-level cached password (see :func:`get_password`).  When a
    non-empty password is in play AND the command invokes ``sudo`` at a
    command position, rewrite ``sudo`` → ``sudo -S`` (one line of
    ``password + "\n"`` per invocation) and, for the eight package managers in
    :data:`_PKG_MANAGER_RE`, append ``y\\n`` x3 to answer their "Do you want
    to continue?" prompts.  Interactive programs that are *not* one of those
    package managers never receive a stray ``y``.

    On Windows, or when there is no password or no sudo invocation, returns
    ``(command, None)`` — the caller then feeds stdin from ``/dev/null``.
    """
    pw = sudo_pw if sudo_pw is not None else get_password()
    if pw and not is_windows() and has_invocation(command):
        command, count = inject_dash_s(command)
        stdin_text = (pw + "\n") * count
        if _PKG_MANAGER_RE.search(command):
            stdin_text += "y\n" * 3
        return command, stdin_text
    return command, None


def scrub_prompt(stderr: str, sudo_pw: str | None = None) -> str:
    """Remove sudo's ``[sudo] password for …`` prompt lines from stderr.

    ``sudo``/``sudo -S`` echoes a prompt to stderr; that must not reach the
    model (and it's a credential plumbing leak).  A no-op when no password is
    in play or on Windows, so non-sudo runs keep their stderr untouched.
    """
    if sudo_pw and not is_windows():
        return (
            "\n".join(
                line for line in stderr.splitlines()
                if not line.startswith("[sudo] password for")
            ).strip()
        )
    return stderr


# ── Class API ──────────────────────────────────────────────────────────────


class SudoHandler:
    """Convenience facade over the module-level sudo plumbing.

    All methods delegate to the module functions, so a ``SudoHandler`` and
    the free-function API share state — setting a password on one is
    visible through the other.  Existing code using the historical names
    (``execute_shell.set_sudo_password`` etc.) and new code using
    ``SudoHandler().set_password(...)`` are interchangeable.
    """

    def has_invocation(self, command: str) -> bool:
        return has_invocation(command)

    def needs_password(self, result: ToolResult) -> bool:
        return needs_password(result)

    def prepare_for_run(
        self, command: str, pw: str | None = None
    ) -> tuple[str, str | None]:
        return prepare_for_run(command, pw)

    def scrub_prompt(self, stderr: str, pw: str | None = None) -> str:
        return scrub_prompt(stderr, pw)

    def get_password(self) -> str | None:
        return get_password()

    def set_password(self, password: str) -> None:
        set_password(password)

    def clear_password(self) -> None:
        clear_password()

    def inject_dash_s(self, command: str) -> tuple[str, int]:
        return inject_dash_s(command)


#: Shared module-level handler — ``from natshell.tools.sudo import SUDO``.
SUDO = SudoHandler()
