"""Execute shell commands — the primary tool for system interaction."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import time

from natshell.platform import is_windows
from natshell.tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# Maximum characters of output to return to the model (mutable — scaled by agent loop)
_max_output_chars = 4000
_head_chars = 2000
_tail_chars = 1500


def configure_limits(max_output_chars: int) -> None:
    """Set shell output truncation limits (called by agent loop based on context size)."""
    global _max_output_chars, _head_chars, _tail_chars, _base_max_output_chars
    _max_output_chars = max_output_chars
    _head_chars = max_output_chars // 2
    _tail_chars = int(max_output_chars * 0.375)
    _base_max_output_chars = max_output_chars


def reset_limits() -> None:
    """Restore default truncation limits (used by tests)."""
    configure_limits(4000)


# Step-aware output scaling — reduces output budget as context fills up
_base_max_output_chars: int = 4000


def configure_step_scaling(step: int, max_steps: int) -> None:
    """Progressively reduce output limits as the step count increases.

    Called by the agent loop at the start of each step.  The scale factor
    drops linearly from 1.0 (step 0) to 0.3 (step == max_steps), so later
    tool results occupy less context and leave room for the model to reason.
    """
    if max_steps <= 0:
        return
    scale = max(0.3, 1.0 - 0.7 * (step / max_steps))
    effective = int(_base_max_output_chars * scale)
    configure_limits(effective)

# ── Sudo password support ───────────────────────────────────────────────────

_sudo_password: str | None = None
_sudo_password_time: float = 0.0
_SUDO_PW_TIMEOUT = 300  # 5 minutes

_SUDO_RE = re.compile(r"\bsudo\b")

# Splits a command on shell operators, keeping delimiters.
# Used to identify sub-commands that start with sudo (vs. "sudo" appearing
# inside quoted arguments like `echo "use sudo"`).
_CMD_SPLIT_RE = re.compile(r"(&&|\|\||[;&|(])")

# Package manager commands that may prompt "Do you want to continue? [Y/n]"
_PKG_MANAGER_RE = re.compile(
    r"\b(?:apt|apt-get|dnf|yum|pacman|zypper|apk|emerge)\b"
)


def _inject_sudo_dash_s(command: str) -> tuple[str, int]:
    """Replace ``sudo`` with ``sudo -S`` only at command-invocation positions.

    Returns ``(modified_command, replacement_count)``.  Unlike a naive
    ``\\bsudo\\b`` substitution this avoids matching ``sudo`` inside string
    arguments (e.g. ``echo "use sudo"``), preventing password leaks to
    stdin-reading commands that follow in a pipeline or chain.
    """
    parts = _CMD_SPLIT_RE.split(command)
    count = 0
    result_parts: list[str] = []
    for part in parts:
        # Shell-operator delimiters pass through unchanged
        if _CMD_SPLIT_RE.fullmatch(part):
            result_parts.append(part)
            continue
        # Check if this sub-command starts with sudo (optional leading whitespace)
        stripped = part.lstrip()
        if re.match(r"sudo(?:\s|$)", stripped):
            idx = part.index("sudo")
            part = part[:idx] + "sudo -S" + part[idx + 4:]
            count += 1
        result_parts.append(part)
    return "".join(result_parts), count


def _has_sudo_invocation(command: str) -> bool:
    """Return True if the command contains ``sudo`` at a command-invocation position."""
    _, count = _inject_sudo_dash_s(command)
    return count > 0


# stderr patterns that mean "sudo wanted a password but couldn't get one"
_SUDO_NEEDS_PW = [
    "sudo: a terminal is required to read the password",
    "sudo: a password is required",
    "sudo: no tty present and no askpass program specified",
    "sudo: no password was provided",
]

# Environment variables that should not be exposed to LLM-executed commands
_SENSITIVE_ENV_VARS = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "GITLAB_TOKEN",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "DATABASE_URL",
    "DB_PASSWORD",
    "NATSHELL_API_KEY",
    "REDIS_URL",
    "MONGODB_URI",
    "AMQP_URL",
}

# Any env var ending with one of these suffixes is also filtered
_SENSITIVE_SUFFIXES = ("_PASSWORD", "_SECRET", "_TOKEN", "_API_KEY")


def _get_sudo_password() -> str | None:
    """Return the cached sudo password, or None if expired."""
    global _sudo_password, _sudo_password_time
    if _sudo_password and (time.monotonic() - _sudo_password_time) > _SUDO_PW_TIMEOUT:
        _sudo_password = None
    return _sudo_password


def set_sudo_password(password: str) -> None:
    """Cache the sudo password for subsequent execute_shell calls."""
    global _sudo_password, _sudo_password_time
    _sudo_password = password
    _sudo_password_time = time.monotonic()


def clear_sudo_password() -> None:
    """Clear the cached sudo password."""
    global _sudo_password
    _sudo_password = None


def needs_sudo_password(result: ToolResult) -> bool:
    """Return True if the result indicates sudo needed a password it didn't get."""
    if result.exit_code == 0:
        return False
    return any(msg in result.error for msg in _SUDO_NEEDS_PW)


# ── Tool definition ─────────────────────────────────────────────────────────

DEFINITION = ToolDefinition(
    name="execute_shell",
    description=(
        "Execute a shell command on the user's system and return the output. "
        "Use this to run any CLI command: check system state, install packages, "
        "manage services, scan networks, analyze disk usage, process files, etc. "
        "The command runs as the current user via bash. Use sudo when elevated "
        "privileges are needed. Prefer single commands per call; for multi-step "
        "operations, call this tool multiple times and observe results between steps."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": (
                    "Maximum seconds to wait for the command to complete. "
                    "Default 60. Use higher values for long-running operations "
                    "like network scans or package installs. Maximum 300."
                ),
            },
        },
        "required": ["command"],
    },
    requires_confirmation=False,  # Safety classifier handles per-command checks
)


# Long-running command patterns → minimum timeout (seconds)
_LONG_RUNNING_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    # Network scanning/discovery
    (re.compile(r"\bnmap\b"), 120),
    (re.compile(r"\barp-scan\b"), 120),
    (re.compile(r"\btcpdump\b"), 120),
    (re.compile(r"\bnetdiscover\b"), 120),
    (re.compile(r"\bmasscan\b"), 120),
    # Package management
    (re.compile(r"\bapt\s+(install|upgrade|dist-upgrade|full-upgrade)\b"), 300),
    (re.compile(r"\bapt-get\s+(install|upgrade|dist-upgrade)\b"), 300),
    (re.compile(r"\bdnf\s+(install|update|upgrade)\b"), 300),
    (re.compile(r"\byum\s+(install|update)\b"), 300),
    (re.compile(r"\bpacman\s+-S"), 300),
    (re.compile(r"\bbrew\s+(install|upgrade)\b"), 300),
    # Build/compile
    (re.compile(r"\bmake\b"), 300),
    (re.compile(r"\bcargo\s+build\b"), 300),
    (re.compile(r"\bnpm\s+(install|ci)\b"), 300),
    (re.compile(r"\bpip\s+install\b"), 300),
    (re.compile(r"\bgcc\b|\bg\+\+\b"), 120),
    (re.compile(r"\brustc\b"), 120),
    # Filesystem scans
    (re.compile(r"\bfind\s+/"), 120),
    (re.compile(r"\bdu\s+.*-[a-zA-Z]*s"), 120),
    (re.compile(r"\brsync\b"), 300),
    (re.compile(r"\bwget\b|\bcurl\b.*-[oO]"), 120),
    # Disk operations
    (re.compile(r"\bdd\b"), 300),
]


def _min_timeout_for(command: str) -> int:
    """Return the minimum timeout for a command based on known long-running patterns."""
    for pattern, min_timeout in _LONG_RUNNING_PATTERNS:
        if pattern.search(command):
            return min_timeout
    return 0


def _truncate_output(text: str) -> tuple[str, bool]:
    """Truncate output to fit in context window, preserving head and tail."""
    if len(text) <= _max_output_chars:
        return text, False

    lines = text.splitlines()
    head = text[:_head_chars]
    tail = text[-_tail_chars:]

    # Count omitted lines for the message
    head_lines = head.count("\n")
    tail_lines = tail.count("\n")
    omitted = len(lines) - head_lines - tail_lines

    truncated = f"{head}\n... [{omitted} lines truncated] ...\n{tail}"
    return truncated, True


async def execute_shell(
    command: str,
    timeout: int = 60,
) -> ToolResult:
    """Execute a shell command and return structured results."""
    # Coerce timeout to int (LLMs may send it as a string) and clamp
    try:
        timeout = int(timeout)
    except (TypeError, ValueError):
        timeout = 60
    timeout = max(1, min(timeout, 300))

    # Auto-raise timeout for known long-running commands
    min_timeout = _min_timeout_for(command)
    if min_timeout > timeout:
        logger.info("Auto-raised timeout %ds → %ds for long-running command", timeout, min_timeout)
        timeout = min_timeout
    timeout = max(1, min(timeout, 300))  # re-clamp after auto-raise

    # Redact sudo -S from log output to avoid leaking password plumbing
    sudo_pw = _get_sudo_password()
    log_cmd = command
    if sudo_pw and _has_sudo_invocation(command):
        log_cmd, _ = _inject_sudo_dash_s(command)
    logger.info(f"Executing: {log_cmd} (timeout={timeout}s)")

    # Filter sensitive environment variables before passing to subprocess
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _SENSITIVE_ENV_VARS
        and not any(k.endswith(s) for s in _SENSITIVE_SUFFIXES)
    }
    env["LC_ALL"] = "C"  # Consistent output for parsing

    try:
        run_kwargs: dict = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
            "env": env,
            "cwd": os.getcwd(),
        }

        # start_new_session is POSIX-only (setsid). On Windows, use
        # CREATE_NEW_PROCESS_GROUP to achieve similar isolation.
        if is_windows():
            run_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            run_kwargs["start_new_session"] = True

        # If we have a cached sudo password and the command invokes sudo
        # at a command position (not inside string arguments), add -S so
        # sudo reads the password from stdin.  Uses _inject_sudo_dash_s()
        # to avoid matching "sudo" inside quoted text — prevents password
        # leakage to stdin-reading commands.
        # Sudo is not applicable on Windows (UAC is a different paradigm).
        if not is_windows() and sudo_pw and _has_sudo_invocation(command):
            command, sudo_count = _inject_sudo_dash_s(command)
            stdin_data = (sudo_pw + "\n") * sudo_count
            # Append trailing "y\n" only for package manager commands that
            # may prompt "Do you want to continue? [Y/n]".  Not appended
            # for other commands to avoid feeding "y" to interactive programs
            # (fdisk, mysql, python, etc.) that read from stdin.
            if _PKG_MANAGER_RE.search(command):
                stdin_data += "y\n" * 3
            run_kwargs["input"] = stdin_data
        else:
            run_kwargs["stdin"] = subprocess.DEVNULL

        # Build the shell command list based on platform
        if is_windows():
            shell_cmd = [
                "powershell", "-NoProfile", "-NonInteractive",
                "-Command", command,
            ]
        else:
            shell_cmd = ["bash", "-c", command]

        # Run in thread to avoid blocking the async event loop
        result = await asyncio.to_thread(
            subprocess.run,
            shell_cmd,
            **run_kwargs,
        )

        stdout, stdout_truncated = _truncate_output(result.stdout)
        stderr, stderr_truncated = _truncate_output(result.stderr)

        # sudo -S echoes a password prompt to stderr — strip it
        if sudo_pw and not is_windows():
            stderr = "\n".join(
                line
                for line in stderr.splitlines()
                if not line.startswith("[sudo] password for")
            ).strip()

        return ToolResult(
            output=stdout,
            error=stderr,
            exit_code=result.returncode,
            truncated=stdout_truncated or stderr_truncated,
        )

    except subprocess.TimeoutExpired:
        return ToolResult(
            output="",
            error=f"Command timed out after {timeout} seconds.",
            exit_code=124,
        )
    except FileNotFoundError:
        shell_name = "PowerShell" if is_windows() else "bash"
        return ToolResult(
            output="",
            error=f"{shell_name} not found. Is it installed?",
            exit_code=127,
        )
    except Exception as e:
        return ToolResult(
            output="",
            error=f"Failed to execute command: {type(e).__name__}: {e}",
            exit_code=1,
        )
