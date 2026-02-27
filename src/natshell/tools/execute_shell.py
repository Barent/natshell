"""Execute shell commands — the primary tool for system interaction."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import time
from natshell.tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# Maximum characters of output to return to the model
MAX_OUTPUT_CHARS = 4000
HEAD_CHARS = 2000
TAIL_CHARS = 1500

# ── Sudo password support ───────────────────────────────────────────────────

_sudo_password: str | None = None
_sudo_password_time: float = 0.0
_SUDO_PW_TIMEOUT = 300  # 5 minutes

_SUDO_RE = re.compile(r'\bsudo\b')

# stderr patterns that mean "sudo wanted a password but couldn't get one"
_SUDO_NEEDS_PW = [
    "sudo: a terminal is required to read the password",
    "sudo: a password is required",
    "sudo: no tty present and no askpass program specified",
]

# Environment variables that should not be exposed to LLM-executed commands
_SENSITIVE_ENV_VARS = {
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "GITHUB_TOKEN", "GH_TOKEN", "GITLAB_TOKEN",
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
    "DATABASE_URL", "DB_PASSWORD",
    "NATSHELL_API_KEY",
}


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
                    "Default 30. Use higher values for long-running operations "
                    "like network scans or package installs. Maximum 300."
                ),
            },
        },
        "required": ["command"],
    },
    requires_confirmation=False,  # Safety classifier handles per-command checks
)


def _truncate_output(text: str) -> tuple[str, bool]:
    """Truncate output to fit in context window, preserving head and tail."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text, False

    lines = text.splitlines()
    head = text[:HEAD_CHARS]
    tail = text[-TAIL_CHARS:]

    # Count omitted lines for the message
    head_lines = head.count("\n")
    tail_lines = tail.count("\n")
    omitted = len(lines) - head_lines - tail_lines

    truncated = (
        f"{head}\n"
        f"... [{omitted} lines truncated] ...\n"
        f"{tail}"
    )
    return truncated, True


async def execute_shell(
    command: str,
    timeout: int = 30,
) -> ToolResult:
    """Execute a shell command and return structured results."""
    # Clamp timeout
    timeout = max(1, min(timeout, 300))

    # Redact sudo -S from log output to avoid leaking password plumbing
    sudo_pw = _get_sudo_password()
    log_cmd = command
    if sudo_pw and _SUDO_RE.search(command):
        log_cmd = _SUDO_RE.sub("sudo -S", command, count=1)
    logger.info(f"Executing: {log_cmd} (timeout={timeout}s)")

    # Filter sensitive environment variables before passing to subprocess
    env = {k: v for k, v in os.environ.items() if k not in _SENSITIVE_ENV_VARS}
    env["LC_ALL"] = "C"  # Consistent output for parsing

    try:
        run_kwargs: dict = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
            "env": env,
            "cwd": os.getcwd(),
            "start_new_session": True,
        }

        # If we have a cached sudo password and the command uses sudo,
        # add -S so sudo reads the password from stdin.
        if sudo_pw and _SUDO_RE.search(command):
            command = _SUDO_RE.sub("sudo -S", command, count=1)
            run_kwargs["input"] = sudo_pw + "\n"
        else:
            run_kwargs["stdin"] = subprocess.DEVNULL

        # Run in thread to avoid blocking the async event loop
        result = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c", command],
            **run_kwargs,
        )

        stdout, stdout_truncated = _truncate_output(result.stdout)
        stderr, stderr_truncated = _truncate_output(result.stderr)

        # sudo -S echoes a password prompt to stderr — strip it
        if sudo_pw:
            stderr = "\n".join(
                line for line in stderr.splitlines()
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
        return ToolResult(
            output="",
            error="bash not found. Is bash installed?",
            exit_code=127,
        )
    except Exception as e:
        return ToolResult(
            output="",
            error=f"Failed to execute command: {type(e).__name__}: {e}",
            exit_code=1,
        )
