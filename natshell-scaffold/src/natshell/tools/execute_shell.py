"""Execute shell commands — the primary tool for system interaction."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from natshell.tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# Maximum characters of output to return to the model
MAX_OUTPUT_CHARS = 4000
HEAD_CHARS = 2000
TAIL_CHARS = 1500

DEFINITION = ToolDefinition(
    name="execute_shell",
    description=(
        "Execute a shell command on the user's Linux system and return the output. "
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

    logger.info(f"Executing: {command} (timeout={timeout}s)")

    env = os.environ.copy()
    env["LC_ALL"] = "C"  # Consistent output for parsing

    try:
        # Run in thread to avoid blocking the async event loop
        result = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=os.getcwd(),
        )

        stdout, stdout_truncated = _truncate_output(result.stdout)
        stderr, stderr_truncated = _truncate_output(result.stderr)

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
