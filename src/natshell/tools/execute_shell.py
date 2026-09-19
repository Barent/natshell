"""Execute shell commands — the primary tool for system interaction.

Sudo plumbing (password cache, ``sudo -S`` injection, prompt scrubbing) lives
in :mod:`natshell.tools.sudo`; the historical ``execute_shell`` names are
re-exposed below so existing imports — ``execute_shell.set_sudo_password``,
``execute_shell._has_sudo_invocation``, ``execute_shell.needs_sudo_password``,
etc. — keep working unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import subprocess
from typing import Any, Callable

from natshell.platform import is_windows
from natshell.tools import sudo
from natshell.tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# Maximum characters of output to return to the model (mutable — scaled by agent loop)
_max_output_chars = 4000
_head_chars = 2000
_tail_chars = 1500


# Never scale below this: at a tail of 0, text[-0:] returns the *entire*
# string, so the truncation would return more than an untruncated result.
_MIN_OUTPUT_CHARS = 200


def _apply_limits(max_output_chars: int) -> None:
    """Set the effective limits without touching the scaling baseline."""
    global _max_output_chars, _head_chars, _tail_chars
    max_output_chars = max(_MIN_OUTPUT_CHARS, max_output_chars)
    _max_output_chars = max_output_chars
    _head_chars = max_output_chars // 2
    _tail_chars = max(1, int(max_output_chars * 0.375))


def configure_limits(max_output_chars: int) -> None:
    """Set shell output truncation limits (called by agent loop based on context size).

    This sets the baseline that per-step scaling is computed *from*.  Step
    scaling must not come through here — see configure_step_scaling.
    """
    global _base_max_output_chars
    _apply_limits(max_output_chars)
    _base_max_output_chars = max(_MIN_OUTPUT_CHARS, max_output_chars)


def reset_limits() -> None:
    """Restore default truncation limits (used by tests)."""
    configure_limits(4000)


# Step-aware output scaling — reduces output budget as context fills up.
# This is the baseline the per-step scale factor is applied to; it is set by
# configure_limits and must stay fixed for the duration of a run.
_base_max_output_chars: int = 4000


def configure_step_scaling(step: int, max_steps: int) -> None:
    """Progressively reduce output limits as the step count increases.

    Called by the agent loop at the start of each step.  The scale factor
    drops linearly from 1.0 (step 0) to 0.3 (step == max_steps), so later
    tool results occupy less context and leave room for the model to reason.

    The scale is always applied to the fixed baseline.  It used to be applied
    via configure_limits, which reassigned the baseline to the just-scaled
    value, so the factor compounded every step: a normal 15-step run decayed
    the budget from 4000 to 1 rather than to 1200.
    """
    if max_steps <= 0:
        return
    scale = max(0.3, 1.0 - 0.7 * (step / max_steps))
    _apply_limits(int(_base_max_output_chars * scale))

# ── Sudo plumbing ──────────────────────────────────────────────────────────
#
# The helpers that used to live in this module (password cache, ``sudo -S``
# injection, needs-password detection, prompt scrub) moved to
# :mod:`natshell.tools.sudo` (R1 "SudoHandler" unit).  The aliases below keep
# every historical ``execute_shell`` import path working — callers such as
# ``agent/sudo_retry.py``, ``app.py`` and the test-suite reach for
# ``execute_shell.set_sudo_password`` and friends — while the two shell
# run-paths here call the shared implementations directly via ``sudo.*`` so
# blocking and streaming can never drift.
SUDO = sudo.SUDO
SudoHandler = sudo.SudoHandler  # re-export for historical import paths
set_sudo_password = sudo.set_password
clear_sudo_password = sudo.clear_password
needs_sudo_password = sudo.needs_password
_SUDO_NEEDS_PW = sudo._SUDO_NEEDS_PW
_PKG_MANAGER_RE = sudo._PKG_MANAGER_RE
_SUDO_PW_TIMEOUT = sudo._SUDO_PW_TIMEOUT
_get_sudo_password = sudo.get_password
_has_sudo_invocation = sudo.has_invocation
_inject_sudo_dash_s = sudo.inject_dash_s
_prepare_sudo = sudo.prepare_for_run
_scrub_sudo_prompt = sudo.scrub_prompt

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


def _filtered_env() -> dict[str, str]:
    """The environment passed to executed commands, with all sensitive
    variables removed.  Shared by the blocking and streaming paths so the
    two can never drift."""
    return {
        k: v
        for k, v in os.environ.items()
        if k not in _SENSITIVE_ENV_VARS
        and not any(k.endswith(s) for s in _SENSITIVE_SUFFIXES)
    }


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


def _effective_timeout(command: str, timeout: int) -> int:
    """Coerce, clamp and auto-raise a requested timeout for ``command``.

    Shared by the blocking and streaming paths so their timeout behaviour is
    byte-identical (including the ``1 <= t <= 300`` clamp and the long-running
    minimums from :data:`_LONG_RUNNING_PATTERNS`).
    """
    try:
        timeout = int(timeout)
    except (TypeError, ValueError):
        timeout = 60
    timeout = max(1, min(timeout, 300))
    # Auto-raise timeout for known long-running commands
    min_timeout = _min_timeout_for(command)
    if min_timeout > timeout:
        logger.info(
            "Auto-raised timeout %ds → %ds for long-running command",
            timeout,
            min_timeout,
        )
        timeout = min_timeout
    return max(1, min(timeout, 300))  # re-clamp after auto-raise


def _truncate_output(text: str) -> tuple[str, bool]:
    """Truncate output to fit in context window, preserving head and tail."""
    if len(text) <= _max_output_chars:
        return text, False

    lines = text.splitlines()
    head = text[:_head_chars]
    # Guard the slice directly as well as at the limit-setting sites:
    # text[-0:] is the whole string, which would make "truncation" return
    # more than the untruncated output rather than less.
    tail = text[-_tail_chars:] if _tail_chars > 0 else ""

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
    timeout = _effective_timeout(command, timeout)

    # Redact sudo -S from log output to avoid leaking password plumbing
    sudo_pw = _get_sudo_password()
    log_cmd = command
    if sudo_pw and _has_sudo_invocation(command):
        log_cmd, _ = _inject_sudo_dash_s(command)
    logger.info(f"Executing: {log_cmd} (timeout={timeout}s)")

    # Filter sensitive environment variables before passing to subprocess
    env = _filtered_env()
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

        # Sudo password injection (see _prepare_sudo for the rules; it is
        # shared with stream_execute_shell so the two paths stay in lockstep).
        command, stdin_data = _prepare_sudo(command)
        if stdin_data is not None:
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
        stderr = _scrub_sudo_prompt(stderr, sudo_pw)

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


async def stream_execute_shell(
    command: str,
    on_chunk: Callable[[str], None],
    timeout: int = 60,
) -> ToolResult:
    """Execute a shell command the same way as :func:`execute_shell`, but
    forward each stdout chunk to ``on_chunk`` as it arrives (R2-4).

    This is *not* a looser re-implementation: it reuses the very same helpers
    the blocking path uses — :func:`_effective_timeout`, :func:`_filtered_env`,
    :func:`_prepare_sudo` and :func:`_scrub_sudo_prompt` — so timeout
    behaviour, env filtering, sudo ``-S`` injection and the ``y\\n`` rule, and
    stderr scrubbing are byte-identical to :func:`execute_shell`.

    The process runs via ``asyncio.create_subprocess_exec`` (invoking
    ``bash -c`` on POSIX, or ``powershell -Command`` on Windows — the same
    argv the blocking path builds).  stdout is pumped in 4 KiB chunks to
    ``on_chunk``; stderr is drained to a buffer in the background (so a
    full stderr pipe can never dead-lock the run); on timeout the child's
    whole session is killed and the classic exit-124 shape is returned.

    Args:
        command:  The bash command to execute.
        timeout:  Maximum seconds (clamped/auto-raised exactly as the
            blocking path).
        on_chunk: Synchronous callback invoked for each stdout chunk as it
            arrives.  It must be safe to call from the running event loop
            and must not await; it is invoked before the final
            :class:`ToolResult` is built.

    Returns:
        A :class:`ToolResult` with stdout/stderr truncated, scrubbed and
        capped exactly the way :func:`execute_shell` would produce them.
    """
    timeout = _effective_timeout(command, timeout)

    # Redact sudo -S from log output to avoid leaking password plumbing
    sudo_pw = _get_sudo_password()
    log_cmd = command
    if sudo_pw and _has_sudo_invocation(command):
        log_cmd, _ = _inject_sudo_dash_s(command)
    logger.info(f"Streaming: {log_cmd} (timeout={timeout}s)")

    env = _filtered_env()
    env["LC_ALL"] = "C"  # Consistent output for parsing

    command, stdin_data = _prepare_sudo(command)

    # Build the shell command list based on platform (mirrors execute_shell)
    if is_windows():
        shell_cmd = [
            "powershell", "-NoProfile", "-NonInteractive",
            "-Command", command,
        ]
    else:
        shell_cmd = ["bash", "-c", command]

    stdin_arg: Any
    if stdin_data is not None:
        # The child will read the sudo password / package-manager answers
        # from stdin (exactly what the blocking path's ``input=`` supplied).
        stdin_arg = asyncio.subprocess.PIPE
    else:
        # Same as the blocking path: no password injection → stdin /dev/null.
        stdin_arg = subprocess.DEVNULL

    try:
        if is_windows():
            proc = await asyncio.create_subprocess_exec(
                *shell_cmd,
                env=env,
                cwd=os.getcwd(),
                stdin=stdin_arg,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                ),
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *shell_cmd,
                env=env,
                cwd=os.getcwd(),
                stdin=stdin_arg,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
    except FileNotFoundError:
        shell_name = "PowerShell" if is_windows() else "bash"
        return ToolResult(
            output="",
            error=f"{shell_name} not found. Is it installed?",
            exit_code=127,
        )

    # stderr must be drained concurrently with stdout — a child that fills
    # its stderr pipe while we're reading stdout would deadlock otherwise.
    assert proc.stderr is not None
    stderr_task = asyncio.create_task(_read_to_end(proc.stderr))

    stdout = bytearray()
    exit_code: int | None = None
    timed_out = False
    try:
        # Hand the sudo password / ``y\n`` answers to the child right after
        # spawn so ``sudo -S`` and the package-manager prompt both see them
        # before we drain stdout.  The payload is a handful of lines and is
        # buffered in the pipe, so the write can never block.
        if proc.stdin is not None and stdin_data is not None:
            try:
                proc.stdin.write(stdin_data.encode("utf-8"))
                await proc.stdin.drain()
                proc.stdin.close()
            except (ConnectionResetError, BrokenPipeError):
                pass

        # Pump stdout in chunks, forwarding each as it arrives.
        assert proc.stdout is not None
        try:
            async with asyncio.timeout(timeout):
                while True:
                    chunk = await proc.stdout.read(4096)
                    if not chunk:
                        break
                    stdout.extend(chunk)
                    on_chunk(chunk.decode("utf-8", errors="replace"))
                exit_code = await proc.wait()
        except TimeoutError:
            timed_out = True
    except Exception as e:
        _kill_tree(proc)
        stderr_bytes = await _collect_stderr(stderr_task)
        return ToolResult(
            output="",
            error=f"Failed to execute command: {type(e).__name__}: {e}",
            exit_code=1,
        )

    if timed_out:
        # Kill the whole session so child-of-child survives nothing, then
        # mirror the blocking path's timeout shape exactly.
        _kill_tree(proc)
        try:
            async with asyncio.timeout(5):
                await proc.wait()
        except (TimeoutError, ProcessLookupError):
            pass
    stderr_bytes = await _collect_stderr(stderr_task)
    if timed_out:
        return ToolResult(
            output="",
            error=f"Command timed out after {timeout} seconds.",
            exit_code=124,
        )

    # Same post-processing as execute_shell: truncate head+tail, scrub the
    # sudo password prompt from stderr, report the truncated flag.
    out = stdout.decode("utf-8", errors="replace")
    err = stderr_bytes.decode("utf-8", errors="replace")

    out_text, stdout_truncated = _truncate_output(out)
    err_text, stderr_truncated = _truncate_output(err)
    err_text = _scrub_sudo_prompt(err_text, sudo_pw)

    return ToolResult(
        output=out_text,
        error=err_text,
        exit_code=exit_code if exit_code is not None else 0,
        truncated=stdout_truncated or stderr_truncated,
    )


# ── Streaming-path process helpers (R2-4) ────────────────────────────────────


async def _read_to_end(stream: asyncio.StreamReader) -> bytes:
    """Read a subprocess pipe to EOF, returning everything it contained.

    Run as a task so stderr never blocks on us reading stdout (a child that
    fills its stderr pipe first would otherwise deadlock the run).
    """
    data = await stream.read()
    return data or b""


async def _collect_stderr(stderr_task: asyncio.Task) -> bytes:
    """Await a stderr-drain task, tolerating a dead child on the way out."""
    try:
        return await stderr_task
    except (
        asyncio.CancelledError,
        ProcessLookupError,
        OSError,
        subprocess.SubprocessError,
    ):
        return b""


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Best-effort termination of a spawned process (and its job on Windows).

    POSIX children run in their own session (``start_new_session=True``, the
    same isolation the blocking path gets via ``subprocess.run``), so killing
    the process group on timeout mirrors ``subprocess.run``'s kill — which
    only ever kills the direct child but does so as far as its pipes are
    concerned.  We keep it simple and robust: SIGKILL the process, then the
    process group, swallowing "already dead" noise.
    """
    if proc is None:
        return
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        return
    if not is_windows():
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError, TypeError):
            pass
