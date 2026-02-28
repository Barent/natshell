"""Run code snippets — write temp file, execute, clean up in one step."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from natshell.tools.execute_shell import _SENSITIVE_ENV_VARS, _truncate_output
from natshell.tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# Interpreted languages: command, file extension
_INTERPRETERS: dict[str, tuple[str, str]] = {
    "python": ("python3", ".py"),
    "javascript": ("node", ".js"),
    "bash": ("bash", ".sh"),
    "ruby": ("ruby", ".rb"),
    "perl": ("perl", ".pl"),
    "php": ("php", ".php"),
}

# Compiled languages: compiler command, file extension, extra flags
_COMPILERS: dict[str, tuple[str, str, list[str]]] = {
    "c": ("gcc", ".c", ["-o"]),
    "cpp": ("g++", ".cpp", ["-o"]),
    "rust": ("rustc", ".rs", ["-o"]),
}

# Special cases handled inline: go ("go run")

DEFINITION = ToolDefinition(
    name="run_code",
    description=(
        "Execute a code snippet and return its output. Supports interpreted "
        "languages (python, javascript, bash, ruby, perl, php), compiled "
        "languages (c, cpp, rust), and go. Handles temp file creation, "
        "compilation (if needed), execution, and cleanup automatically."
    ),
    parameters={
        "type": "object",
        "properties": {
            "language": {
                "type": "string",
                "description": (
                    "Programming language: python, javascript, bash, ruby, "
                    "perl, php, c, cpp, rust, go."
                ),
            },
            "code": {
                "type": "string",
                "description": "The source code to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": ("Maximum seconds to wait for execution. Default 30. Max 300."),
            },
        },
        "required": ["language", "code"],
    },
    requires_confirmation=True,  # Always confirm code execution
)


def _filtered_env() -> dict[str, str]:
    """Return environment with sensitive vars stripped."""
    env = {k: v for k, v in os.environ.items() if k not in _SENSITIVE_ENV_VARS}
    env["LC_ALL"] = "C"
    return env


async def run_code(language: str, code: str, timeout: int = 30) -> ToolResult:
    """Execute a code snippet and return structured results."""
    language = language.lower().strip()
    timeout = max(1, min(timeout, 300))

    # Check if language is supported
    is_interpreted = language in _INTERPRETERS
    is_compiled = language in _COMPILERS
    is_go = language == "go"

    if not (is_interpreted or is_compiled or is_go):
        supported = sorted(list(_INTERPRETERS.keys()) + list(_COMPILERS.keys()) + ["go"])
        return ToolResult(
            error=f"Unsupported language: {language}. Supported: {', '.join(supported)}",
            exit_code=1,
        )

    # Determine file extension
    if is_interpreted:
        cmd_name, ext = _INTERPRETERS[language]
    elif is_compiled:
        cmd_name, ext, _ = _COMPILERS[language]
    else:  # go
        cmd_name, ext = "go", ".go"

    # Check that the required tool is installed
    if not shutil.which(cmd_name):
        return ToolResult(
            error=f"{cmd_name} not found. Is {language} installed?",
            exit_code=1,
        )

    env = _filtered_env()
    src_file = None
    bin_file = None

    try:
        # Write code to temp file
        src_fd, src_path = tempfile.mkstemp(suffix=ext, prefix="natshell_")
        src_file = Path(src_path)
        with os.fdopen(src_fd, "w") as f:
            f.write(code)

        run_kwargs: dict = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
            "env": env,
            "cwd": os.getcwd(),
            "start_new_session": True,
            "stdin": subprocess.DEVNULL,
        }

        if is_interpreted:
            result = await asyncio.to_thread(
                subprocess.run,
                [cmd_name, src_path],
                **run_kwargs,
            )

        elif is_compiled:
            # Compile to a temp binary
            bin_fd, bin_path = tempfile.mkstemp(prefix="natshell_bin_")
            os.close(bin_fd)
            bin_file = Path(bin_path)

            _, _, flags = _COMPILERS[language]
            compile_cmd = [cmd_name, src_path] + flags + [bin_path]

            compile_result = await asyncio.to_thread(
                subprocess.run,
                compile_cmd,
                **run_kwargs,
            )

            if compile_result.returncode != 0:
                stderr, truncated = _truncate_output(compile_result.stderr)
                return ToolResult(
                    output="",
                    error=f"Compilation failed:\n{stderr}",
                    exit_code=compile_result.returncode,
                    truncated=truncated,
                )

            # Run the compiled binary
            result = await asyncio.to_thread(
                subprocess.run,
                [bin_path],
                **run_kwargs,
            )

        else:  # go
            result = await asyncio.to_thread(
                subprocess.run,
                ["go", "run", src_path],
                **run_kwargs,
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
            error=f"Execution timed out after {timeout} seconds.",
            exit_code=124,
        )
    except Exception as e:
        return ToolResult(
            output="",
            error=f"Failed to execute code: {type(e).__name__}: {e}",
            exit_code=1,
        )
    finally:
        # Clean up temp files
        if src_file and src_file.exists():
            src_file.unlink()
        if bin_file and bin_file.exists():
            bin_file.unlink()
