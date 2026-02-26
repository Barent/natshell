"""Read file contents tool."""

from __future__ import annotations

import os
from pathlib import Path
from natshell.tools.registry import ToolDefinition, ToolResult

DEFINITION = ToolDefinition(
    name="read_file",
    description=(
        "Read the contents of a file on the filesystem. Returns the text content. "
        "Useful for inspecting configuration files, logs, scripts, and code. "
        "Large files are truncated to max_lines from the start."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file to read.",
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of lines to return. Default 200.",
            },
        },
        "required": ["path"],
    },
)


async def read_file(path: str, max_lines: int = 200) -> ToolResult:
    """Read a file and return its contents."""
    target = Path(path).expanduser().resolve()

    if not target.exists():
        return ToolResult(error=f"File not found: {target}", exit_code=1)
    if not target.is_file():
        return ToolResult(error=f"Not a file: {target}", exit_code=1)
    if not os.access(target, os.R_OK):
        return ToolResult(error=f"Permission denied: {target}", exit_code=1)

    try:
        lines = target.read_text(errors="replace").splitlines()
        truncated = len(lines) > max_lines
        content = "\n".join(lines[:max_lines])
        if truncated:
            content += f"\n... [{len(lines) - max_lines} more lines]"
        return ToolResult(output=content, truncated=truncated)
    except Exception as e:
        return ToolResult(error=f"Error reading file: {e}", exit_code=1)
