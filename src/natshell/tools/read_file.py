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
        "Large files are truncated to max_lines. Use offset to start reading "
        "from a specific line number."
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
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (1-based). Default 1.",
            },
            "limit": {
                "type": "integer",
                "description": "Alias for max_lines. Maximum number of lines to return.",
            },
        },
        "required": ["path"],
    },
)


async def read_file(
    path: str,
    max_lines: int = 200,
    offset: int = 1,
    limit: int | None = None,
) -> ToolResult:
    """Read a file and return its contents."""
    # Coerce params to int (LLMs may send strings like "63" or "63.0")
    try:
        max_lines = int(float(max_lines))
    except (TypeError, ValueError):
        max_lines = 200
    try:
        offset = int(float(offset))
    except (TypeError, ValueError):
        offset = 1
    if limit is not None:
        try:
            max_lines = int(float(limit))
        except (TypeError, ValueError):
            pass

    offset = max(1, offset)  # 1-based

    target = Path(path).expanduser().resolve()

    if not target.exists():
        return ToolResult(error=f"File not found: {target}", exit_code=1)
    if not target.is_file():
        return ToolResult(error=f"Not a file: {target}", exit_code=1)
    if not os.access(target, os.R_OK):
        return ToolResult(error=f"Permission denied: {target}", exit_code=1)

    try:
        all_lines = target.read_text(errors="replace").splitlines()
        total = len(all_lines)

        # Apply offset (convert 1-based to 0-based)
        start = offset - 1
        lines = all_lines[start : start + max_lines]
        truncated = (start + max_lines) < total

        content = "\n".join(lines)
        if truncated:
            remaining = total - (start + len(lines))
            content += f"\n... [{remaining} more lines]"
        if start > 0:
            content = f"[starting at line {offset}]\n" + content
        return ToolResult(output=content, truncated=truncated)
    except Exception as e:
        return ToolResult(error=f"Error reading file: {e}", exit_code=1)
