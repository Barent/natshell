"""List directory contents tool."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from natshell.tools.registry import ToolDefinition, ToolResult

DEFINITION = ToolDefinition(
    name="list_directory",
    description=(
        "List the contents of a directory with file sizes and types. "
        "More structured than raw ls output. Shows file type, size, and name. "
        "Runs as the current user — if the directory requires elevated privileges, "
        "use execute_shell with sudo instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list. Default: current directory.",
            },
            "show_hidden": {
                "type": "boolean",
                "description": "Include hidden files (dotfiles). Default false.",
            },
            "max_entries": {
                "type": "integer",
                "description": "Maximum entries to return. Default 100.",
            },
        },
        "required": [],
    },
)


def _human_size(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f}{unit}" if unit != "B" else f"{size}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


async def list_directory(
    path: str = ".", show_hidden: bool = False, max_entries: int = 100
) -> ToolResult:
    """List directory contents with metadata."""
    target = Path(path).expanduser().resolve()

    if not target.exists():
        return ToolResult(error=f"Directory not found: {target}", exit_code=1)
    if not target.is_dir():
        return ToolResult(error=f"Not a directory: {target}", exit_code=1)

    try:
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = [f"Directory: {target}\n"]
        count = 0
        for entry in entries:
            if not show_hidden and entry.name.startswith("."):
                continue
            if count >= max_entries:
                remaining = sum(1 for _ in entries) - count
                lines.append(f"... and more entries")
                break

            try:
                st = entry.stat()
                kind = "d" if entry.is_dir() else "l" if entry.is_symlink() else "f"
                size = _human_size(st.st_size) if not entry.is_dir() else "-"
                lines.append(f"  {kind}  {size:>8s}  {entry.name}")
            except OSError:
                lines.append(f"  ?  {'?':>8s}  {entry.name}")
            count += 1

        return ToolResult(output="\n".join(lines))
    except PermissionError:
        return ToolResult(
            error=(
                f"Permission denied: {target}. "
                "This tool runs as the current user. To access this directory, "
                "use execute_shell with sudo (e.g. sudo ls)."
            ),
            exit_code=1,
        )
