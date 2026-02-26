"""Search for text in files or find files by name."""

from __future__ import annotations

import asyncio
import subprocess
from natshell.tools.registry import ToolDefinition, ToolResult

DEFINITION = ToolDefinition(
    name="search_files",
    description=(
        "Search for text within files (grep) or find files by name pattern (find). "
        "For text search: searches recursively with line numbers. "
        "For file search: use file_pattern with glob syntax like '*.py' or '*.log'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Text pattern to search for within files (grep), or leave empty to just find files by name.",
            },
            "path": {
                "type": "string",
                "description": "Root directory to search from. Default: current directory.",
            },
            "file_pattern": {
                "type": "string",
                "description": "Glob pattern for file names to search, e.g. '*.py', '*.conf'. Default: all files.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum result lines. Default 50.",
            },
        },
        "required": ["pattern"],
    },
)


async def search_files(
    pattern: str,
    path: str = ".",
    file_pattern: str = "*",
    max_results: int = 50,
) -> ToolResult:
    """Search files by content or name."""
    try:
        if pattern:
            # Text search with grep
            cmd = ["grep", "-rn", "--include", file_pattern, "-m", str(max_results), pattern, path]
        else:
            # File name search with find
            cmd = ["find", path, "-name", file_pattern, "-maxdepth", "5"]

        result = await asyncio.to_thread(
            subprocess.run, cmd,
            capture_output=True, text=True, timeout=30,
        )

        output = result.stdout
        lines = output.splitlines()
        truncated = len(lines) > max_results
        if truncated:
            output = "\n".join(lines[:max_results]) + f"\n... [{len(lines) - max_results} more results]"

        return ToolResult(
            output=output or "(no matches found)",
            error=result.stderr if result.returncode > 1 else "",
            exit_code=0 if result.returncode <= 1 else result.returncode,
            truncated=truncated,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(error="Search timed out after 30 seconds", exit_code=124)
    except Exception as e:
        return ToolResult(error=f"Search error: {e}", exit_code=1)
