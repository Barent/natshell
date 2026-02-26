"""Write file contents tool."""

from __future__ import annotations

from pathlib import Path
from natshell.tools.registry import ToolDefinition, ToolResult

DEFINITION = ToolDefinition(
    name="write_file",
    description=(
        "Write content to a file on the filesystem. Can overwrite or append. "
        "Creates parent directories if they don't exist. "
        "Use this for creating config files, scripts, or modifying text files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to write to.",
            },
            "content": {
                "type": "string",
                "description": "The text content to write.",
            },
            "mode": {
                "type": "string",
                "enum": ["overwrite", "append"],
                "description": "Write mode. Default 'overwrite'.",
            },
        },
        "required": ["path", "content"],
    },
    requires_confirmation=True,  # Always confirm file writes
)


async def write_file(
    path: str, content: str, mode: str = "overwrite"
) -> ToolResult:
    """Write content to a file."""
    target = Path(path).expanduser().resolve()

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        file_mode = "a" if mode == "append" else "w"
        target.write_text(content) if file_mode == "w" else target.open("a").write(content)
        action = "Appended to" if mode == "append" else "Wrote"
        return ToolResult(output=f"{action} {target} ({len(content)} chars)")
    except Exception as e:
        return ToolResult(error=f"Error writing file: {e}", exit_code=1)
