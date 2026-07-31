"""Write file contents tool."""

from __future__ import annotations

from natshell.backup import get_backup_manager
from natshell.tools.file_tracker import get_tracker
from natshell.tools.registry import ToolDefinition, ToolResult
from natshell.tools.safe_path import resolve_write_target

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


async def write_file(path: str, content: str, mode: str = "overwrite") -> ToolResult:
    """Write content to a file."""
    target, error = resolve_write_target(path)
    if target is None:
        return ToolResult(error=error, exit_code=1)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # A backup that could not be taken is reported rather than ignored: the
        # user's /undo is the only thing standing behind an overwrite.
        if target.exists() and get_backup_manager().backup(target) is None:
            return ToolResult(
                error=f"Refusing to write {target}: it exists but could not be backed up.",
                exit_code=1,
            )
        if mode == "append":
            with target.open("a") as f:
                f.write(content)
            action = "Appended to"
        else:
            target.write_text(content)
            action = "Wrote"
        # Invalidate tracker — file contents changed
        get_tracker().invalidate(str(target))
        return ToolResult(output=f"{action} {target} ({len(content)} chars)")
    except Exception as e:
        return ToolResult(error=f"Error writing file: {e}", exit_code=1)
