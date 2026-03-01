"""Edit file contents via search-and-replace — surgical edits without rewriting."""

from __future__ import annotations

from pathlib import Path

from natshell.tools.registry import ToolDefinition, ToolResult

DEFINITION = ToolDefinition(
    name="edit_file",
    description=(
        "Make a targeted edit to an existing file by replacing an exact text match. "
        "The old_text must match exactly one location in the file (unique match). "
        "Use this for small, precise changes — fixing bugs, updating values, "
        "adding/removing lines. For new files, use write_file instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit.",
            },
            "old_text": {
                "type": "string",
                "description": (
                    "Exact text to find and replace. Can and should span multiple lines "
                    "when changing a block of code. Must match exactly one location in "
                    "the file. Include enough surrounding context to be unique."
                ),
            },
            "new_text": {
                "type": "string",
                "description": ("Replacement text. Use empty string to delete the matched text."),
            },
        },
        "required": ["path", "old_text", "new_text"],
    },
    requires_confirmation=True,  # Always confirm file edits
)


async def edit_file(path: str, old_text: str, new_text: str) -> ToolResult:
    """Replace a unique occurrence of old_text with new_text in a file."""
    target = Path(path).expanduser().resolve()

    if not target.exists():
        return ToolResult(error=f"File not found: {target}", exit_code=1)

    if not target.is_file():
        return ToolResult(error=f"Not a file: {target}", exit_code=1)

    try:
        content = target.read_text()
    except Exception as e:
        return ToolResult(error=f"Error reading file: {e}", exit_code=1)

    count = content.count(old_text)
    if count == 0:
        # Include file content so the model can see what's actually there
        preview_lines = content.splitlines()[:200]
        preview = "\n".join(preview_lines)
        if len(content.splitlines()) > 200:
            preview += f"\n... [{len(content.splitlines()) - 200} more lines]"
        return ToolResult(
            error=f"old_text not found in file. Current contents of {target}:\n{preview}",
            exit_code=1,
        )
    if count > 1:
        return ToolResult(
            error=f"old_text matches {count} locations (must be unique)",
            exit_code=1,
        )

    # Find line number for reporting
    line_num = content[: content.index(old_text)].count("\n") + 1

    new_content = content.replace(old_text, new_text, 1)

    try:
        target.write_text(new_content)
    except Exception as e:
        return ToolResult(error=f"Error writing file: {e}", exit_code=1)

    old_lines = old_text.count("\n") + 1
    new_lines = new_text.count("\n") + 1 if new_text else 0

    # Build context snippet: 10 lines before + new text + 10 lines after
    context_lines = new_content.splitlines()
    edit_end = line_num - 1 + new_lines  # 0-based end of new text
    snippet_start = max(0, line_num - 1 - 10)
    snippet_end = min(len(context_lines), edit_end + 10)
    snippet = context_lines[snippet_start:snippet_end]

    # Cap total snippet at 60 lines
    if len(snippet) > 60:
        keep = 20
        omitted = len(snippet) - keep * 2
        snippet = (
            snippet[:keep]
            + [f"... [{omitted} lines omitted]"]
            + snippet[-keep:]
        )

    # Format with line numbers
    numbered = "\n".join(
        f"{snippet_start + i + 1:4d} | {line}"
        for i, line in enumerate(snippet)
    )

    return ToolResult(
        output=(
            f"Edited {target} at line {line_num}:"
            f" replaced {old_lines} lines with {new_lines} lines\n\n"
            f"[lines {snippet_start + 1}-{snippet_end} after edit]\n{numbered}"
        )
    )
