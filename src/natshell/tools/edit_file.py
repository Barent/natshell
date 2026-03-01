"""Edit file contents via search-and-replace — surgical edits without rewriting."""

from __future__ import annotations

import difflib
from pathlib import Path

from natshell.backup import get_backup_manager
from natshell.tools.file_tracker import get_tracker
from natshell.tools.registry import ToolDefinition, ToolResult

DEFINITION = ToolDefinition(
    name="edit_file",
    description=(
        "Make a targeted edit to an existing file by replacing an exact text match. "
        "The old_text must match exactly one location in the file (unique match). "
        "Use this for small, precise changes — fixing bugs, updating values, "
        "adding/removing lines. For new files, use write_file instead. "
        "If old_text matches multiple locations, use start_line/end_line to "
        "restrict the search to a specific line range."
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
            "start_line": {
                "type": "integer",
                "description": (
                    "Optional: restrict search to lines starting from this line number (1-based). "
                    "Use when old_text matches multiple locations to target a specific one."
                ),
            },
            "end_line": {
                "type": "integer",
                "description": (
                    "Optional: restrict search to lines up to and including "
                    "this line number (1-based). Use with start_line to "
                    "target a specific region."
                ),
            },
        },
        "required": ["path", "old_text", "new_text"],
    },
    requires_confirmation=True,  # Always confirm file edits
)


def _fuzzy_match_hint(old_text: str, content: str) -> str:
    """Find the closest match to old_text in the file using SequenceMatcher.

    Slides a window of len(old_text lines) across the file lines and returns
    a hint string if similarity > 50%. Caps search at first+last 500 lines
    for files over 1000 lines.
    """
    old_lines = old_text.splitlines(keepends=True)
    file_lines = content.splitlines(keepends=True)
    window_size = len(old_lines)

    if window_size == 0 or len(file_lines) == 0:
        return ""

    # Cap search for large files
    if len(file_lines) > 1000:
        search_regions = [
            (0, file_lines[:500]),
            (len(file_lines) - 500, file_lines[-500:]),
        ]
    else:
        search_regions = [(0, file_lines)]

    best_ratio = 0.0
    best_start = 0
    best_end = 0
    best_region_offset = 0

    for region_offset, region_lines in search_regions:
        for i in range(len(region_lines) - window_size + 1):
            candidate = region_lines[i : i + window_size]
            ratio = difflib.SequenceMatcher(
                None, "".join(old_lines), "".join(candidate)
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
                best_end = i + window_size
                best_region_offset = region_offset

    if best_ratio <= 0.50:
        return ""

    abs_start = best_region_offset + best_start
    abs_end = best_region_offset + best_end
    pct = int(best_ratio * 100)

    # Build numbered preview of the closest match
    if len(content.splitlines(keepends=True)) > 1000:
        # Re-fetch from original file_lines using absolute indices
        match_lines = file_lines[abs_start:abs_end]
    else:
        match_lines = search_regions[0][1][best_start:best_end]

    numbered = "\n".join(
        f"{abs_start + i + 1:4d} | {line.rstrip()}"
        for i, line in enumerate(match_lines)
    )

    return (
        f"\n\nClosest match ({pct}% similar) at lines {abs_start + 1}-{abs_end}:\n"
        f"{numbered}"
    )


def _find_occurrence_lines(content: str, old_text: str, count: int) -> str:
    """Find line numbers of each occurrence of old_text in content."""
    lines = []
    search_from = 0
    for _ in range(count):
        idx = content.index(old_text, search_from)
        line_num = content[:idx].count("\n") + 1
        lines.append(f"line {line_num}")
        search_from = idx + 1
    return (
        f"Found at: {', '.join(lines)}. "
        "Include more surrounding context in old_text for a unique match, "
        "or use start_line/end_line to target a specific occurrence."
    )


async def edit_file(
    path: str,
    old_text: str,
    new_text: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> ToolResult:
    """Replace a unique occurrence of old_text with new_text in a file."""
    target = Path(path).expanduser().resolve()

    if not target.exists():
        return ToolResult(error=f"File not found: {target}", exit_code=1)

    if not target.is_file():
        return ToolResult(error=f"Not a file: {target}", exit_code=1)

    # File tracker gate — block edits on partially-read files
    tracker = get_tracker()
    allowed, reason = tracker.can_edit(str(target))
    if not allowed:
        return ToolResult(error=reason, exit_code=1)

    try:
        content = target.read_text()
    except Exception as e:
        return ToolResult(error=f"Error reading file: {e}", exit_code=1)

    # Coerce start_line/end_line to int (LLMs may send strings)
    if start_line is not None:
        try:
            start_line = int(float(start_line))
        except (TypeError, ValueError):
            start_line = None
    if end_line is not None:
        try:
            end_line = int(float(end_line))
        except (TypeError, ValueError):
            end_line = None

    # If start_line/end_line provided, search within that region
    if start_line is not None or end_line is not None:
        all_lines = content.splitlines(keepends=True)
        sl = max(1, start_line or 1) - 1  # convert to 0-based
        el = min(len(all_lines), end_line or len(all_lines))
        region = "".join(all_lines[sl:el])

        region_count = region.count(old_text)
        if region_count == 0:
            hint = _fuzzy_match_hint(old_text, content)
            preview_lines = content.splitlines()[:200]
            preview = "\n".join(preview_lines)
            if len(content.splitlines()) > 200:
                preview += f"\n... [{len(content.splitlines()) - 200} more lines]"
            return ToolResult(
                error=(
                    f"old_text not found in lines {sl + 1}-{el} of {target}."
                    f"{hint}\n\nCurrent contents of {target}:\n{preview}"
                ),
                exit_code=1,
            )
        if region_count > 1:
            return ToolResult(
                error=(
                    f"old_text matches {region_count} locations within lines "
                    f"{sl + 1}-{el} (must be unique). Narrow the line range or "
                    "add more context."
                ),
                exit_code=1,
            )

        # Compute absolute offset: prefix + position within region
        prefix = "".join(all_lines[:sl])
        region_idx = region.index(old_text)
        abs_idx = len(prefix) + region_idx

        # Replace at the absolute position
        new_content = content[:abs_idx] + new_text + content[abs_idx + len(old_text) :]
        line_num = content[:abs_idx].count("\n") + 1
    else:
        # Standard full-file search
        count = content.count(old_text)
        if count == 0:
            hint = _fuzzy_match_hint(old_text, content)
            preview_lines = content.splitlines()[:200]
            preview = "\n".join(preview_lines)
            if len(content.splitlines()) > 200:
                preview += f"\n... [{len(content.splitlines()) - 200} more lines]"
            return ToolResult(
                error=(
                    f"old_text not found in file."
                    f"{hint}\n\nCurrent contents of {target}:\n{preview}"
                ),
                exit_code=1,
            )
        if count > 1:
            location_info = _find_occurrence_lines(content, old_text, count)
            return ToolResult(
                error=f"old_text matches {count} locations (must be unique). {location_info}",
                exit_code=1,
            )

        line_num = content[: content.index(old_text)].count("\n") + 1
        new_content = content.replace(old_text, new_text, 1)

    try:
        get_backup_manager().backup(target)
        target.write_text(new_content)
    except Exception as e:
        return ToolResult(error=f"Error writing file: {e}", exit_code=1)

    # Invalidate tracker — file contents changed
    tracker.invalidate(str(target))

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
