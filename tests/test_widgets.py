"""Tests for widget utility functions (pure functions, no Textual app needed)."""

from __future__ import annotations

from natshell.inference.engine import ToolCall
from natshell.ui.widgets import (
    _color_diff,
    _escape,
    _format_tool_detail,
    _format_tool_summary,
)

# ─── _escape ────────────────────────────────────────────────────────────────


class TestEscape:
    def test_escapes_brackets(self):
        assert _escape("[bold]text[/]") == "\\[bold]text\\[/]"

    def test_plain_text_unchanged(self):
        assert _escape("hello world") == "hello world"

    def test_empty_string(self):
        assert _escape("") == ""


# ─── _format_tool_summary ───────────────────────────────────────────────────


class TestFormatToolSummary:
    def test_execute_shell(self):
        tc = ToolCall(id="1", name="execute_shell", arguments={"command": "ls"})
        assert _format_tool_summary(tc) == "Command:"

    def test_edit_file(self):
        tc = ToolCall(id="1", name="edit_file", arguments={"path": "/tmp/x.py"})
        result = _format_tool_summary(tc)
        assert "/tmp/x.py" in result
        assert "Edit" in result

    def test_write_file_overwrite(self):
        tc = ToolCall(id="1", name="write_file", arguments={"path": "/tmp/x"})
        assert "Write" in _format_tool_summary(tc)

    def test_write_file_append(self):
        tc = ToolCall(
            id="1", name="write_file", arguments={"path": "/tmp/x", "mode": "append"}
        )
        assert "Append" in _format_tool_summary(tc)

    def test_run_code(self):
        tc = ToolCall(
            id="1", name="run_code", arguments={"language": "python", "code": "pass"}
        )
        assert "python" in _format_tool_summary(tc)


# ─── _format_tool_detail ────────────────────────────────────────────────────


class TestFormatToolDetail:
    def test_execute_shell_shows_command(self):
        tc = ToolCall(id="1", name="execute_shell", arguments={"command": "echo hi"})
        assert "echo hi" in _format_tool_detail(tc)

    def test_edit_file_shows_diff(self):
        tc = ToolCall(
            id="1",
            name="edit_file",
            arguments={
                "path": "/tmp/x.py",
                "old_text": "value = 1\n",
                "new_text": "value = 2\n",
            },
        )
        result = _format_tool_detail(tc)
        # Should contain colored diff markers
        assert "[red]" in result
        assert "[green]" in result
        assert "value = 1" in result
        assert "value = 2" in result

    def test_edit_file_identical_shows_no_changes(self):
        tc = ToolCall(
            id="1",
            name="edit_file",
            arguments={"path": "/tmp/x.py", "old_text": "same", "new_text": "same"},
        )
        result = _format_tool_detail(tc)
        assert "no changes" in result

    def test_run_code_shows_code(self):
        tc = ToolCall(
            id="1",
            name="run_code",
            arguments={"language": "python", "code": "print(42)"},
        )
        assert "print(42)" in _format_tool_detail(tc)

    def test_write_file_new_file_shows_preview(self, tmp_path):
        path = tmp_path / "new.txt"
        tc = ToolCall(
            id="1",
            name="write_file",
            arguments={"path": str(path), "content": "hello new file"},
        )
        result = _format_tool_detail(tc)
        assert "hello new file" in result

    def test_write_file_overwrite_shows_diff(self, tmp_path):
        path = tmp_path / "existing.txt"
        path.write_text("old line\n")
        tc = ToolCall(
            id="1",
            name="write_file",
            arguments={"path": str(path), "content": "new line\n"},
        )
        result = _format_tool_detail(tc)
        assert "[red]" in result
        assert "[green]" in result

    def test_unknown_tool_shows_arguments(self):
        tc = ToolCall(id="1", name="unknown_tool", arguments={"key": "val"})
        assert "val" in _format_tool_detail(tc)


# ─── _color_diff ─────────────────────────────────────────────────────────────


class TestColorDiff:
    def test_removed_lines_red(self):
        result = _color_diff(["-removed line"])
        assert "[red]" in result

    def test_added_lines_green(self):
        result = _color_diff(["+added line"])
        assert "[green]" in result

    def test_hunk_headers_cyan(self):
        result = _color_diff(["@@ -1,3 +1,3 @@"])
        assert "[cyan]" in result

    def test_context_lines_not_colored(self):
        result = _color_diff([" context line"])
        assert "[red]" not in result
        assert "[green]" not in result
        assert "[cyan]" not in result
