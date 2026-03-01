"""Tests for the coding tools: edit_file and run_code."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from natshell.tools.edit_file import edit_file
from natshell.tools.file_tracker import get_tracker, reset_tracker
from natshell.tools.read_file import read_file
from natshell.tools.run_code import run_code

# ─── edit_file ────────────────────────────────────────────────────────────────


class TestEditFile:
    def setup_method(self):
        reset_tracker()

    def teardown_method(self):
        reset_tracker()

    async def test_successful_replacement(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world\nfoo bar\n")
            path = f.name
        try:
            result = await edit_file(path, "hello world", "goodbye world")
            assert result.exit_code == 0
            assert "Edited" in result.output
            assert "line 1" in result.output
            assert Path(path).read_text() == "goodbye world\nfoo bar\n"
            # Context snippet with line numbers
            assert "1 | goodbye world" in result.output
            assert "2 | foo bar" in result.output
        finally:
            os.unlink(path)

    async def test_old_text_not_found(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world\n")
            path = f.name
        try:
            result = await edit_file(path, "nonexistent text", "replacement")
            assert result.exit_code == 1
            assert "not found" in result.error
            assert "hello world" in result.error  # file content included
        finally:
            os.unlink(path)

    async def test_old_text_not_found_preview_200_lines(self):
        """Error preview should include up to 200 lines of file content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for i in range(300):
                f.write(f"line {i}\n")
            path = f.name
        try:
            result = await edit_file(path, "nonexistent text", "replacement")
            assert result.exit_code == 1
            # Line 199 (0-indexed) should be in the preview
            assert "line 199" in result.error
            # Line 200 (0-indexed) should NOT be in the preview
            assert "line 200" not in result.error
            # Should show remaining count
            assert "100 more lines" in result.error
        finally:
            os.unlink(path)

    async def test_old_text_matches_multiple(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("abc\nabc\nabc\n")
            path = f.name
        try:
            result = await edit_file(path, "abc", "xyz")
            assert result.exit_code == 1
            assert "3 locations" in result.error
        finally:
            os.unlink(path)

    async def test_empty_new_text_deletes(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("keep\nremove_me\nkeep\n")
            path = f.name
        try:
            result = await edit_file(path, "remove_me\n", "")
            assert result.exit_code == 0
            assert Path(path).read_text() == "keep\nkeep\n"
        finally:
            os.unlink(path)

    async def test_file_not_found(self):
        result = await edit_file("/nonexistent/file.txt", "old", "new")
        assert result.exit_code == 1
        assert "not found" in result.error.lower()

    async def test_directory_path(self):
        result = await edit_file("/tmp", "old", "new")
        assert result.exit_code == 1
        assert "not a file" in result.error.lower()

    async def test_line_number_accuracy(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line1\nline2\nline3\ntarget_line\nline5\n")
            path = f.name
        try:
            result = await edit_file(path, "target_line", "replaced_line")
            assert result.exit_code == 0
            assert "line 4" in result.output
        finally:
            os.unlink(path)

    async def test_multiline_replacement(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("start\nold1\nold2\nend\n")
            path = f.name
        try:
            result = await edit_file(path, "old1\nold2", "new1\nnew2\nnew3")
            assert result.exit_code == 0
            assert "replaced 2 lines with 3 lines" in result.output
            assert Path(path).read_text() == "start\nnew1\nnew2\nnew3\nend\n"
            # Context snippet includes surrounding lines
            assert "1 | start" in result.output
            assert "2 | new1" in result.output
            assert "5 | end" in result.output
        finally:
            os.unlink(path)

    async def test_context_snippet_capped(self):
        """Context snippet should be capped at 60 lines with omission marker."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            # 15 lines before + 1 target + 15 lines after = 31 lines
            lines = [f"before_{i}" for i in range(15)]
            lines.append("TARGET_LINE")
            lines.extend(f"after_{i}" for i in range(15))
            f.write("\n".join(lines) + "\n")
            path = f.name
        try:
            # Replace 1 line with 80 lines — total snippet would be ~100 lines
            big_replacement = "\n".join(f"new_line_{i}" for i in range(80))
            result = await edit_file(path, "TARGET_LINE", big_replacement)
            assert result.exit_code == 0
            assert "lines omitted" in result.output
            # Count actual output lines in the snippet (after the header)
            snippet_section = result.output.split("after edit]\n", 1)[1]
            snippet_lines = snippet_section.strip().splitlines()
            assert len(snippet_lines) <= 41  # 20 + 1 omission + 20
        finally:
            os.unlink(path)

    async def test_does_not_create_directories(self):
        """edit_file should not create parent dirs (unlike write_file)."""
        result = await edit_file("/tmp/nonexistent_dir_abc/file.txt", "old", "new")
        assert result.exit_code == 1

    async def test_fuzzy_match_suggestion(self):
        """When old_text is close but not exact, error should include 'Closest match'."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("def hello_world():\n    print('Hello!')\n    return True\n")
            path = f.name
        try:
            # Close but wrong — "helo_world" instead of "hello_world"
            result = await edit_file(path, "def helo_world():\n    print('Hello!')", "replaced")
            assert result.exit_code == 1
            assert "Closest match" in result.error
            assert "similar" in result.error
        finally:
            os.unlink(path)

    async def test_multiple_matches_show_line_numbers(self):
        """When old_text matches multiple locations, error should show line numbers."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("abc\ndef\nabc\nghi\nabc\n")
            path = f.name
        try:
            result = await edit_file(path, "abc", "xyz")
            assert result.exit_code == 1
            assert "3 locations" in result.error
            assert "line 1" in result.error
            assert "line 3" in result.error
            assert "line 5" in result.error
            assert "start_line/end_line" in result.error
        finally:
            os.unlink(path)

    async def test_start_end_line_targets_first_match(self):
        """start_line/end_line should target the first of two duplicate matches."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("abc\ndef\nabc\nghi\n")
            path = f.name
        try:
            result = await edit_file(path, "abc", "xyz", start_line=1, end_line=1)
            assert result.exit_code == 0
            assert Path(path).read_text() == "xyz\ndef\nabc\nghi\n"
        finally:
            os.unlink(path)

    async def test_start_end_line_targets_second_match(self):
        """start_line/end_line should target the second of two duplicate matches."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("abc\ndef\nabc\nghi\n")
            path = f.name
        try:
            result = await edit_file(path, "abc", "xyz", start_line=3, end_line=3)
            assert result.exit_code == 0
            assert Path(path).read_text() == "abc\ndef\nxyz\nghi\n"
        finally:
            os.unlink(path)

    async def test_blocked_by_partial_read(self):
        """edit_file should be blocked when file was only partially read."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world\n")
            path = f.name
        try:
            get_tracker().record_read(path, truncated=True)
            result = await edit_file(path, "hello world", "goodbye world")
            assert result.exit_code == 1
            assert "partially read" in result.error
        finally:
            os.unlink(path)

    async def test_read_truncated_then_edit_blocked(self):
        """Integration: read_file truncates → edit_file blocked."""
        # Create a file large enough to truncate at 5-line limit
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for i in range(20):
                f.write(f"line {i}\n")
            path = f.name
        try:
            result = await read_file(path, max_lines=5)
            assert result.truncated is True

            result = await edit_file(path, "line 0", "changed 0")
            assert result.exit_code == 1
            assert "partially read" in result.error
        finally:
            os.unlink(path)

    async def test_read_full_then_edit_allowed(self):
        """Integration: full read → edit allowed."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world\nfoo bar\n")
            path = f.name
        try:
            result = await read_file(path)
            assert result.truncated is False

            result = await edit_file(path, "hello world", "goodbye world")
            assert result.exit_code == 0
        finally:
            os.unlink(path)

    async def test_read_truncated_continue_then_edit_allowed(self):
        """Integration: truncated + continuation → edit allowed."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for i in range(20):
                f.write(f"line {i}\n")
            path = f.name
        try:
            # First read — truncated
            result = await read_file(path, max_lines=5)
            assert result.truncated is True

            # Continuation reads until full
            result = await read_file(path, max_lines=5, offset=6)
            result = await read_file(path, max_lines=5, offset=11)
            result = await read_file(path, max_lines=5, offset=16)
            assert result.truncated is False

            result = await edit_file(path, "line 0", "changed 0")
            assert result.exit_code == 0
        finally:
            os.unlink(path)


# ─── run_code ─────────────────────────────────────────────────────────────────


class TestRunCode:
    async def test_python_execution(self):
        result = await run_code("python", "print('hello from python')")
        assert result.exit_code == 0
        assert "hello from python" in result.output

    async def test_python_nonzero_exit(self):
        result = await run_code("python", "raise ValueError('oops')")
        assert result.exit_code != 0
        assert "ValueError" in result.error

    async def test_bash_execution(self):
        result = await run_code("bash", "echo 'hello from bash'")
        assert result.exit_code == 0
        assert "hello from bash" in result.output

    async def test_timeout_enforcement(self):
        result = await run_code("python", "import time; time.sleep(10)", timeout=1)
        assert result.exit_code == 124
        assert "timed out" in result.error.lower()

    async def test_temp_file_cleanup(self):
        """Temp files should not remain after execution."""
        import glob

        before = set(glob.glob("/tmp/natshell_*"))
        await run_code("python", "print('cleanup test')")
        after = set(glob.glob("/tmp/natshell_*"))
        # No new natshell_ temp files should remain
        assert after - before == set()

    async def test_unsupported_language(self):
        result = await run_code("brainfuck", "++++++++++.")
        assert result.exit_code == 1
        assert "unsupported language" in result.error.lower()

    async def test_language_case_insensitive(self):
        result = await run_code("Python", "print('case test')")
        assert result.exit_code == 0
        assert "case test" in result.output

    async def test_timeout_clamped_to_max(self):
        result = await run_code("python", "print('ok')", timeout=9999)
        assert result.exit_code == 0

    async def test_interpreter_mapping(self):
        """Verify key language mappings exist."""
        from natshell.tools.run_code import _COMPILERS, _INTERPRETERS

        assert "python" in _INTERPRETERS
        assert "javascript" in _INTERPRETERS
        assert "bash" in _INTERPRETERS
        assert "c" in _COMPILERS
        assert "cpp" in _COMPILERS
        assert "rust" in _COMPILERS

    async def test_python_multiline(self):
        code = "for i in range(3):\n    print(i)"
        result = await run_code("python", code)
        assert result.exit_code == 0
        assert "0" in result.output
        assert "1" in result.output
        assert "2" in result.output
