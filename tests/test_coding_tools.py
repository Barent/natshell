"""Tests for the coding tools: edit_file and run_code."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from natshell.tools.edit_file import edit_file
from natshell.tools.run_code import run_code


# ─── edit_file ────────────────────────────────────────────────────────────────


class TestEditFile:
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
        finally:
            os.unlink(path)

    async def test_does_not_create_directories(self):
        """edit_file should not create parent dirs (unlike write_file)."""
        result = await edit_file("/tmp/nonexistent_dir_abc/file.txt", "old", "new")
        assert result.exit_code == 1


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
        from natshell.tools.run_code import _INTERPRETERS, _COMPILERS
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
