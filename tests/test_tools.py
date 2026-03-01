"""Tests for NatShell tools."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from natshell.tools.execute_shell import _min_timeout_for, _truncate_output, execute_shell
from natshell.tools.list_directory import list_directory
from natshell.tools.read_file import read_file
from natshell.tools.registry import create_default_registry
from natshell.tools.search_files import search_files
from natshell.tools.write_file import write_file


@pytest.fixture(autouse=True)
def _reset_tool_limits():
    yield
    from natshell.tools.execute_shell import reset_limits as reset_shell
    from natshell.tools.read_file import reset_limits as reset_read

    reset_shell()
    reset_read()


# ─── execute_shell ───────────────────────────────────────────────────────────


class TestExecuteShell:
    async def test_simple_echo(self):
        result = await execute_shell("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.output

    async def test_exit_code(self):
        result = await execute_shell("false")
        assert result.exit_code != 0

    async def test_stderr(self):
        result = await execute_shell("echo err >&2")
        assert "err" in result.error

    async def test_timeout(self):
        result = await execute_shell("sleep 10", timeout=1)
        assert result.exit_code == 124
        assert "timed out" in result.error.lower()

    async def test_timeout_clamped_to_max(self):
        # Should not raise; timeout gets clamped to 300
        result = await execute_shell("echo ok", timeout=9999)
        assert result.exit_code == 0

    async def test_environment_lc_all(self):
        result = await execute_shell("echo $LC_ALL")
        assert result.output.strip() == "C"

    async def test_multiline_output(self):
        result = await execute_shell("seq 1 5")
        lines = result.output.strip().splitlines()
        assert len(lines) == 5

    async def test_output_truncation(self):
        # Generate large output
        result = await execute_shell("seq 1 100000")
        assert result.truncated

    async def test_truncate_function_short(self):
        text = "short text"
        out, truncated = _truncate_output(text)
        assert out == text
        assert not truncated

    async def test_truncate_function_long(self):
        text = "x\n" * 10000
        out, truncated = _truncate_output(text)
        assert truncated
        assert "truncated" in out


# ─── auto-timeout ─────────────────────────────────────────────────────────────


class TestAutoTimeout:
    def test_nmap(self):
        assert _min_timeout_for("nmap -sn 192.168.1.0/24") == 120

    def test_apt_install(self):
        assert _min_timeout_for("apt install foo") == 300

    def test_make(self):
        assert _min_timeout_for("make -j4") == 300

    def test_echo_no_match(self):
        assert _min_timeout_for("echo hello") == 0

    def test_find_root(self):
        assert _min_timeout_for("find / -name foo") == 120

    def test_ls_no_match(self):
        assert _min_timeout_for("ls -la") == 0


# ─── read_file ───────────────────────────────────────────────────────────────


class TestReadFile:
    async def test_read_existing_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line1\nline2\nline3\n")
            path = f.name
        try:
            result = await read_file(path)
            assert result.exit_code == 0
            assert "line1" in result.output
            assert "line2" in result.output
        finally:
            os.unlink(path)

    async def test_read_missing_file(self):
        result = await read_file("/nonexistent/file.txt")
        assert result.exit_code == 1
        assert "not found" in result.error.lower()

    async def test_read_truncation(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for i in range(500):
                f.write(f"line {i}\n")
            path = f.name
        try:
            result = await read_file(path, max_lines=10)
            assert result.truncated
            assert "more lines" in result.output
        finally:
            os.unlink(path)

    async def test_read_directory_fails(self):
        result = await read_file("/tmp")
        assert result.exit_code == 1
        assert "not a file" in result.error.lower()


# ─── write_file ──────────────────────────────────────────────────────────────


class TestWriteFile:
    async def test_write_new_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.txt")
            result = await write_file(path, "hello world")
            assert result.exit_code == 0
            assert Path(path).read_text() == "hello world"

    async def test_write_creates_directories(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "dir", "test.txt")
            result = await write_file(path, "nested")
            assert result.exit_code == 0
            assert Path(path).read_text() == "nested"

    async def test_write_overwrite(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("old content")
            path = f.name
        try:
            result = await write_file(path, "new content", mode="overwrite")
            assert result.exit_code == 0
            assert Path(path).read_text() == "new content"
        finally:
            os.unlink(path)

    async def test_write_append(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("first")
            path = f.name
        try:
            result = await write_file(path, " second", mode="append")
            assert result.exit_code == 0
            assert Path(path).read_text() == "first second"
        finally:
            os.unlink(path)


# ─── list_directory ──────────────────────────────────────────────────────────


class TestListDirectory:
    async def test_list_current_dir(self):
        result = await list_directory(".")
        assert result.exit_code == 0
        assert "Directory:" in result.output

    async def test_list_nonexistent(self):
        result = await list_directory("/nonexistent/dir")
        assert result.exit_code == 1

    async def test_list_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, ".hidden").touch()
            Path(d, "visible").touch()

            result_no_hidden = await list_directory(d, show_hidden=False)
            assert ".hidden" not in result_no_hidden.output
            assert "visible" in result_no_hidden.output

            result_hidden = await list_directory(d, show_hidden=True)
            assert ".hidden" in result_hidden.output

    async def test_list_shows_dirs_and_files(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "subdir").mkdir()
            Path(d, "file.txt").write_text("test")

            result = await list_directory(d)
            assert "subdir" in result.output
            assert "file.txt" in result.output


# ─── search_files ────────────────────────────────────────────────────────────


class TestSearchFiles:
    async def test_text_search(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "test.txt").write_text("hello world\nfoo bar\n")
            result = await search_files("hello", path=d)
            assert result.exit_code == 0
            assert "hello" in result.output

    async def test_no_match(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "test.txt").write_text("nothing here\n")
            result = await search_files("zzzznotfound", path=d)
            assert "no matches" in result.output.lower()

    async def test_file_pattern_filter(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "code.py").write_text("import os\n")
            Path(d, "readme.md").write_text("import os\n")
            result = await search_files("import", path=d, file_pattern="*.py")
            assert "code.py" in result.output
            assert "readme.md" not in result.output


# ─── registry ────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_create_default_registry(self):
        registry = create_default_registry()
        assert "execute_shell" in registry.tool_names
        assert "read_file" in registry.tool_names
        assert "write_file" in registry.tool_names
        assert "edit_file" in registry.tool_names
        assert "list_directory" in registry.tool_names
        assert "search_files" in registry.tool_names
        assert "run_code" in registry.tool_names
        assert "natshell_help" in registry.tool_names

    def test_get_tool_schemas(self):
        registry = create_default_registry()
        schemas = registry.get_tool_schemas()
        assert len(schemas) == 8
        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "parameters" in schema["function"]

    async def test_execute_unknown_tool(self):
        registry = create_default_registry()
        result = await registry.execute("nonexistent_tool", {})
        assert result.exit_code == 1
        assert "unknown tool" in result.error.lower()

    async def test_execute_via_registry(self):
        registry = create_default_registry()
        result = await registry.execute("execute_shell", {"command": "echo registry_test"})
        assert result.exit_code == 0
        assert "registry_test" in result.output

    async def test_remap_wrong_arg_name(self):
        """LLM sends wrong arg name (e.g. 'param' instead of 'topic')."""
        registry = create_default_registry()
        # natshell_help expects {"topic": "..."}, but send {"param": "..."}
        result = await registry.execute("natshell_help", {"param": "commands"})
        assert result.exit_code == 0
        assert "slash commands" in result.output.lower() or "/help" in result.output

    async def test_remap_wrong_arg_name_multiple_params(self):
        """Remap works when value count matches schema property count."""
        registry = create_default_registry()
        # search_files expects {"path": ..., "pattern": ..., ...}
        # Send wrong names but right count won't match (more required), so it
        # should fail gracefully
        result = await registry.execute("search_files", {"wrong": "value"})
        assert result.exit_code == 1
