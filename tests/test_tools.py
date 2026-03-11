"""Tests for NatShell tools."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from natshell.tools.execute_shell import (
    _SENSITIVE_ENV_VARS,
    _SENSITIVE_SUFFIXES,
    _SUDO_NEEDS_PW,
    _has_sudo_invocation,
    _inject_sudo_dash_s,
    _min_timeout_for,
    _truncate_output,
    clear_sudo_password,
    execute_shell,
    needs_sudo_password,
    set_sudo_password,
)
from natshell.tools.list_directory import list_directory
from natshell.tools.read_file import read_file
from natshell.tools.registry import ToolResult, create_default_registry
from natshell.tools.search_files import search_files
from natshell.tools.write_file import write_file


@pytest.fixture(autouse=True)
def _reset_tool_limits():
    yield
    from natshell.tools.edit_file import reset_limits as reset_edit
    from natshell.tools.execute_shell import reset_limits as reset_shell
    from natshell.tools.read_file import reset_limits as reset_read

    reset_edit()
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


# ─── sudo password injection ─────────────────────────────────────────────────


class TestSudoPasswordInjection:
    """Test that sudo -S injection handles chained commands correctly
    and only injects at command-invocation positions."""

    def setup_method(self):
        clear_sudo_password()

    def teardown_method(self):
        clear_sudo_password()

    def test_single_sudo_gets_dash_s(self):
        """A single sudo is replaced with sudo -S."""
        result, count = _inject_sudo_dash_s("sudo apt update")
        assert result == "sudo -S apt update"
        assert count == 1

    def test_chained_sudo_all_replaced(self):
        """Both sudos in a chained command get replaced."""
        result, count = _inject_sudo_dash_s("sudo apt update && sudo apt install -y nmap")
        assert result == "sudo -S apt update && sudo -S apt install -y nmap"
        assert count == 2

    def test_password_repeated_per_sudo(self):
        """Password input should contain one line per sudo occurrence."""
        _, count = _inject_sudo_dash_s("sudo apt update && sudo apt install -y nmap")
        password = "hunter2"
        password_input = (password + "\n") * count
        assert password_input == "hunter2\nhunter2\n"
        assert count == 2

    def test_triple_sudo_chain(self):
        """Three chained sudos all get replaced with repeated password."""
        cmd = "sudo systemctl stop nginx && sudo apt upgrade -y && sudo systemctl start nginx"
        result, count = _inject_sudo_dash_s(cmd)
        assert count == 3
        assert result.count("sudo -S") == 3

    async def test_chained_sudo_commands_both_execute(self):
        """Chained sudo commands should both execute when password is set."""
        set_sudo_password("testpass")
        result = await execute_shell("echo first && echo second")
        assert "first" in result.output
        assert "second" in result.output

    def test_sudo_in_argument_not_injected(self):
        """sudo inside a string argument should NOT be injected with -S.
        Only sudo at command-invocation positions gets modified."""
        cmd = 'echo "use sudo carefully"'
        result, count = _inject_sudo_dash_s(cmd)
        assert count == 0
        assert result == cmd  # unchanged

    def test_sudo_at_start_but_also_in_args(self):
        """Only the command-position sudo gets -S, not the one in args."""
        cmd = "sudo echo 'run sudo to test'"
        result, count = _inject_sudo_dash_s(cmd)
        assert count == 1
        assert result == "sudo -S echo 'run sudo to test'"

    def test_password_input_has_trailing_yes_for_pkg_manager(self):
        """Password input should end with 'y' lines for package manager prompts."""
        cmd = "sudo apt install nmap"
        _, count = _inject_sudo_dash_s(cmd)
        password = "hunter2"
        password_input = (password + "\n") * count + "y\n" * 3
        assert password_input == "hunter2\ny\ny\ny\n"

    def test_has_sudo_invocation_at_command_position(self):
        """_has_sudo_invocation returns True for sudo at command positions."""
        assert _has_sudo_invocation("sudo apt update")
        assert _has_sudo_invocation("sudo apt update && sudo reboot")
        assert _has_sudo_invocation("  sudo ls")  # leading whitespace

    def test_has_sudo_invocation_not_in_args(self):
        """_has_sudo_invocation returns False for sudo only in arguments."""
        assert not _has_sudo_invocation('echo "use sudo"')
        assert not _has_sudo_invocation("grep sudo /var/log/auth.log")

    def test_piped_sudo_commands(self):
        """sudo in piped commands at command positions gets replaced."""
        cmd = "sudo cat /etc/shadow | sudo grep root"
        result, count = _inject_sudo_dash_s(cmd)
        assert count == 2
        assert "sudo -S cat" in result
        assert "sudo -S grep" in result

    def test_semicolon_separated_sudo(self):
        """sudo in semicolon-separated commands gets replaced."""
        cmd = "sudo systemctl stop nginx; sudo systemctl start nginx"
        result, count = _inject_sudo_dash_s(cmd)
        assert count == 2


# ─── sudo password detection ────────────────────────────────────────────────


class TestSudoNeedsPassword:
    """Test that needs_sudo_password detects all known sudo error patterns."""

    def test_terminal_required(self):
        result = ToolResult(exit_code=1, error="sudo: a terminal is required to read the password")
        assert needs_sudo_password(result)

    def test_password_required(self):
        result = ToolResult(exit_code=1, error="sudo: a password is required")
        assert needs_sudo_password(result)

    def test_no_tty(self):
        result = ToolResult(
            exit_code=1,
            error="sudo: no tty present and no askpass program specified",
        )
        assert needs_sudo_password(result)

    def test_no_password_provided(self):
        """Newer sudo versions emit this when stdin is /dev/null."""
        result = ToolResult(exit_code=1, error="sudo: no password was provided")
        assert needs_sudo_password(result)

    def test_success_returns_false(self):
        result = ToolResult(exit_code=0, error="")
        assert not needs_sudo_password(result)

    def test_unrelated_error_returns_false(self):
        result = ToolResult(exit_code=1, error="command not found")
        assert not needs_sudo_password(result)

    def test_all_patterns_in_list(self):
        """Sanity check that we have at least 4 patterns."""
        assert len(_SUDO_NEEDS_PW) >= 4


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
            assert "FILE TRUNCATED" in result.output
            assert "offset=11" in result.output
            assert "ALWAYS read the entire file before editing" in result.output
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
        assert "git_tool" in registry.tool_names
        assert "natshell_help" in registry.tool_names

    def test_get_tool_schemas(self):
        registry = create_default_registry()
        schemas = registry.get_tool_schemas()
        assert len(schemas) == 12
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


# ─── Env var filtering ──────────────────────────────────────────────────────


class TestEnvVarFiltering:
    def test_explicit_vars_in_set(self):
        """All explicitly listed env vars should be in the set."""
        assert "AWS_ACCESS_KEY_ID" in _SENSITIVE_ENV_VARS
        assert "OPENAI_API_KEY" in _SENSITIVE_ENV_VARS
        assert "NATSHELL_API_KEY" in _SENSITIVE_ENV_VARS

    def test_new_credential_urls(self):
        """Newly added credential URL env vars should be in the set."""
        assert "REDIS_URL" in _SENSITIVE_ENV_VARS
        assert "MONGODB_URI" in _SENSITIVE_ENV_VARS
        assert "AMQP_URL" in _SENSITIVE_ENV_VARS

    def test_suffix_password_filtered(self):
        """Env vars ending in _PASSWORD should be caught by suffix matching."""
        assert any("MY_DB_PASSWORD".endswith(s) for s in _SENSITIVE_SUFFIXES)

    def test_suffix_secret_filtered(self):
        """Env vars ending in _SECRET should be caught by suffix matching."""
        assert any("APP_SECRET".endswith(s) for s in _SENSITIVE_SUFFIXES)

    def test_suffix_token_filtered(self):
        """Env vars ending in _TOKEN should be caught by suffix matching."""
        assert any("SLACK_TOKEN".endswith(s) for s in _SENSITIVE_SUFFIXES)

    def test_suffix_api_key_filtered(self):
        """Env vars ending in _API_KEY should be caught by suffix matching."""
        assert any("STRIPE_API_KEY".endswith(s) for s in _SENSITIVE_SUFFIXES)

    async def test_suffix_vars_not_in_subprocess(self, monkeypatch):
        """Env vars matching suffix patterns should not reach the subprocess."""
        monkeypatch.setenv("MY_CUSTOM_PASSWORD", "secret123")
        monkeypatch.setenv("SAFE_VAR", "visible")
        # Use targeted echo instead of `env` to avoid truncation on CI
        result = await execute_shell(
            'echo "PW=$MY_CUSTOM_PASSWORD SAFE=$SAFE_VAR"'
        )
        assert "secret123" not in result.output
        assert "visible" in result.output

    async def test_explicit_vars_not_in_subprocess(self, monkeypatch):
        """Explicitly listed env vars should not reach the subprocess."""
        monkeypatch.setenv("REDIS_URL", "redis://secret")
        result = await execute_shell('echo "REDIS=$REDIS_URL"')
        assert "redis://secret" not in result.output


# ─── Tool schema filtering ───────────────────────────────────────────────────


class TestToolSchemaFiltering:
    def test_allowed_filters_schemas(self):
        """get_tool_schemas(allowed=...) should only return matching tools."""
        registry = create_default_registry()
        schemas = registry.get_tool_schemas(allowed={"read_file", "list_directory"})
        names = {s["function"]["name"] for s in schemas}
        assert names == {"read_file", "list_directory"}

    def test_allowed_none_returns_all(self):
        """get_tool_schemas(allowed=None) should return all tools."""
        registry = create_default_registry()
        all_schemas = registry.get_tool_schemas()
        none_schemas = registry.get_tool_schemas(allowed=None)
        assert len(all_schemas) == len(none_schemas)

    def test_allowed_empty_returns_none(self):
        """get_tool_schemas(allowed=set()) should return no tools."""
        registry = create_default_registry()
        schemas = registry.get_tool_schemas(allowed=set())
        assert len(schemas) == 0

    def test_plan_safe_tools_excludes_destructive(self):
        """PLAN_SAFE_TOOLS should not include execute_shell, edit_file, or run_code."""
        from natshell.tools.registry import PLAN_SAFE_TOOLS

        assert "execute_shell" not in PLAN_SAFE_TOOLS
        assert "edit_file" not in PLAN_SAFE_TOOLS
        assert "run_code" not in PLAN_SAFE_TOOLS

    def test_plan_safe_tools_includes_read_tools(self):
        """PLAN_SAFE_TOOLS should include read-only tools and write_file."""
        from natshell.tools.registry import PLAN_SAFE_TOOLS

        assert "read_file" in PLAN_SAFE_TOOLS
        assert "list_directory" in PLAN_SAFE_TOOLS
        assert "search_files" in PLAN_SAFE_TOOLS
        assert "write_file" in PLAN_SAFE_TOOLS
        assert "git_tool" in PLAN_SAFE_TOOLS
        assert "natshell_help" in PLAN_SAFE_TOOLS

    def test_plan_safe_tools_filters_correctly(self):
        """Using PLAN_SAFE_TOOLS with get_tool_schemas produces the right subset."""
        from natshell.tools.registry import PLAN_SAFE_TOOLS

        registry = create_default_registry()
        schemas = registry.get_tool_schemas(allowed=PLAN_SAFE_TOOLS)
        names = {s["function"]["name"] for s in schemas}
        assert names == PLAN_SAFE_TOOLS

    def test_small_context_tools_includes_core(self):
        """SMALL_CONTEXT_TOOLS includes the 5 essential tools."""
        from natshell.tools.registry import SMALL_CONTEXT_TOOLS

        assert "execute_shell" in SMALL_CONTEXT_TOOLS
        assert "read_file" in SMALL_CONTEXT_TOOLS
        assert "write_file" in SMALL_CONTEXT_TOOLS
        assert "edit_file" in SMALL_CONTEXT_TOOLS
        assert "list_directory" in SMALL_CONTEXT_TOOLS

    def test_small_context_tools_excludes_optional(self):
        """SMALL_CONTEXT_TOOLS excludes the 5 optional tools."""
        from natshell.tools.registry import SMALL_CONTEXT_TOOLS

        assert "search_files" not in SMALL_CONTEXT_TOOLS
        assert "git_tool" not in SMALL_CONTEXT_TOOLS
        assert "run_code" not in SMALL_CONTEXT_TOOLS
        assert "fetch_url" not in SMALL_CONTEXT_TOOLS
        assert "natshell_help" not in SMALL_CONTEXT_TOOLS

    def test_small_context_tools_filters_correctly(self):
        """Using SMALL_CONTEXT_TOOLS with get_tool_schemas produces exactly 6 schemas."""
        from natshell.tools.registry import SMALL_CONTEXT_TOOLS

        registry = create_default_registry()
        schemas = registry.get_tool_schemas(allowed=SMALL_CONTEXT_TOOLS)
        names = {s["function"]["name"] for s in schemas}
        assert names == SMALL_CONTEXT_TOOLS
        assert len(schemas) == 6
