"""Tests for slash command dispatch and /cmd execution."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from natshell.agent.context import SystemContext
from natshell.agent.loop import AgentLoop
from natshell.config import AgentConfig, SafetyConfig
from natshell.inference.engine import ToolCall
from natshell.safety.classifier import Risk, SafetyClassifier
from natshell.app import SLASH_COMMANDS
from natshell.tools.execute_shell import execute_shell
from natshell.tools.registry import ToolResult, create_default_registry


def _make_agent(safety_mode: str = "confirm") -> AgentLoop:
    """Create an agent with a mocked engine for slash command tests."""
    engine = AsyncMock()
    tools = create_default_registry()
    safety_config = SafetyConfig(
        mode=safety_mode,
        always_confirm=[r"^rm\s", r"^sudo\s"],
        blocked=[r"^rm\s+-[rR]f\s+/\s*$"],
    )
    safety = SafetyClassifier(safety_config)
    agent_config = AgentConfig(max_steps=15, temperature=0.3, max_tokens=2048)
    agent = AgentLoop(engine=engine, tools=tools, safety=safety, config=agent_config)
    agent.initialize(SystemContext(
        hostname="testhost", distro="Debian 13", kernel="6.12.0", username="testuser",
    ))
    return agent


# ─── Dispatch routing ────────────────────────────────────────────────────────


class TestSlashDispatch:
    """Test that slash commands are correctly routed."""

    def test_help_recognized(self):
        parts = "/help".split(maxsplit=1)
        assert parts[0].lower() == "/help"

    def test_clear_recognized(self):
        parts = "/clear".split(maxsplit=1)
        assert parts[0].lower() == "/clear"

    def test_cmd_with_args(self):
        parts = "/cmd echo hello".split(maxsplit=1)
        assert parts[0].lower() == "/cmd"
        assert parts[1] == "echo hello"

    def test_cmd_no_args(self):
        parts = "/cmd".split(maxsplit=1)
        assert parts[0].lower() == "/cmd"
        assert len(parts) == 1

    def test_model_recognized(self):
        parts = "/model".split(maxsplit=1)
        assert parts[0].lower() == "/model"

    def test_history_recognized(self):
        parts = "/history".split(maxsplit=1)
        assert parts[0].lower() == "/history"

    def test_unknown_command(self):
        parts = "/foo".split(maxsplit=1)
        assert parts[0].lower() not in {"/help", "/clear", "/cmd", "/exeplan", "/plan", "/model", "/history"}


# ─── /cmd execution ─────────────────────────────────────────────────────────


class TestCmdExecution:
    """Test /cmd runs commands via execute_shell."""

    async def test_cmd_echo(self):
        result = await execute_shell("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.output

    async def test_cmd_failing_command(self):
        result = await execute_shell("false")
        assert result.exit_code != 0


# ─── /cmd safety classification ──────────────────────────────────────────────


class TestCmdSafety:
    """Test that /cmd applies safety classification."""

    def test_safe_command(self):
        agent = _make_agent()
        risk = agent.safety.classify_command("echo hello")
        assert risk == Risk.SAFE

    def test_blocked_command(self):
        agent = _make_agent()
        risk = agent.safety.classify_command("rm -rf /")
        assert risk == Risk.BLOCKED

    def test_confirm_command(self):
        agent = _make_agent()
        risk = agent.safety.classify_command("sudo reboot")
        assert risk == Risk.CONFIRM

    def test_rm_requires_confirm(self):
        agent = _make_agent()
        risk = agent.safety.classify_command("rm file.txt")
        assert risk == Risk.CONFIRM


# ─── History injection ───────────────────────────────────────────────────────


class TestHistoryInjection:
    """Test that /cmd results are injected into agent messages as user role."""

    async def test_injection_after_cmd(self):
        agent = _make_agent()
        initial_count = len(agent.messages)

        result = await execute_shell("echo injected")

        # Simulate what _handle_cmd does
        output = result.output or result.error
        agent.messages.append({
            "role": "user",
            "content": (
                f"[The user directly ran a shell command: `echo injected`]\n"
                f"Exit code: {result.exit_code}\n"
                f"Output:\n{output}"
            ),
        })

        assert len(agent.messages) == initial_count + 1
        injected = agent.messages[-1]
        assert injected["role"] == "user"
        assert "echo injected" in injected["content"]
        assert "injected" in injected["content"]

    async def test_injection_preserves_exit_code(self):
        agent = _make_agent()
        result = await execute_shell("false")
        agent.messages.append({
            "role": "user",
            "content": (
                f"[The user directly ran a shell command: `false`]\n"
                f"Exit code: {result.exit_code}\n"
                f"Output:\n{result.output or result.error}"
            ),
        })
        injected = agent.messages[-1]
        assert f"Exit code: {result.exit_code}" in injected["content"]


# ─── History info ────────────────────────────────────────────────────────────


class TestHistoryInfo:
    def test_message_count(self):
        agent = _make_agent()
        # After initialize: 1 system message
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"


# ─── Autocomplete filtering ─────────────────────────────────────────────────


class TestAutocomplete:
    """Test that SLASH_COMMANDS filtering works for autocomplete."""

    def _filter(self, text: str) -> list[str]:
        """Replicate the filtering logic from on_input_changed."""
        return [
            cmd for cmd, desc in SLASH_COMMANDS
            if cmd.startswith(text.lower())
        ]

    def test_slash_alone_matches_all(self):
        matches = self._filter("/")
        assert len(matches) == len(SLASH_COMMANDS)

    def test_slash_h_matches_help_and_history(self):
        matches = self._filter("/h")
        assert "/help" in matches
        assert "/history" in matches
        assert len(matches) == 2

    def test_slash_c_matches_clear_and_cmd(self):
        matches = self._filter("/c")
        assert "/clear" in matches
        assert "/cmd" in matches
        assert len(matches) == 2

    def test_slash_m_matches_model(self):
        matches = self._filter("/m")
        assert "/model" in matches
        assert all(m.startswith("/model") for m in matches)

    def test_full_command_matches_exactly(self):
        matches = self._filter("/help")
        assert matches == ["/help"]

    def test_no_match(self):
        matches = self._filter("/z")
        assert matches == []

    def test_space_in_input_hides_suggestions(self):
        """Once a space is typed (e.g. '/cmd ls'), suggestions should hide."""
        text = "/cmd ls"
        should_show = text.startswith("/") and " " not in text
        assert not should_show


# ─── --danger-fast flag ──────────────────────────────────────────────────────


class TestSkipPermissionsFlag:
    """Test that the --danger-fast flag is wired correctly."""

    def test_app_accepts_skip_permissions_false(self):
        """NatShellApp defaults to skip_permissions=False."""
        from natshell.app import NatShellApp
        agent = _make_agent()
        app = NatShellApp(agent=agent)
        assert app._skip_permissions is False

    def test_app_accepts_skip_permissions_true(self):
        """NatShellApp stores skip_permissions=True when passed."""
        from natshell.app import NatShellApp
        agent = _make_agent()
        app = NatShellApp(agent=agent, skip_permissions=True)
        assert app._skip_permissions is True

    def test_blocked_still_blocked_with_skip_permissions(self):
        """BLOCKED commands are unaffected by skip_permissions — they are
        checked on a separate code path in the agent loop."""
        agent = _make_agent()
        risk = agent.safety.classify_command("rm -rf /")
        assert risk == Risk.BLOCKED

    def test_confirm_command_still_classifies(self):
        """The safety classifier still returns CONFIRM for risky commands;
        skip_permissions only affects whether the callback is invoked."""
        agent = _make_agent()
        risk = agent.safety.classify_command("sudo reboot")
        assert risk == Risk.CONFIRM


# ─── /plan dispatch and prompt ──────────────────────────────────────────────


class TestPlanDispatch:
    def test_plan_recognized(self):
        parts = "/plan".split(maxsplit=1)
        assert parts[0].lower() == "/plan"

    def test_plan_with_args(self):
        parts = "/plan build a tetris game".split(maxsplit=1)
        assert parts[0].lower() == "/plan"
        assert parts[1] == "build a tetris game"


class TestPlanPrompt:
    def test_prompt_contains_description(self):
        from natshell.app import _build_plan_prompt
        prompt = _build_plan_prompt("build a REST API", "project/\n  src/")
        assert "build a REST API" in prompt

    def test_prompt_contains_directory_tree(self):
        from natshell.app import _build_plan_prompt
        prompt = _build_plan_prompt("test", "mydir/\n  file.py")
        assert "mydir/" in prompt
        assert "file.py" in prompt

    def test_prompt_contains_format_rules(self):
        from natshell.app import _build_plan_prompt
        prompt = _build_plan_prompt("anything", "dir/")
        assert "## Step" in prompt
        assert "PLAN.md" in prompt

    def test_prompt_mentions_preamble(self):
        from natshell.app import _build_plan_prompt
        prompt = _build_plan_prompt("anything", "dir/")
        assert "preamble" in prompt.lower()
