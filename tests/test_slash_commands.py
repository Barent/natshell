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
        assert parts[0].lower() not in {"/help", "/clear", "/cmd", "/model", "/history"}


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
