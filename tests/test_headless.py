"""Tests for headless mode."""

from __future__ import annotations

from unittest.mock import AsyncMock

from natshell.agent.context import SystemContext
from natshell.agent.loop import AgentLoop
from natshell.config import AgentConfig, SafetyConfig
from natshell.headless import run_headless
from natshell.inference.engine import CompletionResult, ToolCall
from natshell.safety.classifier import SafetyClassifier
from natshell.tools.registry import create_default_registry


def _make_agent(
    responses: list[CompletionResult],
    max_steps: int = 15,
    safety_mode: str = "confirm",
) -> AgentLoop:
    """Create an agent with mocked inference for headless testing."""
    engine = AsyncMock()
    engine.chat_completion = AsyncMock(side_effect=responses)
    tools = create_default_registry()
    safety_config = SafetyConfig(
        mode=safety_mode,
        always_confirm=[r"^rm\s", r"^sudo\s"],
        blocked=[r"^rm\s+-[rR]f\s+/\s*$"],
    )
    safety = SafetyClassifier(safety_config)
    agent_config = AgentConfig(max_steps=max_steps, temperature=0.3, max_tokens=2048)
    agent = AgentLoop(
        engine=engine, tools=tools, safety=safety, config=agent_config
    )
    agent.initialize(
        SystemContext(
            hostname="test", distro="Test", kernel="6.0", username="tester"
        )
    )
    return agent


# ─── Basic operation ─────────────────────────────────────────────────────────


class TestHeadlessBasic:
    async def test_text_response_to_stdout(self, capsys):
        agent = _make_agent([CompletionResult(content="Hello from headless!")])
        code = await run_headless(agent, "hi")
        assert code == 0
        captured = capsys.readouterr()
        assert "Hello from headless!" in captured.out

    async def test_error_returns_exit_code_1(self):
        agent = _make_agent([])
        agent.engine.chat_completion = AsyncMock(
            side_effect=Exception("boom")
        )
        code = await run_headless(agent, "fail")
        assert code == 1

    async def test_empty_response_error(self):
        agent = _make_agent([CompletionResult(content=None)])
        code = await run_headless(agent, "hi")
        assert code == 1

    async def test_error_with_response_still_returns_1(self, capsys):
        """Any error should return exit code 1, even if a response was also produced."""
        agent = _make_agent([])
        # Simulate an agent that yields both ERROR and RESPONSE events
        from natshell.agent.loop import AgentEvent, EventType

        async def _fake_handler(prompt, **kw):
            yield AgentEvent(type=EventType.RESPONSE, data="Got a response")
            yield AgentEvent(type=EventType.ERROR, data="But also an error")

        agent.handle_user_message = _fake_handler
        code = await run_headless(agent, "mixed")
        assert code == 1


# ─── Tool execution ─────────────────────────────────────────────────────────


class TestHeadlessToolExecution:
    async def test_safe_tool_executes(self, capsys):
        agent = _make_agent([
            CompletionResult(tool_calls=[
                ToolCall(id="1", name="execute_shell", arguments={"command": "echo test"})
            ]),
            CompletionResult(content="Command ran successfully."),
        ])
        code = await run_headless(agent, "run echo test")
        assert code == 0
        captured = capsys.readouterr()
        assert "Command ran successfully." in captured.out

    async def test_tool_output_to_stderr(self, capsys):
        agent = _make_agent([
            CompletionResult(tool_calls=[
                ToolCall(id="1", name="execute_shell", arguments={"command": "echo hi"})
            ]),
            CompletionResult(content="Done."),
        ])
        code = await run_headless(agent, "run it")
        assert code == 0
        captured = capsys.readouterr()
        assert "hi" in captured.err  # Tool output goes to stderr


# ─── Confirmation handling ───────────────────────────────────────────────────


class TestHeadlessConfirmation:
    async def test_auto_decline_by_default(self, capsys):
        """Without --danger-fast, confirmations are auto-declined."""
        agent = _make_agent([
            CompletionResult(tool_calls=[
                ToolCall(
                    id="1", name="execute_shell",
                    arguments={"command": "rm important.txt"},
                )
            ]),
            CompletionResult(content="OK, I won't do that."),
        ])
        code = await run_headless(agent, "delete files", auto_approve=False)
        assert code == 0
        captured = capsys.readouterr()
        assert "declined" in captured.err.lower()

    async def test_auto_approve_with_flag(self, capsys):
        """With auto_approve=True, confirmations are accepted."""
        agent = _make_agent(
            [
                CompletionResult(tool_calls=[
                    ToolCall(
                        id="1", name="execute_shell",
                        arguments={"command": "rm tempfile.txt"},
                    )
                ]),
                CompletionResult(content="File removed."),
            ],
            safety_mode="confirm",
        )
        code = await run_headless(agent, "delete temp", auto_approve=True)
        assert code == 0
        captured = capsys.readouterr()
        assert "auto-approved" in captured.err.lower()


# ─── Blocked commands ────────────────────────────────────────────────────────


class TestHeadlessBlocked:
    async def test_blocked_command_reported(self, capsys):
        agent = _make_agent([
            CompletionResult(tool_calls=[
                ToolCall(
                    id="1", name="execute_shell",
                    arguments={"command": "rm -rf /"},
                )
            ]),
            CompletionResult(content="That was blocked."),
        ])
        await run_headless(agent, "destroy everything")
        captured = capsys.readouterr()
        assert "BLOCKED" in captured.err


# ─── Multi-step ──────────────────────────────────────────────────────────────


class TestHeadlessMultiStep:
    async def test_multi_step_produces_stats(self, capsys):
        agent = _make_agent([
            CompletionResult(tool_calls=[
                ToolCall(id="1", name="execute_shell", arguments={"command": "echo a"})
            ]),
            CompletionResult(tool_calls=[
                ToolCall(id="2", name="execute_shell", arguments={"command": "echo b"})
            ]),
            CompletionResult(content="Both done."),
        ])
        code = await run_headless(agent, "do two things")
        assert code == 0
        captured = capsys.readouterr()
        assert "Both done." in captured.out
        assert "steps" in captured.err.lower()


# ─── Planning output ─────────────────────────────────────────────────────────


class TestHeadlessPlanning:
    async def test_planning_text_to_stderr(self, capsys):
        agent = _make_agent([
            CompletionResult(
                content="Let me check...",
                tool_calls=[
                    ToolCall(id="1", name="execute_shell", arguments={"command": "ls"})
                ],
            ),
            CompletionResult(content="Here are the files."),
        ])
        code = await run_headless(agent, "list files")
        assert code == 0
        captured = capsys.readouterr()
        assert "Let me check" in captured.err
        assert "Here are the files." in captured.out
