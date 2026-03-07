"""Tests for headless mode."""

from __future__ import annotations

from unittest.mock import AsyncMock

from natshell.agent.context import SystemContext
from natshell.agent.loop import AgentLoop
from natshell.config import AgentConfig, SafetyConfig
from natshell.headless import run_headless, run_headless_exeplan, run_headless_plan
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


# ─── Headless plan generation ────────────────────────────────────────────────


class TestHeadlessPlan:
    async def test_plan_generation_creates_plan(self, capsys, tmp_path, monkeypatch):
        """run_headless_plan should return 0 when PLAN.md is created."""
        monkeypatch.chdir(tmp_path)

        # Simulate the agent writing PLAN.md via write_file tool
        plan_content = (
            "# Test Plan\n\nPreamble.\n\n"
            "## Step 1: Do something\n\nDetails here.\n"
        )

        async def _fake_handler(prompt, **kw):
            from natshell.agent.loop import AgentEvent, EventType

            # Simulate writing PLAN.md
            (tmp_path / "PLAN.md").write_text(plan_content)
            yield AgentEvent(type=EventType.RESPONSE, data="Plan generated.")

        agent = _make_agent([])
        agent.handle_user_message = _fake_handler

        # Mock engine_info for n_ctx
        from unittest.mock import MagicMock

        agent.engine.engine_info = MagicMock(
            return_value=MagicMock(n_ctx=4096)
        )

        code = await run_headless_plan(agent, "build something")
        assert code == 0
        captured = capsys.readouterr()
        assert "Test Plan" in captured.out
        assert "1 steps" in captured.out or "Step 1" in captured.out

    async def test_plan_generation_fails_without_file(self, capsys, tmp_path, monkeypatch):
        """run_headless_plan should return 1 when PLAN.md is not created."""
        monkeypatch.chdir(tmp_path)

        async def _fake_handler(prompt, **kw):
            from natshell.agent.loop import AgentEvent, EventType

            yield AgentEvent(type=EventType.RESPONSE, data="I described a plan.")

        agent = _make_agent([])
        agent.handle_user_message = _fake_handler

        from unittest.mock import MagicMock

        agent.engine.engine_info = MagicMock(
            return_value=MagicMock(n_ctx=4096)
        )

        code = await run_headless_plan(agent, "build something")
        assert code == 1
        captured = capsys.readouterr()
        assert "not created" in captured.err.lower()


# ─── Headless plan execution ─────────────────────────────────────────────────


class TestHeadlessExeplan:
    async def test_exeplan_executes_steps(self, capsys, tmp_path, monkeypatch):
        """run_headless_exeplan should execute all steps and report results."""
        monkeypatch.chdir(tmp_path)

        plan_file = tmp_path / "PLAN.md"
        plan_file.write_text(
            "# Test Plan\n\nPreamble.\n\n"
            "## Step 1: First step\n\nDo the first thing.\n\n"
            "## Step 2: Second step\n\nDo the second thing.\n"
        )

        call_count = 0

        async def _fake_handler(prompt, **kw):
            from natshell.agent.loop import AgentEvent, EventType

            nonlocal call_count
            call_count += 1
            yield AgentEvent(
                type=EventType.RESPONSE,
                data=f"Completed step work (call {call_count}).",
            )

        agent = _make_agent([])
        agent.handle_user_message = _fake_handler

        from unittest.mock import MagicMock

        agent.engine.engine_info = MagicMock(
            return_value=MagicMock(n_ctx=4096)
        )

        code = await run_headless_exeplan(agent, str(plan_file))
        assert code == 0
        captured = capsys.readouterr()
        assert "Passed: 2" in captured.out
        assert "Failed: 0" in captured.out

    async def test_exeplan_file_not_found(self, capsys):
        """run_headless_exeplan should return 1 for missing plan file."""
        agent = _make_agent([])
        code = await run_headless_exeplan(agent, "/nonexistent/PLAN.md")
        assert code == 1
        captured = capsys.readouterr()
        assert "error" in captured.err.lower()

    async def test_exeplan_partial_step(self, capsys, tmp_path, monkeypatch):
        """A step hitting max steps should count as failed."""
        monkeypatch.chdir(tmp_path)

        plan_file = tmp_path / "PLAN.md"
        plan_file.write_text(
            "# Test Plan\n\nPreamble.\n\n"
            "## Step 1: Only step\n\nDo something.\n"
        )

        async def _fake_handler(prompt, **kw):
            from natshell.agent.loop import AgentEvent, EventType

            yield AgentEvent(
                type=EventType.RESPONSE,
                data="Reached the maximum number of steps for this run.",
            )

        agent = _make_agent([])
        agent.handle_user_message = _fake_handler

        from unittest.mock import MagicMock

        agent.engine.engine_info = MagicMock(
            return_value=MagicMock(n_ctx=4096)
        )

        code = await run_headless_exeplan(agent, str(plan_file))
        assert code == 1
        captured = capsys.readouterr()
        assert "Failed: 1" in captured.out
