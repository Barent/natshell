"""Tests for the agent loop with mocked inference."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from natshell.agent.context import SystemContext
from natshell.agent.loop import AgentLoop, EventType
from natshell.config import AgentConfig, SafetyConfig
from natshell.inference.engine import CompletionResult, ToolCall
from natshell.safety.classifier import SafetyClassifier
from natshell.tools.registry import create_default_registry


def _make_agent(
    responses: list[CompletionResult],
    max_steps: int = 15,
    safety_mode: str = "confirm",
) -> AgentLoop:
    """Create an agent with a mocked inference engine that returns scripted responses."""
    engine = AsyncMock()
    engine.chat_completion = AsyncMock(side_effect=responses)

    tools = create_default_registry()

    safety_config = SafetyConfig(
        mode=safety_mode,
        always_confirm=[r"^rm\s", r"^sudo\s"],
        blocked=[r"^rm\s+-[rR]f\s+/\s*$"],
    )
    safety = SafetyClassifier(safety_config)

    agent_config = AgentConfig(
        max_steps=max_steps,
        temperature=0.3,
        max_tokens=2048,
    )

    agent = AgentLoop(
        engine=engine,
        tools=tools,
        safety=safety,
        config=agent_config,
    )
    agent.initialize(SystemContext(
        hostname="testhost",
        distro="Debian 13",
        kernel="6.12.0",
        username="testuser",
    ))
    return agent


async def _collect_events(agent: AgentLoop, message: str, confirm_callback=None):
    """Run the agent and collect all events."""
    events = []
    async for event in agent.handle_user_message(message, confirm_callback=confirm_callback):
        events.append(event)
    return events


# ─── Basic text response ─────────────────────────────────────────────────────


class TestBasicResponse:
    async def test_text_only_response(self):
        agent = _make_agent([
            CompletionResult(content="Hello! How can I help?"),
        ])
        events = await _collect_events(agent, "hi")

        types = [e.type for e in events]
        assert EventType.THINKING in types
        assert EventType.RESPONSE in types
        assert events[-1].data == "Hello! How can I help?"

    async def test_empty_response_yields_error(self):
        agent = _make_agent([
            CompletionResult(content=None),
        ])
        events = await _collect_events(agent, "hi")
        types = [e.type for e in events]
        assert EventType.ERROR in types


# ─── Single tool call cycle ──────────────────────────────────────────────────


class TestSingleToolCall:
    async def test_tool_call_then_response(self):
        agent = _make_agent([
            # Step 1: model calls execute_shell
            CompletionResult(
                tool_calls=[ToolCall(id="1", name="execute_shell", arguments={"command": "echo hi"})],
            ),
            # Step 2: model sees the result and responds
            CompletionResult(content="The command output 'hi'."),
        ])

        events = await _collect_events(agent, "run echo hi")
        types = [e.type for e in events]

        assert EventType.EXECUTING in types
        assert EventType.TOOL_RESULT in types
        assert EventType.RESPONSE in types

        # Verify the tool was actually executed
        tool_result_event = next(e for e in events if e.type == EventType.TOOL_RESULT)
        assert "hi" in tool_result_event.tool_result.output


# ─── Multi-step loop ─────────────────────────────────────────────────────────


class TestMultiStep:
    async def test_two_tool_calls_then_response(self):
        agent = _make_agent([
            CompletionResult(
                content="Let me check...",
                tool_calls=[ToolCall(id="1", name="execute_shell", arguments={"command": "uname -r"})],
            ),
            CompletionResult(
                tool_calls=[ToolCall(id="2", name="execute_shell", arguments={"command": "hostname"})],
            ),
            CompletionResult(content="You're running Linux."),
        ])

        events = await _collect_events(agent, "what system am I on?")
        types = [e.type for e in events]

        assert types.count(EventType.THINKING) == 3
        assert types.count(EventType.EXECUTING) == 2
        assert types.count(EventType.TOOL_RESULT) == 2
        assert EventType.PLANNING in types
        assert EventType.RESPONSE in types


# ─── Max steps ───────────────────────────────────────────────────────────────


class TestMaxSteps:
    async def test_max_steps_cutoff(self):
        # Model always calls a tool, never gives a text response
        responses = [
            CompletionResult(
                tool_calls=[ToolCall(id=str(i), name="execute_shell", arguments={"command": "echo loop"})],
            )
            for i in range(20)
        ]

        agent = _make_agent(responses, max_steps=3)
        events = await _collect_events(agent, "loop forever")

        types = [e.type for e in events]
        # Should end with a response about hitting max steps
        assert EventType.RESPONSE in types
        response_event = next(e for e in events if e.type == EventType.RESPONSE)
        assert "maximum" in response_event.data.lower()


# ─── Blocked commands ────────────────────────────────────────────────────────


class TestBlocked:
    async def test_blocked_command(self):
        agent = _make_agent([
            CompletionResult(
                tool_calls=[ToolCall(id="1", name="execute_shell", arguments={"command": "rm -rf /"})],
            ),
            CompletionResult(content="That command was blocked."),
        ])

        events = await _collect_events(agent, "delete everything")
        types = [e.type for e in events]
        assert EventType.BLOCKED in types


# ─── Confirmation flow ───────────────────────────────────────────────────────


class TestConfirmation:
    async def test_confirm_accepted(self):
        agent = _make_agent([
            CompletionResult(
                tool_calls=[ToolCall(id="1", name="execute_shell", arguments={"command": "rm file.txt"})],
            ),
            CompletionResult(content="File removed."),
        ])

        async def confirm_yes(tool_call):
            return True

        events = await _collect_events(agent, "remove file.txt", confirm_callback=confirm_yes)
        types = [e.type for e in events]

        assert EventType.CONFIRM_NEEDED in types
        assert EventType.EXECUTING in types
        assert EventType.TOOL_RESULT in types

    async def test_confirm_declined(self):
        agent = _make_agent([
            CompletionResult(
                tool_calls=[ToolCall(id="1", name="execute_shell", arguments={"command": "rm file.txt"})],
            ),
            CompletionResult(content="OK, I won't do that."),
        ])

        async def confirm_no(tool_call):
            return False

        events = await _collect_events(agent, "remove file.txt", confirm_callback=confirm_no)
        types = [e.type for e in events]

        assert EventType.CONFIRM_NEEDED in types
        # Should NOT have EXECUTING since user declined
        assert EventType.EXECUTING not in types


# ─── Conversation history ────────────────────────────────────────────────────


class TestConversationHistory:
    async def test_messages_accumulated(self):
        agent = _make_agent([
            CompletionResult(content="Response 1"),
        ])
        await _collect_events(agent, "first message")

        # System + user + assistant = 3 messages
        assert len(agent.messages) == 3
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[1]["role"] == "user"
        assert agent.messages[2]["role"] == "assistant"

    async def test_clear_history(self):
        agent = _make_agent([
            CompletionResult(content="Response 1"),
        ])
        await _collect_events(agent, "first message")

        agent.clear_history()
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"

    async def test_tool_exchange_in_history(self):
        agent = _make_agent([
            CompletionResult(
                tool_calls=[ToolCall(id="t1", name="execute_shell", arguments={"command": "echo test"})],
            ),
            CompletionResult(content="Done."),
        ])
        await _collect_events(agent, "run a test")

        # System + user + assistant(tool_call) + tool(result) + assistant(text) = 5
        assert len(agent.messages) == 5
        assert agent.messages[2]["role"] == "assistant"
        assert agent.messages[2].get("tool_calls") is not None
        assert agent.messages[3]["role"] == "tool"
        assert agent.messages[4]["role"] == "assistant"
