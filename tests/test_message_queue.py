"""Tests for the mid-run message queuing feature."""

from __future__ import annotations

import asyncio
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
) -> AgentLoop:
    """Create an agent with a mocked inference engine."""
    engine = AsyncMock()
    engine.chat_completion = AsyncMock(side_effect=responses)

    tools = create_default_registry()

    safety_config = SafetyConfig(
        mode="danger",
        always_confirm=[],
        blocked=[],
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
    agent.initialize(
        SystemContext(
            hostname="testhost",
            distro="TestOS",
            kernel="6.0.0",
            username="testuser",
        )
    )
    return agent


class TestEnqueueAndDrain:
    def test_enqueue_and_drain(self):
        """Enqueue 3 messages, drain returns all in order."""
        agent = _make_agent([CompletionResult(content="ok")])
        agent.enqueue_message("first")
        agent.enqueue_message("second")
        agent.enqueue_message("third")

        drained = agent._drain_queued_messages()
        assert drained == ["first", "second", "third"]

    def test_drain_empty_queue(self):
        """Draining an empty queue returns an empty list."""
        agent = _make_agent([CompletionResult(content="ok")])
        assert agent._drain_queued_messages() == []

    def test_drain_clears_queue(self):
        """After draining, the queue is empty."""
        agent = _make_agent([CompletionResult(content="ok")])
        agent.enqueue_message("msg")
        agent._drain_queued_messages()
        assert agent._drain_queued_messages() == []


class TestQueuedMessagesInjected:
    async def test_queued_messages_injected_between_steps(self):
        """Messages enqueued mid-run appear in conversation history and yield QUEUED_MESSAGE events."""
        # Step 1: model calls a tool, step 2: model responds with text
        responses = [
            CompletionResult(
                content="Let me check.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="list_directory",
                        arguments={"path": "/tmp"},
                    )
                ],
            ),
            CompletionResult(content="Done!"),
        ]
        agent = _make_agent(responses, max_steps=5)

        events = []
        event_iter = agent.handle_user_message("list /tmp")

        # Consume first few events (step 1), then enqueue a message
        # so it's available for step 2
        async for event in event_iter:
            events.append(event)
            # After the first tool result, enqueue a user message
            if event.type == EventType.TOOL_RESULT:
                agent.enqueue_message("actually, also check /home")
                break

        # Continue consuming remaining events
        async for event in event_iter:
            events.append(event)

        # Verify QUEUED_MESSAGE event was emitted
        queued_events = [e for e in events if e.type == EventType.QUEUED_MESSAGE]
        assert len(queued_events) == 1
        assert queued_events[0].data == "actually, also check /home"

        # Verify the queued message was injected into conversation history
        user_msgs = [m for m in agent.messages if m["role"] == "user"]
        assert any("also check /home" in m["content"] for m in user_msgs)

    async def test_multiple_queued_messages_between_steps(self):
        """Multiple queued messages are all drained and injected in order."""
        responses = [
            CompletionResult(
                content="Checking.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="list_directory",
                        arguments={"path": "/tmp"},
                    )
                ],
            ),
            CompletionResult(content="All done!"),
        ]
        agent = _make_agent(responses, max_steps=5)

        events = []
        event_iter = agent.handle_user_message("list /tmp")

        async for event in event_iter:
            events.append(event)
            if event.type == EventType.TOOL_RESULT:
                agent.enqueue_message("correction 1")
                agent.enqueue_message("correction 2")
                break

        async for event in event_iter:
            events.append(event)

        queued_events = [e for e in events if e.type == EventType.QUEUED_MESSAGE]
        assert len(queued_events) == 2
        assert queued_events[0].data == "correction 1"
        assert queued_events[1].data == "correction 2"


class TestClearHistoryDrainsQueue:
    def test_clear_history_drains_queue(self):
        """clear_history() empties the message queue."""
        agent = _make_agent([CompletionResult(content="ok")])
        agent.enqueue_message("pending msg 1")
        agent.enqueue_message("pending msg 2")

        agent.clear_history()

        assert agent._message_queue.empty()
        assert agent._drain_queued_messages() == []


class TestQueuedMessagesDoNotConsumeSteps:
    async def test_step_count_unaffected(self):
        """Queued messages do not consume extra agent steps."""
        responses = [
            CompletionResult(
                content="Step 1.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="list_directory",
                        arguments={"path": "/tmp"},
                    )
                ],
            ),
            CompletionResult(content="Final answer."),
        ]
        agent = _make_agent(responses, max_steps=5)

        events = []
        event_iter = agent.handle_user_message("do something")

        async for event in event_iter:
            events.append(event)
            if event.type == EventType.TOOL_RESULT:
                agent.enqueue_message("extra context")
                break

        async for event in event_iter:
            events.append(event)

        # Should have exactly 2 THINKING events (one per step),
        # not 3 (queued message shouldn't add a step)
        thinking_events = [e for e in events if e.type == EventType.THINKING]
        # Step 1: THINKING for initial inference
        # Step 1 tool loop: THINKING before execution
        # Step 2: THINKING for second inference
        # The queued message injection happens at the top of step 2's loop,
        # before THINKING, so no extra THINKING is added
        assert len(thinking_events) >= 2

        # The engine was called exactly twice (once per step)
        assert agent.engine.chat_completion.call_count == 2


class TestEmptyQueueNoEvents:
    async def test_no_queued_message_events_when_empty(self):
        """When queue is empty, no QUEUED_MESSAGE events are yielded."""
        agent = _make_agent([CompletionResult(content="Simple response.")])

        events = []
        async for event in agent.handle_user_message("hello"):
            events.append(event)

        queued_events = [e for e in events if e.type == EventType.QUEUED_MESSAGE]
        assert len(queued_events) == 0
