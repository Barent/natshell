"""Arguments must be bound to parameter names *before* they are classified.

The registry repairs a tool call whose argument names are wrong by zipping the
values onto the expected keys by position.  That repair used to happen inside
execute(), after the safety classifier had already run and passed judgement on
the unrepaired names — so the classifier read arguments.get("command", "") from
a dict with no "command" key, saw an empty string, matched nothing, and
returned SAFE.  The registry then supplied the missing name and ran it.

That defeats CONFIRM and BLOCKED alike, on one token the model gets wrong for
its own reasons rather than an attacker's.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from natshell.agent.context import SystemContext
from natshell.agent.loop import AgentLoop, EventType
from natshell.config import AgentConfig, SafetyConfig
from natshell.inference.engine import CompletionResult, ToolCall
from natshell.safety.classifier import Risk, SafetyClassifier
from natshell.tools.registry import create_default_registry

# The names small models actually confabulate, from the tool-call training data
# of whichever shell tool they saw most.
WRONG_NAMES = ["cmd", "shell_command", "bash_command", "input"]


class TestNormalizeArguments:
    def test_correct_arguments_pass_through_unchanged(self):
        registry = create_default_registry()
        args = {"command": "ls -la"}
        assert registry.normalize_arguments("execute_shell", args) == args

    @pytest.mark.parametrize("wrong", WRONG_NAMES)
    def test_wrong_name_is_bound_to_the_real_parameter(self, wrong):
        registry = create_default_registry()
        normalized = registry.normalize_arguments("execute_shell", {wrong: "ls -la"})
        assert normalized == {"command": "ls -la"}

    def test_unmappable_arguments_return_none(self):
        registry = create_default_registry()
        assert registry.normalize_arguments("execute_shell", {"a": 1, "b": 2, "c": 3}) is None

    def test_unknown_tool_is_left_alone(self):
        registry = create_default_registry()
        args = {"whatever": 1}
        assert registry.normalize_arguments("no_such_tool", args) == args


class TestClassificationSeesNormalizedArguments:
    """The property that matters: classify and execute must read the same call."""

    @pytest.mark.parametrize("wrong", WRONG_NAMES)
    def test_blocked_command_under_a_wrong_name_is_blocked(self, wrong):
        registry = create_default_registry()
        safety = SafetyClassifier(SafetyConfig(mode="confirm", always_confirm=[], blocked=[]))
        normalized = registry.normalize_arguments("execute_shell", {wrong: "rm -rf /"})
        assert safety.classify_tool_call("execute_shell", normalized) == Risk.BLOCKED

    @pytest.mark.parametrize(
        "tool,wrong_args,expected",
        [
            ("execute_shell", {"cmd": "rm notes.txt"}, Risk.CONFIRM),
            ("write_file", {"file": "/tmp/x", "text": "hi"}, Risk.CONFIRM),
            ("read_file", {"file": "/home/u/.ssh/id_rsa"}, Risk.CONFIRM),
        ],
    )
    def test_risk_bearing_tools_under_wrong_names(self, tool, wrong_args, expected):
        registry = create_default_registry()
        safety = SafetyClassifier(
            SafetyConfig(mode="confirm", always_confirm=[r"^rm\s"], blocked=[])
        )
        normalized = registry.normalize_arguments(tool, wrong_args)
        assert safety.classify_tool_call(tool, normalized) == expected

    def test_unnormalized_call_is_what_the_bug_looked_like(self):
        """Documents the defect: the raw dict really does classify SAFE."""
        safety = SafetyClassifier(SafetyConfig(mode="confirm", always_confirm=[], blocked=[]))
        assert safety.classify_tool_call("execute_shell", {"cmd": "rm -rf /"}) == Risk.SAFE


def _agent_emitting(tool_call: ToolCall) -> AgentLoop:
    engine = AsyncMock()
    engine.chat_completion = AsyncMock(
        side_effect=[
            CompletionResult(tool_calls=[tool_call]),
            CompletionResult(content="done"),
        ]
    )
    safety = SafetyClassifier(
        SafetyConfig(mode="confirm", always_confirm=[r"^rm\s"], blocked=[r"^rm\s+-[rR]f\s+/\s*$"])
    )
    agent = AgentLoop(
        engine=engine,
        tools=create_default_registry(),
        safety=safety,
        config=AgentConfig(max_steps=5, temperature=0.3, max_tokens=256),
    )
    agent.initialize(
        SystemContext(
            hostname="testhost", distro="Debian 13", kernel="6.12.0", username="testuser"
        )
    )
    return agent


async def _events(agent: AgentLoop, confirm_callback=None) -> list[EventType]:
    return [
        event.type
        async for event in agent.handle_user_message("go", confirm_callback=confirm_callback)
    ]


class TestAgentLoopNormalizesBeforeClassifying:
    async def test_blocked_command_under_a_wrong_name_never_executes(self):
        agent = _agent_emitting(
            ToolCall(id="1", name="execute_shell", arguments={"cmd": "rm -rf /"})
        )
        types = await _events(agent)
        assert EventType.BLOCKED in types
        assert EventType.EXECUTING not in types

    async def test_confirm_command_under_a_wrong_name_prompts(self):
        agent = _agent_emitting(
            ToolCall(id="1", name="execute_shell", arguments={"cmd": "rm notes.txt"})
        )
        seen: list[ToolCall] = []

        async def confirm(tool_call):
            seen.append(tool_call)
            return False

        types = await _events(agent, confirm_callback=confirm)
        assert EventType.CONFIRM_NEEDED in types
        assert EventType.EXECUTING not in types
        # The dialog must show the call as it will run, not as the model wrote it.
        assert seen and seen[0].arguments == {"command": "rm notes.txt"}

    async def test_safe_command_under_a_wrong_name_still_runs(self):
        agent = _agent_emitting(
            ToolCall(id="1", name="execute_shell", arguments={"cmd": "echo hi"})
        )
        types = await _events(agent)
        assert EventType.EXECUTING in types
        assert EventType.BLOCKED not in types
