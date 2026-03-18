"""Tests for the agent loop with mocked inference."""

from __future__ import annotations

import json
import types
from unittest.mock import AsyncMock

from natshell.agent.context import SystemContext
from natshell.agent.loop import AgentLoop, EventType, _is_analysis_request, _is_plan_request
from natshell.config import AgentConfig, SafetyConfig
from natshell.inference.engine import CompletionResult, EngineInfo, ToolCall
from natshell.inference.local import (
    _MISTRAL_TOOL_CALLS_RE,
    _THINK_RE,
    _TOOL_CALL_RE,
    LocalEngine,
    _detect_model_family,
    _format_tools_for_prompt,
    _format_tools_for_prompt_mistral,
    _infer_context_size,
)
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
    agent.initialize(
        SystemContext(
            hostname="testhost",
            distro="Debian 13",
            kernel="6.12.0",
            username="testuser",
        )
    )
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
        agent = _make_agent(
            [
                CompletionResult(content="Hello! How can I help?"),
            ]
        )
        events = await _collect_events(agent, "hi")

        types = [e.type for e in events]
        assert EventType.THINKING in types
        assert EventType.RESPONSE in types
        assert events[-1].data == "Hello! How can I help?"

    async def test_empty_response_yields_error(self):
        agent = _make_agent(
            [
                CompletionResult(content=None),
            ]
        )
        events = await _collect_events(agent, "hi")
        types = [e.type for e in events]
        assert EventType.ERROR in types


# ─── Single tool call cycle ──────────────────────────────────────────────────


class TestSingleToolCall:
    async def test_tool_call_then_response(self):
        agent = _make_agent(
            [
                # Step 1: model calls execute_shell
                CompletionResult(
                    tool_calls=[
                        ToolCall(id="1", name="execute_shell", arguments={"command": "echo hi"})
                    ],
                ),
                # Step 2: model sees the result and responds
                CompletionResult(content="The command output 'hi'."),
            ]
        )

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
        agent = _make_agent(
            [
                CompletionResult(
                    content="Let me check...",
                    tool_calls=[
                        ToolCall(id="1", name="execute_shell", arguments={"command": "uname -r"})
                    ],
                ),
                CompletionResult(
                    tool_calls=[
                        ToolCall(id="2", name="execute_shell", arguments={"command": "hostname"})
                    ],
                ),
                CompletionResult(content="You're running Linux."),
            ]
        )

        events = await _collect_events(agent, "what system am I on?")
        types = [e.type for e in events]

        # 3 from loop iterations + 2 from pre-EXECUTING restarts
        assert types.count(EventType.THINKING) == 5
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
                tool_calls=[
                    ToolCall(id=str(i), name="execute_shell", arguments={"command": "echo loop"})
                ],
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


# ─── Effective max steps scaling ─────────────────────────────────────────────


class TestEffectiveMaxSteps:
    def _make_loop(self, max_steps: int = 15) -> AgentLoop:
        engine = AsyncMock()
        tools = create_default_registry()
        safety = SafetyClassifier(SafetyConfig(mode="confirm"))
        config = AgentConfig(max_steps=max_steps, temperature=0.3, max_tokens=2048)
        agent = AgentLoop(engine=engine, tools=tools, safety=safety, config=config)
        return agent

    def test_default_15_stays_15_for_small_ctx(self):
        agent = self._make_loop(max_steps=15)
        assert agent._effective_max_steps(4096) == 15

    def test_scales_to_25_for_8k_ctx(self):
        agent = self._make_loop(max_steps=15)
        assert agent._effective_max_steps(8192) == 25

    def test_scales_to_35_for_16k_ctx(self):
        agent = self._make_loop(max_steps=15)
        assert agent._effective_max_steps(16384) == 35

    def test_scales_to_50_for_32k_ctx(self):
        agent = self._make_loop(max_steps=15)
        assert agent._effective_max_steps(32768) == 50

    def test_scales_to_60_for_128k_ctx(self):
        agent = self._make_loop(max_steps=15)
        assert agent._effective_max_steps(131072) == 60

    def test_scales_to_120_for_256k_ctx(self):
        agent = self._make_loop(max_steps=15)
        assert agent._effective_max_steps(262144) == 120

    def test_scales_to_150_for_512k_ctx(self):
        agent = self._make_loop(max_steps=15)
        assert agent._effective_max_steps(524288) == 150

    def test_scales_to_200_for_1m_ctx(self):
        agent = self._make_loop(max_steps=15)
        assert agent._effective_max_steps(1048576) == 200

    def test_explicit_override_respected(self):
        """When user sets max_steps != 15, auto-scaling is disabled."""
        agent = self._make_loop(max_steps=10)
        assert agent._effective_max_steps(32768) == 10

    def test_high_override_respected(self):
        agent = self._make_loop(max_steps=50)
        assert agent._effective_max_steps(4096) == 50


# ─── Blocked commands ────────────────────────────────────────────────────────


class TestBlocked:
    async def test_blocked_command(self):
        agent = _make_agent(
            [
                CompletionResult(
                    tool_calls=[
                        ToolCall(id="1", name="execute_shell", arguments={"command": "rm -rf /"})
                    ],
                ),
                CompletionResult(content="That command was blocked."),
            ]
        )

        events = await _collect_events(agent, "delete everything")
        types = [e.type for e in events]
        assert EventType.BLOCKED in types


# ─── Confirmation flow ───────────────────────────────────────────────────────


class TestConfirmation:
    async def test_confirm_accepted(self):
        agent = _make_agent(
            [
                CompletionResult(
                    tool_calls=[
                        ToolCall(id="1", name="execute_shell", arguments={"command": "rm file.txt"})
                    ],
                ),
                CompletionResult(content="File removed."),
            ]
        )

        async def confirm_yes(tool_call):
            return True

        events = await _collect_events(agent, "remove file.txt", confirm_callback=confirm_yes)
        types = [e.type for e in events]

        assert EventType.CONFIRM_NEEDED in types
        assert EventType.EXECUTING in types
        assert EventType.TOOL_RESULT in types

    async def test_confirm_declined(self):
        agent = _make_agent(
            [
                CompletionResult(
                    tool_calls=[
                        ToolCall(id="1", name="execute_shell", arguments={"command": "rm file.txt"})
                    ],
                ),
                CompletionResult(content="OK, I won't do that."),
            ]
        )

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
        agent = _make_agent(
            [
                CompletionResult(content="Response 1"),
            ]
        )
        await _collect_events(agent, "first message")

        # System + user + assistant = 3 messages
        assert len(agent.messages) == 3
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[1]["role"] == "user"
        assert agent.messages[2]["role"] == "assistant"

    async def test_clear_history(self):
        agent = _make_agent(
            [
                CompletionResult(content="Response 1"),
            ]
        )
        await _collect_events(agent, "first message")

        agent.clear_history()
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"

    async def test_tool_exchange_in_history(self):
        agent = _make_agent(
            [
                CompletionResult(
                    tool_calls=[
                        ToolCall(id="t1", name="execute_shell", arguments={"command": "echo test"})
                    ],
                ),
                CompletionResult(content="Done."),
            ]
        )
        await _collect_events(agent, "run a test")

        # System + user + assistant(tool_call) + tool(result) + assistant(text) = 5
        assert len(agent.messages) == 5
        assert agent.messages[2]["role"] == "assistant"
        assert agent.messages[2].get("tool_calls") is not None
        assert agent.messages[3]["role"] == "tool"
        assert agent.messages[4]["role"] == "assistant"


# ─── Truncated response (finish_reason="length") ────────────────────────────


class TestTruncatedResponse:
    async def test_length_with_only_think_tags_yields_error(self):
        """When the model spends all tokens thinking, we should get an error."""
        agent = _make_agent(
            [
                CompletionResult(
                    content="<think>I need to think about this really hard...</think>",
                    finish_reason="length",
                ),
            ]
        )
        events = await _collect_events(agent, "do something")
        types = [e.type for e in events]
        assert EventType.ERROR in types
        error_event = next(e for e in events if e.type == EventType.ERROR)
        assert "truncated" in error_event.data.lower()

    async def test_length_with_empty_content_yields_error(self):
        """finish_reason=length with no content should yield an error."""
        agent = _make_agent(
            [
                CompletionResult(
                    content=None,
                    finish_reason="length",
                ),
            ]
        )
        events = await _collect_events(agent, "do something")
        types = [e.type for e in events]
        assert EventType.ERROR in types

    async def test_length_with_real_content_passes_through(self):
        """If there's actual content despite length truncation, show it."""
        agent = _make_agent(
            [
                CompletionResult(
                    content="<think></think>Here is a partial answer that got cut off",
                    finish_reason="length",
                ),
            ]
        )
        events = await _collect_events(agent, "explain something")
        types = [e.type for e in events]
        # Should be treated as a normal response, not an error
        assert EventType.RESPONSE in types


# ─── Think tag and tool_call tag parsing ─────────────────────────────────────


class TestQwen3Parsing:
    def test_think_regex_strips_think_blocks(self):
        text = "<think>some reasoning</think>Actual response"
        result = _THINK_RE.sub("", text).strip()
        assert result == "Actual response"

    def test_think_regex_strips_empty_think(self):
        text = "<think></think>Hello"
        result = _THINK_RE.sub("", text).strip()
        assert result == "Hello"

    def test_think_regex_strips_multiline_think(self):
        text = "<think>\nline1\nline2\n</think>Answer"
        result = _THINK_RE.sub("", text).strip()
        assert result == "Answer"

    def test_tool_call_regex_matches(self):
        text = '<tool_call>{"name": "execute_shell", "arguments": {"command": "ls"}}</tool_call>'
        matches = _TOOL_CALL_RE.findall(text)
        assert len(matches) == 1
        parsed = json.loads(matches[0])
        assert parsed["name"] == "execute_shell"
        assert parsed["arguments"]["command"] == "ls"

    def test_tool_call_regex_with_think_prefix(self):
        text = (
            "<think></think>"
            '<tool_call>{"name": "execute_shell", "arguments": {"command": "whoami"}}</tool_call>'
        )
        # Strip think first, then extract tool calls
        cleaned = _THINK_RE.sub("", text)
        matches = _TOOL_CALL_RE.findall(cleaned)
        assert len(matches) == 1
        parsed = json.loads(matches[0])
        assert parsed["name"] == "execute_shell"


# ─── Mistral [TOOL_CALLS] parsing ─────────────────────────────────────────────


class TestMistralParsing:
    def test_mistral_tool_calls_regex_matches(self):
        text = '[TOOL_CALLS] [{"name": "execute_shell", "arguments": {"command": "ls"}}]'
        match = _MISTRAL_TOOL_CALLS_RE.search(text)
        assert match is not None
        calls = json.loads(match.group(1))
        assert len(calls) == 1
        assert calls[0]["name"] == "execute_shell"
        assert calls[0]["arguments"]["command"] == "ls"

    def test_mistral_tool_calls_multiple_tools(self):
        text = (
            '[TOOL_CALLS] [{"name": "execute_shell", "arguments": {"command": "ls"}},'
            ' {"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}]'
        )
        match = _MISTRAL_TOOL_CALLS_RE.search(text)
        assert match is not None
        calls = json.loads(match.group(1))
        assert len(calls) == 2
        assert calls[0]["name"] == "execute_shell"
        assert calls[1]["name"] == "read_file"

    def test_mistral_tool_calls_with_text_prefix(self):
        text = (
            'Let me check that for you.\n'
            '[TOOL_CALLS] [{"name": "execute_shell", "arguments": {"command": "whoami"}}]'
        )
        match = _MISTRAL_TOOL_CALLS_RE.search(text)
        assert match is not None
        calls = json.loads(match.group(1))
        assert calls[0]["name"] == "execute_shell"

    def test_mistral_tool_calls_multiline(self):
        text = (
            '[TOOL_CALLS]\n'
            '[{"name": "execute_shell", "arguments": {"command": "uname -r"}}]'
        )
        match = _MISTRAL_TOOL_CALLS_RE.search(text)
        assert match is not None
        calls = json.loads(match.group(1))
        assert calls[0]["arguments"]["command"] == "uname -r"


# ─── Model family detection ──────────────────────────────────────────────────


class TestModelFamilyDetection:
    def test_qwen3_4b(self):
        assert _detect_model_family("Qwen3-4B-Q4_K_M.gguf") == "qwen"

    def test_qwen3_8b(self):
        assert _detect_model_family("Qwen3-8B-Q4_K_M.gguf") == "qwen"

    def test_mistral_nemo(self):
        assert _detect_model_family("Mistral-Nemo-Instruct-2407-Q4_K_M.gguf") == "mistral"

    def test_unknown_model_defaults_to_qwen(self):
        assert _detect_model_family("some-random-model.gguf") == "qwen"

    def test_unknown_model_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="natshell.inference.local"):
            _detect_model_family("llama-3.2-7b.gguf")
        assert "Unknown model family" in caplog.text

    def test_qwen_model_no_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="natshell.inference.local"):
            _detect_model_family("Qwen3-4B-Q4_K_M.gguf")
        assert "Unknown model family" not in caplog.text

    def test_mistral_model_no_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="natshell.inference.local"):
            _detect_model_family("Mistral-Nemo-Instruct-2407-Q4_K_M.gguf")
        assert "Unknown model family" not in caplog.text


# ─── Context size override ───────────────────────────────────────────────────


class TestContextSizeOverride:
    def test_mistral_nemo_returns_32768(self):
        assert _infer_context_size("Mistral-Nemo-Instruct-2407-Q4_K_M.gguf") == 32768

    def test_qwen3_4b_returns_4096(self):
        assert _infer_context_size("Qwen3-4B-Q4_K_M.gguf") == 4096

    def test_qwen3_8b_returns_8192(self):
        assert _infer_context_size("Qwen3-8B-Q4_K_M.gguf") == 8192


# ─── max_tokens auto-scaling ────────────────────────────────────────────────


def _make_agent_with_ctx(
    n_ctx: int,
    max_tokens: int = 2048,
) -> AgentLoop:
    """Create an agent whose mock engine reports the given n_ctx."""
    engine = AsyncMock()
    engine.chat_completion = AsyncMock(
        return_value=CompletionResult(content="ok"),
    )
    engine.engine_info = lambda: EngineInfo(engine_type="mock", n_ctx=n_ctx)

    tools = create_default_registry()
    safety = SafetyClassifier(SafetyConfig(mode="confirm", always_confirm=[], blocked=[]))
    agent_config = AgentConfig(max_steps=15, temperature=0.3, max_tokens=max_tokens)

    agent = AgentLoop(engine=engine, tools=tools, safety=safety, config=agent_config)
    agent.initialize(
        SystemContext(
            hostname="testhost",
            distro="Test",
            kernel="6.0",
            username="tester",
        )
    )
    return agent


class TestMaxTokensScaling:
    def test_small_context_uses_25pct(self):
        """4096-token context: n_ctx ≤ 16384 → scaled = 1024 (25% of context)."""
        agent = _make_agent_with_ctx(n_ctx=4096)
        assert agent._max_tokens == 1024

    def test_large_context_scales_up(self):
        """40960-token context: max(2048, min(10240, 16384)) = 10240."""
        agent = _make_agent_with_ctx(n_ctx=40960)
        assert agent._max_tokens == 10240

    def test_very_large_context_capped(self):
        """131072-token context: max(2048, min(32768, 65536)) = 32768."""
        agent = _make_agent_with_ctx(n_ctx=131072)
        assert agent._max_tokens == 32768

    def test_256k_context_scales_to_cap(self):
        """262144-token context: max(2048, min(65536, 65536)) = 65536 (cap)."""
        agent = _make_agent_with_ctx(n_ctx=262144)
        assert agent._max_tokens == 65536

    def test_user_high_floor_capped_on_small_context(self):
        """User sets max_tokens=8192 on 4K context — capped to 25% to avoid starving budget."""
        agent = _make_agent_with_ctx(n_ctx=4096, max_tokens=8192)
        assert agent._max_tokens == 1024

    def test_user_high_floor_respected_on_large_context(self):
        """User sets max_tokens=8192 on 32K context — respected as floor."""
        agent = _make_agent_with_ctx(n_ctx=32768, max_tokens=8192)
        assert agent._max_tokens == 8192

    async def test_engine_swap_recalculates(self):
        """Swapping to an engine with a larger context should update _max_tokens."""
        agent = _make_agent_with_ctx(n_ctx=4096)
        assert agent._max_tokens == 1024

        # Swap to a bigger engine
        new_engine = AsyncMock()
        new_engine.engine_info = lambda: EngineInfo(engine_type="mock", n_ctx=40960)
        new_engine.chat_completion = AsyncMock(
            return_value=CompletionResult(content="ok"),
        )
        await agent.swap_engine(new_engine)
        assert agent._max_tokens == 10240


# ─── Shell output chars scaling ──────────────────────────────────────────────


class TestOutputCharsScaling:
    def _make_loop(self) -> AgentLoop:
        engine = AsyncMock()
        tools = create_default_registry()
        safety = SafetyClassifier(SafetyConfig(mode="confirm"))
        config = AgentConfig(max_steps=15, temperature=0.3, max_tokens=2048)
        return AgentLoop(engine=engine, tools=tools, safety=safety, config=config)

    def test_small_context_4000(self):
        agent = self._make_loop()
        assert agent._effective_max_output_chars(4096) == 4000

    def test_16k_context_8000(self):
        agent = self._make_loop()
        assert agent._effective_max_output_chars(16384) == 8000

    def test_32k_context_12000(self):
        agent = self._make_loop()
        assert agent._effective_max_output_chars(32768) == 12000

    def test_64k_context_16000(self):
        agent = self._make_loop()
        assert agent._effective_max_output_chars(65536) == 16000

    def test_128k_context_32000(self):
        agent = self._make_loop()
        assert agent._effective_max_output_chars(131072) == 32000

    def test_262k_context_64000(self):
        agent = self._make_loop()
        assert agent._effective_max_output_chars(262144) == 64000

    def test_512k_context_96000(self):
        agent = self._make_loop()
        assert agent._effective_max_output_chars(524288) == 96000

    def test_1m_context_128000(self):
        agent = self._make_loop()
        assert agent._effective_max_output_chars(1048576) == 128000


# ─── Read file lines scaling ────────────────────────────────────────────────


class TestReadFileLinesScaling:
    def _make_loop(self) -> AgentLoop:
        engine = AsyncMock()
        tools = create_default_registry()
        safety = SafetyClassifier(SafetyConfig(mode="confirm"))
        config = AgentConfig(max_steps=15, temperature=0.3, max_tokens=2048)
        return AgentLoop(engine=engine, tools=tools, safety=safety, config=config)

    def test_small_context_200(self):
        agent = self._make_loop()
        assert agent._effective_read_file_lines(4096) == 200

    def test_16k_context_500(self):
        agent = self._make_loop()
        assert agent._effective_read_file_lines(16384) == 500

    def test_32k_context_1000(self):
        agent = self._make_loop()
        assert agent._effective_read_file_lines(32768) == 1000

    def test_64k_context_2000(self):
        agent = self._make_loop()
        assert agent._effective_read_file_lines(65536) == 2000

    def test_128k_context_3000(self):
        agent = self._make_loop()
        assert agent._effective_read_file_lines(131072) == 3000

    def test_262k_context_4000(self):
        agent = self._make_loop()
        assert agent._effective_read_file_lines(262144) == 4000

    def test_512k_context_6000(self):
        agent = self._make_loop()
        assert agent._effective_read_file_lines(524288) == 6000

    def test_1m_context_8000(self):
        agent = self._make_loop()
        assert agent._effective_read_file_lines(1048576) == 8000


# ─── Edit failure completion guard ──────────────────────────────────────────


class TestEditFailureGuard:
    async def test_edit_failure_warning_injected(self):
        """When all edits fail and model tries to respond, a warning is injected."""
        import os
        import tempfile

        from natshell.tools.file_tracker import reset_tracker

        reset_tracker()

        # Create a temp file for the edit to target
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("real content\n")
            path = f.name

        try:
            agent = _make_agent(
                [
                    # Step 1: model calls edit_file with wrong old_text (will fail)
                    CompletionResult(
                        tool_calls=[
                            ToolCall(
                                id="1",
                                name="edit_file",
                                arguments={
                                    "path": path,
                                    "old_text": "nonexistent text",
                                    "new_text": "replacement",
                                },
                            )
                        ],
                    ),
                    # Step 2: model tries to declare success
                    CompletionResult(content="I've fixed the bug!"),
                    # Step 3: model responds after the warning
                    CompletionResult(content="Let me re-read and try again."),
                ],
                max_steps=5,
                safety_mode="danger",  # skip confirmation for edit_file
            )

            events = await _collect_events(agent, "fix the bug")

            # The first text response ("I've fixed the bug!") should NOT be yielded
            # because the completion guard intercepts it.
            # The second text response ("Let me re-read...") should be yielded.
            response_events = [e for e in events if e.type == EventType.RESPONSE]
            assert len(response_events) == 1
            assert "re-read" in response_events[0].data

            # Check that the warning was injected into the message history
            user_msgs = [m for m in agent.messages if m["role"] == "user"]
            warning_msgs = [m for m in user_msgs if "[SYSTEM] Warning" in m["content"]]
            assert len(warning_msgs) == 1
        finally:
            os.unlink(path)
            reset_tracker()

    async def test_successful_edit_no_warning(self):
        """When edits succeed, no warning should be injected."""
        import os
        import tempfile

        from natshell.tools.file_tracker import reset_tracker

        reset_tracker()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("old content\n")
            path = f.name

        try:
            agent = _make_agent(
                [
                    # Step 1: model calls edit_file with correct old_text
                    CompletionResult(
                        tool_calls=[
                            ToolCall(
                                id="1",
                                name="edit_file",
                                arguments={
                                    "path": path,
                                    "old_text": "old content",
                                    "new_text": "new content",
                                },
                            )
                        ],
                    ),
                    # Step 2: model declares success
                    CompletionResult(content="Done! I've updated the file."),
                ],
                max_steps=5,
                safety_mode="danger",
            )

            events = await _collect_events(agent, "update the file")

            response_events = [e for e in events if e.type == EventType.RESPONSE]
            assert len(response_events) == 1
            assert "Done" in response_events[0].data

            # No warning should have been injected
            user_msgs = [m for m in agent.messages if m["role"] == "user"]
            warning_msgs = [m for m in user_msgs if "[SYSTEM] Warning" in m.get("content", "")]
            assert len(warning_msgs) == 0
        finally:
            os.unlink(path)
            reset_tracker()


# ─── Remote engine error messages ─────────────────────────────────────────


class TestRemoteEngineErrors:
    async def test_connect_error_message(self):
        """ConnectError should produce a clear error message."""
        from natshell.inference.remote import RemoteEngine

        engine = RemoteEngine(base_url="http://localhost:19", model="test")
        try:
            await engine.chat_completion(messages=[{"role": "user", "content": "hi"}])
            assert False, "Should have raised"
        except ConnectionError as e:
            assert "Cannot connect" in str(e)
            assert "localhost:19" in str(e)
        finally:
            await engine.close()

    async def test_http_status_error_message(self):
        """HTTPStatusError should produce a clear error message with status code."""
        import httpx

        from natshell.inference.remote import RemoteEngine

        engine = RemoteEngine(base_url="http://localhost:11434", model="test")

        # Mock the client.post to raise HTTPStatusError
        mock_response = httpx.Response(
            status_code=404,
            text="model not found",
            request=httpx.Request("POST", "http://localhost:11434/chat/completions"),
        )
        engine.client.post = AsyncMock(side_effect=httpx.HTTPStatusError(
            "not found", request=mock_response.request, response=mock_response
        ))
        try:
            await engine.chat_completion(messages=[{"role": "user", "content": "hi"}])
            assert False, "Should have raised"
        except ConnectionError as e:
            assert "404" in str(e)
        finally:
            await engine.close()

    async def test_fallback_on_connection_error(self):
        """Agent should recognize ConnectionError for fallback."""
        agent = _make_agent([CompletionResult(content="ok")])
        # The _can_fallback needs a RemoteEngine and fallback config
        assert not agent._can_fallback(ConnectionError("test"))

    async def test_connection_error_yields_error_event(self):
        """ConnectionError from remote engine should yield an error event."""
        agent = _make_agent([])
        agent.engine.chat_completion = AsyncMock(
            side_effect=ConnectionError("Cannot connect to server")
        )
        events = await _collect_events(agent, "test")
        types = [e.type for e in events]
        assert EventType.ERROR in types
        error_event = next(e for e in events if e.type == EventType.ERROR)
        assert "Cannot connect" in error_event.data


# ─── Context overflow detection and recovery ────────────────────────────────


class TestContextOverflow:
    async def test_context_overflow_triggers_compaction_and_retry(self):
        """First ContextOverflowError triggers compaction, second call succeeds."""
        from natshell.inference.remote import ContextOverflowError

        # Build an agent with enough history to compact
        agent = _make_agent([])

        # Seed conversation history (system + several exchanges)
        agent.messages.append({"role": "user", "content": "first question"})
        agent.messages.append({"role": "assistant", "content": "first answer"})
        agent.messages.append({"role": "user", "content": "second question"})
        agent.messages.append({"role": "assistant", "content": "second answer"})

        # First call raises overflow, second call succeeds
        agent.engine.chat_completion = AsyncMock(
            side_effect=[
                ContextOverflowError("context length exceeded"),
                CompletionResult(content="Recovered response."),
            ]
        )

        events = await _collect_events(agent, "third question")
        types = [e.type for e in events]

        # Should have an error about compaction
        error_events = [e for e in events if e.type == EventType.ERROR]
        assert any("compacted" in e.data.lower() for e in error_events)

        # Should still get a successful response after retry
        assert EventType.RESPONSE in types
        response_event = next(e for e in events if e.type == EventType.RESPONSE)
        assert response_event.data == "Recovered response."

    async def test_context_overflow_does_not_trigger_local_fallback(self):
        """_can_fallback returns False for ContextOverflowError."""
        from natshell.inference.remote import ContextOverflowError

        agent = _make_agent([CompletionResult(content="ok")])
        assert not agent._can_fallback(ContextOverflowError("context full"))

    async def test_auth_error_does_not_trigger_local_fallback(self):
        """_can_fallback returns False for AuthenticationError."""
        from natshell.inference.remote import AuthenticationError, RemoteEngine

        agent = _make_agent([CompletionResult(content="ok")])
        # _can_fallback requires a RemoteEngine and fallback_config to even consider fallback
        agent.engine = AsyncMock(spec=RemoteEngine)
        agent.fallback_config = True  # truthy sentinel
        assert not agent._can_fallback(AuthenticationError("API key is missing or incorrect"))

    async def test_context_overflow_unrecoverable_shows_clear_message(self):
        """When conversation is too short to compact, suggest /clear."""
        from natshell.inference.remote import ContextOverflowError

        agent = _make_agent([])
        # Only system + user message — too short to compact (≤3 messages)
        agent.engine.chat_completion = AsyncMock(
            side_effect=ContextOverflowError("context length exceeded")
        )

        events = await _collect_events(agent, "short question")
        error_events = [e for e in events if e.type == EventType.ERROR]
        assert any("/clear" in e.data for e in error_events)

    async def test_context_overflow_double_failure_gives_up(self):
        """If overflow happens again after compaction, give up."""
        from natshell.inference.remote import ContextOverflowError

        agent = _make_agent([])

        # Seed enough history to allow compaction
        agent.messages.append({"role": "user", "content": "first question"})
        agent.messages.append({"role": "assistant", "content": "first answer"})
        agent.messages.append({"role": "user", "content": "second question"})
        agent.messages.append({"role": "assistant", "content": "second answer"})

        # Both calls raise overflow — compaction doesn't help
        agent.engine.chat_completion = AsyncMock(
            side_effect=ContextOverflowError("context length exceeded")
        )

        events = await _collect_events(agent, "third question")
        error_events = [e for e in events if e.type == EventType.ERROR]
        # Should get both the "compacted" message and the "still full" message
        assert any("still full" in e.data.lower() for e in error_events)


# ─── Proactive compaction on context pressure ────────────────────────────────


class TestProactiveCompaction:
    async def test_proactive_compaction_on_high_pressure(self):
        """When prompt_tokens/n_ctx > 85%, compaction fires and info event is emitted."""
        # Create agent with known n_ctx
        agent = _make_agent_with_ctx(n_ctx=4096)

        # Seed enough history to allow compaction
        agent.messages.append({"role": "user", "content": "first question"})
        agent.messages.append({"role": "assistant", "content": "first answer"})
        agent.messages.append({"role": "user", "content": "second question"})
        agent.messages.append({"role": "assistant", "content": "second answer"})

        # Return a result with prompt_tokens > 85% of n_ctx
        agent.engine.chat_completion = AsyncMock(
            return_value=CompletionResult(
                content="Response here.",
                prompt_tokens=3600,  # 3600/4096 = 87.9% > 85%
                completion_tokens=50,
            ),
        )

        events = await _collect_events(agent, "third question")
        error_events = [e for e in events if e.type == EventType.ERROR]
        assert any("compacted" in e.data.lower() for e in error_events)

    async def test_no_proactive_compaction_when_low_pressure(self):
        """When context pressure is under threshold, no compaction event."""
        agent = _make_agent_with_ctx(n_ctx=32768)

        agent.engine.chat_completion = AsyncMock(
            return_value=CompletionResult(
                content="Response here.",
                prompt_tokens=1000,  # 1000/32768 = 3% — well under threshold
                completion_tokens=50,
            ),
        )

        events = await _collect_events(agent, "test question")
        error_events = [e for e in events if e.type == EventType.ERROR]
        assert not any("compacted" in e.data.lower() for e in error_events)


# ─── Repetitive read detection ───────────────────────────────────────────────


class TestRepetitiveReadDetection:
    async def test_read_warning_after_3_reads(self):
        """Reading the same file 3 times should trigger a warning."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world\n")
            path = f.name

        try:
            agent = _make_agent(
                [
                    # 3 reads of the same file, then a text response
                    CompletionResult(
                        tool_calls=[
                            ToolCall(id="1", name="read_file", arguments={"path": path})
                        ],
                    ),
                    CompletionResult(
                        tool_calls=[
                            ToolCall(id="2", name="read_file", arguments={"path": path})
                        ],
                    ),
                    CompletionResult(
                        tool_calls=[
                            ToolCall(id="3", name="read_file", arguments={"path": path})
                        ],
                    ),
                    CompletionResult(content="Done reading."),
                ],
                max_steps=5,
                safety_mode="danger",
            )

            await _collect_events(agent, "read the file")

            # Find tool messages in the conversation history for the 3rd read
            tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
            assert len(tool_msgs) == 3
            # The 3rd read's tool message should contain the warning
            assert "3 times" in tool_msgs[2]["content"]
            assert "write_file" in tool_msgs[2]["content"]
        finally:
            os.unlink(path)

    async def test_read_count_resets_after_write(self):
        """Writing to a file should reset its read counter."""
        import os
        import tempfile

        from natshell.tools.file_tracker import reset_tracker

        reset_tracker()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("original content\n")
            path = f.name

        try:
            agent = _make_agent(
                [
                    # 2 reads, then a write, then 1 more read
                    CompletionResult(
                        tool_calls=[
                            ToolCall(id="1", name="read_file", arguments={"path": path})
                        ],
                    ),
                    CompletionResult(
                        tool_calls=[
                            ToolCall(id="2", name="read_file", arguments={"path": path})
                        ],
                    ),
                    CompletionResult(
                        tool_calls=[
                            ToolCall(
                                id="3",
                                name="write_file",
                                arguments={"path": path, "content": "new content\n"},
                            )
                        ],
                    ),
                    CompletionResult(
                        tool_calls=[
                            ToolCall(id="4", name="read_file", arguments={"path": path})
                        ],
                    ),
                    CompletionResult(content="Done."),
                ],
                max_steps=6,
                safety_mode="danger",
            )

            await _collect_events(agent, "update the file")

            # The read after the write should be count=1, so no warning
            tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
            # Last tool message is the 4th (read after write) — should NOT have warning
            assert "times" not in tool_msgs[-1]["content"]
        finally:
            os.unlink(path)
            reset_tracker()

    async def test_no_warning_for_2_reads(self):
        """Reading a file only 2 times should NOT trigger a warning."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some content\n")
            path = f.name

        try:
            agent = _make_agent(
                [
                    CompletionResult(
                        tool_calls=[
                            ToolCall(id="1", name="read_file", arguments={"path": path})
                        ],
                    ),
                    CompletionResult(
                        tool_calls=[
                            ToolCall(id="2", name="read_file", arguments={"path": path})
                        ],
                    ),
                    CompletionResult(content="Done."),
                ],
                max_steps=4,
                safety_mode="danger",
            )

            await _collect_events(agent, "read the file")

            tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
            for msg in tool_msgs:
                assert "times" not in msg["content"]
        finally:
            os.unlink(path)


# ─── Step budget awareness ───────────────────────────────────────────────────


class TestStepBudgetAwareness:
    async def test_50pct_budget_info(self):
        """At 50% of budget, info message should appear in tool messages."""
        # max_steps=10, so step 5 = 50%
        # Use unique commands to avoid duplicate tool call detection
        responses = [
            CompletionResult(
                tool_calls=[
                    ToolCall(
                        id=str(i), name="execute_shell",
                        arguments={"command": f"echo step{i}"},
                    )
                ],
            )
            for i in range(10)
        ] + [CompletionResult(content="Done.")]

        agent = _make_agent(responses, max_steps=10)
        await _collect_events(agent, "do stuff")

        # Context manager may trim older messages, so check any surviving tool msg
        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        all_content = "\n".join(m["content"] for m in tool_msgs)
        assert "steps used" in all_content

    async def test_90pct_urgent_warning(self):
        """At 90% of budget, URGENT warning should appear."""
        # Use unique commands to avoid duplicate tool call detection
        responses = [
            CompletionResult(
                tool_calls=[
                    ToolCall(
                        id=str(i), name="execute_shell",
                        arguments={"command": f"echo step{i}"},
                    )
                ],
            )
            for i in range(10)
        ] + [CompletionResult(content="Done.")]

        agent = _make_agent(responses, max_steps=10)
        await _collect_events(agent, "do stuff")

        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        all_content = "\n".join(m["content"] for m in tool_msgs)
        assert "URGENT" in all_content

    async def test_no_budget_pct_below_50pct(self):
        """Below 50% of budget, no percentage-based step info should appear."""
        responses = [
            CompletionResult(
                tool_calls=[
                    ToolCall(id="1", name="execute_shell", arguments={"command": "echo hi"})
                ],
            ),
            CompletionResult(content="Done."),
        ]

        agent = _make_agent(responses, max_steps=10)
        await _collect_events(agent, "do stuff")

        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        # Step 1 out of 10 = 10% — no "N/M steps used" info
        for msg in tool_msgs:
            assert "steps used" not in msg["content"]

    async def test_early_budget_hint_on_first_step(self):
        """First tool result should include a budget hint."""
        responses = [
            CompletionResult(
                tool_calls=[
                    ToolCall(id="1", name="execute_shell", arguments={"command": "echo hi"})
                ],
            ),
            CompletionResult(content="Done."),
        ]

        agent = _make_agent(responses, max_steps=10)
        await _collect_events(agent, "do stuff")

        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        assert len(tool_msgs) >= 1
        assert "Budget: 10 steps available" in tool_msgs[0]["content"]


# ─── Edit failure guidance ───────────────────────────────────────────────────


class TestEditFailureGuidance:
    async def test_edit_failure_suggests_write_file(self):
        """After 2 failed edits, warning should mention write_file."""
        import os
        import tempfile

        from natshell.tools.file_tracker import reset_tracker

        reset_tracker()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("real content\n")
            path = f.name

        try:
            agent = _make_agent(
                [
                    CompletionResult(
                        tool_calls=[
                            ToolCall(
                                id="1",
                                name="edit_file",
                                arguments={
                                    "path": path,
                                    "old_text": "nonexistent",
                                    "new_text": "replacement",
                                },
                            )
                        ],
                    ),
                    CompletionResult(
                        tool_calls=[
                            ToolCall(
                                id="2",
                                name="edit_file",
                                arguments={
                                    "path": path,
                                    "old_text": "also nonexistent",
                                    "new_text": "replacement",
                                },
                            )
                        ],
                    ),
                    CompletionResult(content="Let me try write_file."),
                ],
                max_steps=5,
                safety_mode="danger",
            )

            await _collect_events(agent, "edit the file")

            tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
            # 2nd edit failure should mention write_file
            # Check the last tool message before the response
            assert "write_file" in tool_msgs[-1]["content"]
        finally:
            os.unlink(path)
            reset_tracker()

    async def test_3_edit_failures_stop_directive(self):
        """After 3 failed edits, warning should say STOP."""
        import os
        import tempfile

        from natshell.tools.file_tracker import reset_tracker

        reset_tracker()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("real content\n")
            path = f.name

        try:
            agent = _make_agent(
                [
                    CompletionResult(
                        tool_calls=[
                            ToolCall(
                                id="1",
                                name="edit_file",
                                arguments={
                                    "path": path,
                                    "old_text": "nonexistent1",
                                    "new_text": "replacement",
                                },
                            )
                        ],
                    ),
                    CompletionResult(
                        tool_calls=[
                            ToolCall(
                                id="2",
                                name="edit_file",
                                arguments={
                                    "path": path,
                                    "old_text": "nonexistent2",
                                    "new_text": "replacement",
                                },
                            )
                        ],
                    ),
                    CompletionResult(
                        tool_calls=[
                            ToolCall(
                                id="3",
                                name="edit_file",
                                arguments={
                                    "path": path,
                                    "old_text": "nonexistent3",
                                    "new_text": "replacement",
                                },
                            )
                        ],
                    ),
                    CompletionResult(content="Let me try write_file."),
                ],
                max_steps=5,
                safety_mode="danger",
            )

            await _collect_events(agent, "edit the file")

            tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
            # 3rd edit failure should say "STOP using edit_file"
            assert "STOP using edit_file" in tool_msgs[-1]["content"]
        finally:
            os.unlink(path)
            reset_tracker()


# ─── Planning intent detection ────────────────────────────────────────────────


class TestPlanRequestDetection:
    def test_create_a_plan(self):
        assert _is_plan_request("create a plan for the migration")

    def test_write_a_plan(self):
        assert _is_plan_request("write a plan to update the API")

    def test_make_a_plan(self):
        assert _is_plan_request("make a plan for refactoring")

    def test_draft_plan(self):
        assert _is_plan_request("draft a plan for the new feature")

    def test_update_plan(self):
        assert _is_plan_request("update the plan for deployment")

    def test_build_a_plan(self):
        assert _is_plan_request("build a plan for the CI pipeline")

    def test_plan_to_update(self):
        assert _is_plan_request("plan to update the database schema")

    def test_plan_for_migration(self):
        assert _is_plan_request("plan for migrating to v2")

    def test_plan_how_to(self):
        assert _is_plan_request("plan how to restructure the codebase")

    def test_not_plan_go_home(self):
        """Casual use of 'plan' should not trigger."""
        assert not _is_plan_request("I plan to go home later")

    def test_not_plain_statement(self):
        assert not _is_plan_request("fix the bug in main.py")

    def test_not_execute_plan(self):
        """Running a plan is not asking to create one."""
        assert not _is_plan_request("execute the plan now")


class TestPlanningIntentInjection:
    async def test_plan_request_injects_system_message(self):
        """When user asks to plan, a system message should be injected."""
        agent = _make_agent(
            [CompletionResult(content="Here is my plan...")],
        )
        await _collect_events(agent, "create a plan for updating the API")

        # Check that a planning mode system message was injected
        system_msgs = [
            m for m in agent.messages
            if m["role"] == "system" and "Planning mode" in m.get("content", "")
        ]
        assert len(system_msgs) == 1
        assert "Do not modify files" in system_msgs[0]["content"]

    async def test_non_plan_request_no_system_message(self):
        """Normal requests should not inject a planning system message."""
        agent = _make_agent(
            [CompletionResult(content="Done!")],
        )
        await _collect_events(agent, "fix the bug in main.py")

        system_msgs = [
            m for m in agent.messages
            if m["role"] == "system" and "Planning mode" in m.get("content", "")
        ]
        assert len(system_msgs) == 0


# ─── ContextOverflowError from local engine ──────────────────────────────────


class TestLocalContextOverflow:
    async def test_local_valueerror_becomes_context_overflow(self):
        """ValueError with 'context window' should become ContextOverflowError."""
        from natshell.inference.remote import ContextOverflowError

        agent = _make_agent([])

        # Seed enough history to allow compaction
        agent.messages.append({"role": "user", "content": "first question"})
        agent.messages.append({"role": "assistant", "content": "first answer"})
        agent.messages.append({"role": "user", "content": "second question"})
        agent.messages.append({"role": "assistant", "content": "second answer"})

        # Simulate a ContextOverflowError (as local.py would raise)
        agent.engine.chat_completion = AsyncMock(
            side_effect=[
                ContextOverflowError("Prompt exceeds local model context window (4096 tokens)"),
                CompletionResult(content="Recovered after compaction."),
            ]
        )

        events = await _collect_events(agent, "another question")
        types = [e.type for e in events]

        # Should recover via compaction
        error_events = [e for e in events if e.type == EventType.ERROR]
        assert any("compacted" in e.data.lower() for e in error_events)
        assert EventType.RESPONSE in types


# ─── Analysis request detection ───────────────────────────────────────────────


class TestAnalysisRequestDetection:
    def test_review_code(self):
        assert _is_analysis_request("review the code in src/natshell/safety/")

    def test_security_audit(self):
        assert _is_analysis_request("do a security audit of the codebase")

    def test_analyze_codebase(self):
        assert _is_analysis_request("analyze this codebase for issues")

    def test_examine_implementation(self):
        assert _is_analysis_request("examine the implementation of the agent loop")

    def test_inspect_security(self):
        assert _is_analysis_request("inspect the security module for vulnerabilities")

    def test_audit_repository(self):
        assert _is_analysis_request("audit this repository")

    def test_review_pr(self):
        assert _is_analysis_request("review the PR for correctness")

    def test_review_diff(self):
        assert _is_analysis_request("review the diff before merging")

    def test_code_review_phrase(self):
        assert _is_analysis_request("do a code review")

    def test_security_analysis_phrase(self):
        assert _is_analysis_request("perform a security analysis")

    def test_codebase_audit_phrase(self):
        assert _is_analysis_request("run a codebase audit")

    def test_review_module(self):
        assert _is_analysis_request("review the module for edge cases")

    # Negative cases
    def test_not_review_dinner(self):
        assert not _is_analysis_request("review my dinner plans")

    def test_not_analyze_disk(self):
        assert not _is_analysis_request("analyze disk usage")

    def test_not_examine_network(self):
        assert not _is_analysis_request("examine network traffic")

    def test_not_plain_fix(self):
        assert not _is_analysis_request("fix the bug in main.py")


class TestAnalysisIntentInjection:
    async def test_analysis_request_injects_system_message(self):
        """When user asks for a review, an analysis mode system message should be injected."""
        agent = _make_agent(
            [CompletionResult(content="Here are my findings...")],
        )
        await _collect_events(agent, "review the code in src/natshell/safety/")

        system_msgs = [
            m for m in agent.messages
            if m["role"] == "system" and "Analysis mode" in m.get("content", "")
        ]
        assert len(system_msgs) == 1
        assert "Verify every finding" in system_msgs[0]["content"]

    async def test_non_analysis_request_no_system_message(self):
        """Normal requests should not inject an analysis system message."""
        agent = _make_agent(
            [CompletionResult(content="Done!")],
        )
        await _collect_events(agent, "fix the bug in main.py")

        system_msgs = [
            m for m in agent.messages
            if m["role"] == "system" and "Analysis mode" in m.get("content", "")
        ]
        assert len(system_msgs) == 0

    async def test_analysis_and_plan_coexist(self):
        """A request to plan a review should trigger both injections."""
        agent = _make_agent(
            [CompletionResult(content="Here is the plan for the review...")],
        )
        await _collect_events(agent, "create a plan to review the code")

        plan_msgs = [
            m for m in agent.messages
            if m["role"] == "system" and "Planning mode" in m.get("content", "")
        ]
        analysis_msgs = [
            m for m in agent.messages
            if m["role"] == "system" and "Analysis mode" in m.get("content", "")
        ]
        assert len(plan_msgs) == 1
        assert len(analysis_msgs) == 1


# ─── System prompt analysis section ──────────────────────────────────────────


class TestSystemPromptAnalysisSection:
    def test_system_prompt_contains_analysis_section(self):
        """The system prompt should include the Analysis & Review section."""
        from natshell.agent.system_prompt import build_system_prompt

        prompt = build_system_prompt(
            SystemContext(
                hostname="testhost",
                distro="Test",
                kernel="6.0",
                username="tester",
            )
        )
        assert "## Analysis & Review" in prompt
        assert "Trace data flows" in prompt


# ─── Compact system prompt ────────────────────────────────────────────────────


class TestCompactSystemPrompt:
    def _ctx(self) -> SystemContext:
        return SystemContext(
            hostname="testhost",
            distro="Test",
            kernel="6.0",
            username="tester",
        )

    def test_8k_context_selects_compact(self):
        """An 8K context window should produce a compact system prompt."""
        agent = _make_agent_with_ctx(n_ctx=8192)
        prompt = agent.messages[0]["content"]
        # Compact mode omits these sections
        assert "## Git Integration" not in prompt
        assert "## Edit Failure Recovery" not in prompt
        assert "## Task Completion" not in prompt
        assert "## Analysis & Review" not in prompt
        assert "## NatShell Configuration" not in prompt

    def test_32k_context_selects_full(self):
        """A 32K context window should produce the full system prompt."""
        agent = _make_agent_with_ctx(n_ctx=32768)
        prompt = agent.messages[0]["content"]
        assert "## Git Integration" in prompt
        assert "## Analysis & Review" in prompt
        assert "## NatShell Configuration" in prompt

    def test_16k_context_selects_full(self):
        """A 16K context window (boundary) should use full mode, not compact."""
        agent = _make_agent_with_ctx(n_ctx=16384)
        prompt = agent.messages[0]["content"]
        assert "## Git Integration" in prompt

    def test_compact_prompt_shorter_than_full(self):
        """Compact prompt should be at least 2000 chars shorter than full."""
        from natshell.agent.system_prompt import build_system_prompt

        ctx = self._ctx()
        full = build_system_prompt(ctx, compact=False)
        compact = build_system_prompt(ctx, compact=True)
        assert len(full) - len(compact) >= 2000

    def test_compact_prompt_keeps_behavior_rules(self):
        """Compact prompt still includes Behavior Rules and /no_think."""
        from natshell.agent.system_prompt import build_system_prompt

        prompt = build_system_prompt(self._ctx(), compact=True)
        assert "## Behavior Rules" in prompt
        assert "/no_think" in prompt
        assert "## System Information" in prompt

    def test_compact_prompt_keeps_code_section(self):
        """Compact prompt includes a condensed Code Editing section."""
        from natshell.agent.system_prompt import build_system_prompt

        prompt = build_system_prompt(self._ctx(), compact=True)
        assert "## Code Editing & Development" in prompt
        assert "FILE TRUNCATED" in prompt

    def test_compact_prompt_has_condensed_behavior_rules(self):
        """Compact prompt uses 8 condensed behavior rules instead of 15."""
        from natshell.agent.system_prompt import build_system_prompt

        prompt = build_system_prompt(self._ctx(), compact=True)
        # Should NOT have full verbose rules
        assert "Use --dry-run flags" not in prompt
        assert "If you don't know how to do something on this specific distro" not in prompt
        # Should have condensed rules
        assert "PLAN before acting" in prompt
        assert "One command at a time" in prompt


# ─── Compact tool formatting ────────────────────────────────────────────────


_SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_shell",
            "description": (
                "Execute a shell command on the user's system"
                " and return the output."
                " Use this for system administration tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum execution time in seconds (default 60, max 300).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_tool",
            "description": (
                "Perform structured git operations."
                " Returns clean output for common git tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["status", "diff", "log", "branch", "commit", "stash"],
                        "description": "The git operation to perform.",
                    },
                    "args": {
                        "type": "string",
                        "description": "Additional arguments for the operation.",
                    },
                },
                "required": ["operation"],
            },
        },
    },
]


class TestCompactToolFormatting:
    def test_compact_output_is_shorter(self):
        """Compact tool formatting produces significantly shorter output."""
        full = _format_tools_for_prompt(_SAMPLE_TOOLS, compact=False)
        compact = _format_tools_for_prompt(_SAMPLE_TOOLS, compact=True)
        assert len(compact) < len(full)
        # Should be substantially shorter (at least 30% reduction)
        assert len(compact) < len(full) * 0.7

    def test_compact_includes_enum_values(self):
        """Compact format preserves enum values inline for parameters."""
        compact = _format_tools_for_prompt(_SAMPLE_TOOLS, compact=True)
        assert "status|diff|log|branch|commit|stash" in compact

    def test_compact_first_sentence_only(self):
        """Compact format uses first sentence of description only."""
        compact = _format_tools_for_prompt(_SAMPLE_TOOLS, compact=True)
        # Should have first sentence
        assert "Execute a shell command on the user's system and return the output." in compact
        # Should NOT have second sentence
        assert "Use this for system administration tasks" not in compact

    def test_compact_omits_param_descriptions(self):
        """Compact format drops parameter descriptions."""
        compact = _format_tools_for_prompt(_SAMPLE_TOOLS, compact=True)
        assert "The shell command to execute" not in compact
        assert "Maximum execution time" not in compact

    def test_compact_marks_required_params(self):
        """Compact format still marks required parameters."""
        compact = _format_tools_for_prompt(_SAMPLE_TOOLS, compact=True)
        assert "command (string, required)" in compact
        assert "operation (status|diff|log|branch|commit|stash, required)" in compact

    def test_compact_header_is_imperative(self):
        """Compact format uses imperative tool call instruction."""
        compact = _format_tools_for_prompt(_SAMPLE_TOOLS, compact=True)
        assert "You MUST" in compact
        assert "NEVER" in compact
        assert '<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>' in compact

    def test_full_format_unchanged(self):
        """Full format still has multi-line header and parameter descriptions."""
        full = _format_tools_for_prompt(_SAMPLE_TOOLS, compact=False)
        assert "You MUST use tools to perform actions." in full
        assert "The shell command to execute." in full
        assert "Parameters:" in full

    def test_compact_mistral_is_shorter(self):
        """Compact Mistral tool formatting is also shorter."""
        full = _format_tools_for_prompt_mistral(_SAMPLE_TOOLS, compact=False)
        compact = _format_tools_for_prompt_mistral(_SAMPLE_TOOLS, compact=True)
        assert len(compact) < len(full)
        assert "status|diff|log|branch|commit|stash" in compact

    def test_compact_mistral_header_is_imperative(self):
        """Compact Mistral format uses imperative tool call instruction."""
        compact = _format_tools_for_prompt_mistral(_SAMPLE_TOOLS, compact=True)
        assert "You MUST" in compact
        assert "NEVER" in compact
        assert '[TOOL_CALLS] [{"name": "tool_name", "arguments": {...}}]' in compact

    def test_8k_engine_uses_compact_tools(self):
        """An 8K context agent should use compact tool formatting for budget estimation."""
        agent = _make_agent_with_ctx(n_ctx=8192)
        # The tool overhead should match compact formatting
        tool_schemas = agent.tools.get_tool_schemas()
        compact_text = _format_tools_for_prompt(tool_schemas, compact=True)
        full_text = _format_tools_for_prompt(tool_schemas, compact=False)
        # Compact overhead should be much smaller than full
        compact_len = len(compact_text) // 4
        full_len = len(full_text) // 4
        assert compact_len < full_len
        # The agent's stored overhead should match compact (not full)
        assert agent._tool_token_overhead <= compact_len + 50  # small margin for tokenizer variance


# ─── Mistral bare-JSON fallback ──────────────────────────────────────────────


def _mistral_parse(response: dict):
    """Call LocalEngine._parse_response with a fake Mistral self."""
    fake = types.SimpleNamespace(model_family="mistral")
    return LocalEngine._parse_response(fake, response)


def _qwen_parse(response: dict):
    """Call LocalEngine._parse_response with a fake Qwen self."""
    fake = types.SimpleNamespace(model_family="qwen")
    return LocalEngine._parse_response(fake, response)


def _make_llama_response(content: str, finish_reason: str = "stop") -> dict:
    return {
        "choices": [
            {"message": {"content": content, "tool_calls": None}, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


class TestMistralBareJsonFallback:
    """Mistral forgot [TOOL_CALLS] prefix — parser should still recover the call."""

    def test_bare_object_parsed(self):
        """Bare JSON object with name+arguments is recovered as a tool call."""
        content = '{"name": "kiwix_search", "arguments": {"query": "linux kernel", "results": 5}}'
        result = _mistral_parse(_make_llama_response(content))
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "kiwix_search"
        assert result.tool_calls[0].arguments == {"query": "linux kernel", "results": 5}

    def test_bare_array_parsed(self):
        """Bare JSON array of tool call objects is recovered."""
        content = '[{"name": "execute_shell", "arguments": {"command": "ls"}}]'
        result = _mistral_parse(_make_llama_response(content))
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "execute_shell"

    def test_bare_json_not_applied_to_qwen(self):
        """Bare JSON fallback does NOT fire for Qwen models (avoid false positives)."""
        content = '{"name": "execute_shell", "arguments": {"command": "ls"}}'
        result = _qwen_parse(_make_llama_response(content))
        assert len(result.tool_calls) == 0

    def test_non_tool_json_ignored(self):
        """JSON that lacks name+arguments keys is not treated as a tool call."""
        content = '{"foo": "bar", "baz": 123}'
        result = _mistral_parse(_make_llama_response(content))
        assert len(result.tool_calls) == 0

    def test_normal_tool_calls_prefix_still_works(self):
        """[TOOL_CALLS] prefix path still works and takes precedence."""
        content = '[TOOL_CALLS] [{"name": "execute_shell", "arguments": {"command": "pwd"}}]'
        result = _mistral_parse(_make_llama_response(content))
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "execute_shell"

    def test_code_fenced_json_parsed(self):
        """JSON wrapped in ```json ... ``` code fences is recovered."""
        content = (
            '```json\n{"name": "execute_shell",'
            ' "arguments": {"command": "find / -name config"}}\n```'
        )
        result = _mistral_parse(_make_llama_response(content))
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "execute_shell"
        assert result.tool_calls[0].arguments == {
            "command": "find / -name config",
        }

    def test_code_fenced_no_lang_tag(self):
        """Code fences without 'json' language tag are also recovered."""
        content = '```\n{"name": "list_directory", "arguments": {"path": "/etc"}}\n```'
        result = _mistral_parse(_make_llama_response(content))
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "list_directory"

    def test_text_before_code_fenced_json(self):
        """Planning text before fenced JSON: tool recovered, text preserved."""
        content = (
            "I will search for the config file.\n"
            '```json\n{"name": "execute_shell",'
            ' "arguments": {"command": "find / -name config"}}\n```'
        )
        result = _mistral_parse(_make_llama_response(content))
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "execute_shell"
        # Planning text should be preserved
        assert result.content is not None
        assert "search for the config file" in result.content

    def test_code_fenced_json_stripped(self):
        """Code-fenced JSON is not leaked to UI as visible text."""
        content = '```json\n{"name": "execute_shell", "arguments": {"command": "ls"}}\n```'
        result = _mistral_parse(_make_llama_response(content))
        assert len(result.tool_calls) == 1
        # Content should be None (no text besides the fenced JSON)
        assert result.content is None

    def test_bare_json_stripped(self):
        """Bare JSON tool calls are not leaked to UI as visible text."""
        content = '{"name": "execute_shell", "arguments": {"command": "ls"}}'
        result = _mistral_parse(_make_llama_response(content))
        assert len(result.tool_calls) == 1
        assert result.content is None

    def test_code_fenced_array_parsed(self):
        """JSON array wrapped in code fences is recovered."""
        content = '```json\n[{"name": "execute_shell", "arguments": {"command": "ls"}}]\n```'
        result = _mistral_parse(_make_llama_response(content))
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "execute_shell"

    def test_code_fenced_non_tool_json_ignored(self):
        """Non-tool JSON inside code fences is not treated as a tool call."""
        content = '```json\n{"config": "value", "debug": true}\n```'
        result = _mistral_parse(_make_llama_response(content))
        assert len(result.tool_calls) == 0

    def test_code_fenced_not_applied_to_qwen(self):
        """Code-fence JSON fallback does NOT fire for Qwen models."""
        content = '```json\n{"name": "execute_shell", "arguments": {"command": "ls"}}\n```'
        result = _qwen_parse(_make_llama_response(content))
        assert len(result.tool_calls) == 0

    def test_bare_json_no_arguments_key(self):
        """Bare JSON with no 'arguments' key is accepted (arguments={})."""
        content = '[{"name": "list_directory"}]'
        result = _mistral_parse(_make_llama_response(content))
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "list_directory"
        assert result.tool_calls[0].arguments == {}

    def test_bare_json_flat_arguments(self):
        """Bare JSON with args at top level (not nested under 'arguments')."""
        content = (
            '[{"name": "execute_shell",'
            ' "command": "nmap -sn 192.168.5.0/24", "timeout": 30}]'
        )
        result = _mistral_parse(_make_llama_response(content))
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "execute_shell"
        assert result.tool_calls[0].arguments == {
            "command": "nmap -sn 192.168.5.0/24",
            "timeout": 30,
        }

    def test_tool_calls_prefix_no_arguments_key(self):
        """[TOOL_CALLS] with no 'arguments' key — args at top level."""
        content = (
            '[TOOL_CALLS] [{"name": "execute_shell",'
            ' "command": "ls -la", "timeout": 10}]'
        )
        result = _mistral_parse(_make_llama_response(content))
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "execute_shell"
        assert result.tool_calls[0].arguments == {
            "command": "ls -la",
            "timeout": 10,
        }

    def test_tool_calls_prefix_empty_arguments(self):
        """[TOOL_CALLS] with no 'arguments' key and no other keys."""
        content = '[TOOL_CALLS] [{"name": "list_directory"}]'
        result = _mistral_parse(_make_llama_response(content))
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "list_directory"
        assert result.tool_calls[0].arguments == {}


# ─── compact Mistral anti-fence instruction ──────────────────────────────────


class TestMistralPromptInstructions:
    """Verify Mistral prompt formatting includes anti-pattern instructions."""

    def test_compact_mistral_anti_fence_instruction(self):
        """Compact Mistral prompt mentions code fences."""
        compact = _format_tools_for_prompt_mistral(_SAMPLE_TOOLS, compact=True)
        assert "code fence" in compact.lower()
        assert "Do NOT describe commands in prose" in compact

    def test_full_mistral_anti_fence_instruction(self):
        """Full Mistral prompt also mentions code fences."""
        full = _format_tools_for_prompt_mistral(_SAMPLE_TOOLS, compact=False)
        assert "code fence" in full.lower()
        assert "Do NOT describe commands in prose" in full


# ─── Mistral message normalization (role alternation) ────────────────────────


def _normalize(messages):
    """Call _normalize_messages_for_mistral without instantiating LocalEngine."""
    engine = object.__new__(LocalEngine)
    return engine._normalize_messages_for_mistral(messages)


class TestMistralMessageNormalization:
    """Verify that _normalize_messages_for_mistral enforces strict role alternation."""

    def test_consecutive_user_messages_merged(self):
        """Two consecutive user messages are merged with \\n\\n separator."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        result = _normalize(msgs)
        assert len(result) == 2
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "first\n\nsecond"

    def test_three_consecutive_user_messages_merged(self):
        """Three consecutive user messages collapse into one."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        result = _normalize(msgs)
        assert len(result) == 2
        assert result[1]["content"] == "a\n\nb\n\nc"

    def test_system_between_users_folded_then_users_merged(self):
        """Mid-conversation system is folded (Pass 1), then adjacent users merge (Pass 2)."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "hint"},
            {"role": "user", "content": "u2"},
        ]
        result = _normalize(msgs)
        # system hint folded into initial system message
        assert result[0]["role"] == "system"
        assert "hint" in result[0]["content"]
        # the two user messages (previously separated by system) are now merged
        assert len(result) == 2
        assert result[1]["content"] == "u1\n\nu2"

    def test_tool_call_pairs_preserved(self):
        """Assistant with tool_calls → tool response pair is never merged."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do something"},
            {
                "role": "assistant",
                "content": "I'll run a command",
                "tool_calls": [{"id": "1"}],
            },
            {"role": "tool", "content": "file1\nfile2", "tool_call_id": "1"},
            {"role": "assistant", "content": "Here are your files."},
        ]
        result = _normalize(msgs)
        assert len(result) == 5
        assert result[2]["role"] == "assistant"
        assert "tool_calls" in result[2]
        assert result[3]["role"] == "tool"
        assert result[4]["role"] == "assistant"

    def test_assistant_with_tool_calls_not_merged(self):
        """An assistant message with tool_calls is never merged with adjacent assistant."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "thinking"},
            {"role": "assistant", "content": "calling", "tool_calls": [{"id": "1"}]},
        ]
        result = _normalize(msgs)
        # Should NOT merge because second has tool_calls
        assert len(result) == 4
        assert result[2]["content"] == "thinking"
        assert "tool_calls" in result[3]

    def test_plain_consecutive_assistants_merged(self):
        """Consecutive plain assistant messages (no tool_calls) are merged."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "part 1"},
            {"role": "assistant", "content": "part 2"},
        ]
        result = _normalize(msgs)
        assert len(result) == 3
        assert result[2]["content"] == "part 1\n\npart 2"

    def test_normal_alternation_unchanged(self):
        """Properly alternating messages pass through unchanged."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": "goodbye"},
        ]
        result = _normalize(msgs)
        assert len(result) == 5
        assert [m["role"] for m in result] == ["system", "user", "assistant", "user", "assistant"]

    def test_cmd_then_new_question(self):
        """Realistic /cmd scenario: command output (user) → new question (user) merged."""
        msgs = [
            {"role": "system", "content": "You are NatShell."},
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": "Here are the files."},
            # /cmd appends command output as user message
            {"role": "user", "content": "Command output:\n$ ls\nfile1.py  file2.py"},
            # Next user question
            {"role": "user", "content": "What does file1.py do?"},
        ]
        result = _normalize(msgs)
        assert len(result) == 4
        assert result[3]["role"] == "user"
        assert "Command output:" in result[3]["content"]
        assert "What does file1.py do?" in result[3]["content"]

    def test_context_summary_system_merged(self):
        """Context summary system message mid-conversation is folded into initial system."""
        msgs = [
            {"role": "system", "content": "You are NatShell."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {
                "role": "system",
                "content": "[Context summary] Previous conversation discussed files.",
            },
            {"role": "user", "content": "continue"},
        ]
        result = _normalize(msgs)
        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert "[Context summary]" in result[0]["content"]
        assert result[3]["role"] == "user"

    def test_empty_messages(self):
        """Empty input returns empty output."""
        assert _normalize([]) == []

    def test_system_only(self):
        """Single system message passes through."""
        msgs = [{"role": "system", "content": "sys"}]
        result = _normalize(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "system"

    def test_tool_messages_never_merged(self):
        """Tool-role messages are never touched or merged."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "call", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "result1", "tool_call_id": "1"},
            {"role": "assistant", "content": "call2", "tool_calls": [{"id": "2"}]},
            {"role": "tool", "content": "result2", "tool_call_id": "2"},
            {"role": "assistant", "content": "done"},
        ]
        result = _normalize(msgs)
        assert len(result) == 7
        assert [m["role"] for m in result] == [
            "system", "user", "assistant", "tool", "assistant", "tool", "assistant"
        ]


# ─── skip_intent_detection ───────────────────────────────────────────────────


class TestSkipIntentDetection:
    """Test that skip_intent_detection=True prevents planning/analysis injections."""

    @staticmethod
    async def _collect_with_kwargs(agent, message, **kwargs):
        events = []
        async for event in agent.handle_user_message(message, **kwargs):
            events.append(event)
        return events

    async def test_plan_generation_skips_intent_injection(self):
        """With skip_intent_detection=True, no '[Planning mode]' message is injected."""
        agent = _make_agent([
            CompletionResult(content="I will create a plan", tool_calls=None),
        ])
        # This prompt would normally trigger _is_plan_request
        prompt = "Create a plan for building a REST API and write_file PLAN.md"
        await self._collect_with_kwargs(agent, prompt, skip_intent_detection=True)
        system_msgs = [
            m for m in agent.messages
            if m["role"] == "system" and "[Planning mode]" in m.get("content", "")
        ]
        assert len(system_msgs) == 0

    async def test_plan_request_without_skip_injects_planning_mode(self):
        """Without skip_intent_detection, planning mode IS injected for plan requests."""
        agent = _make_agent([
            CompletionResult(content="Here's my plan", tool_calls=None),
        ])
        prompt = "Create a plan for building a REST API"
        await self._collect_with_kwargs(agent, prompt, skip_intent_detection=False)
        system_msgs = [
            m for m in agent.messages
            if m["role"] == "system" and "[Planning mode]" in m.get("content", "")
        ]
        assert len(system_msgs) == 1

    async def test_analysis_request_skipped_with_flag(self):
        """With skip_intent_detection=True, no '[Analysis mode]' message is injected."""
        agent = _make_agent([
            CompletionResult(content="Analysis complete", tool_calls=None),
        ])
        prompt = "Review the codebase security"
        await self._collect_with_kwargs(agent, prompt, skip_intent_detection=True)
        system_msgs = [
            m for m in agent.messages
            if m["role"] == "system" and "[Analysis mode]" in m.get("content", "")
        ]
        assert len(system_msgs) == 0


# ─── Small context tool filtering ────────────────────────────────────────────


class TestSmallContextToolFilter:
    def test_small_ctx_activates_filter(self):
        """n_ctx=4096 should activate SMALL_CONTEXT_TOOLS filter."""
        from natshell.tools.registry import SMALL_CONTEXT_TOOLS

        agent = _make_agent_with_ctx(4096)
        assert agent._context_tool_filter == SMALL_CONTEXT_TOOLS

    def test_edge_ctx_8192_activates_filter(self):
        """n_ctx=8192 (edge case) should activate the filter."""
        from natshell.tools.registry import SMALL_CONTEXT_TOOLS

        agent = _make_agent_with_ctx(8192)
        assert agent._context_tool_filter == SMALL_CONTEXT_TOOLS

    def test_large_ctx_no_filter(self):
        """n_ctx=16384 should NOT activate the filter."""
        agent = _make_agent_with_ctx(16384)
        assert agent._context_tool_filter is None

    async def test_per_call_filter_intersects_with_context_filter(self):
        """Per-call tool_filter should intersect with the context filter."""
        agent = _make_agent_with_ctx(4096)
        # Per-call filter includes read_file and git_tool.
        # git_tool is NOT in SMALL_CONTEXT_TOOLS, so the effective filter
        # should be just {read_file}.
        per_call = {"read_file", "git_tool"}
        events = []
        async for event in agent.handle_user_message("test", tool_filter=per_call):
            events.append(event)

        # Verify the engine was called with schemas filtered to the intersection
        call_args = agent.engine.chat_completion.call_args
        schemas = call_args.kwargs.get("tools") or call_args[1].get("tools", [])
        schema_names = {s["function"]["name"] for s in schemas}
        assert schema_names == {"read_file"}


# ─── Sudo retry prepends sudo ────────────────────────────────────────────────


class TestSudoRetryPrependsSudo:
    async def test_retry_prepends_sudo_when_missing(self):
        """When 'apt install nmap' triggers a sudo error, the retry should
        prepend 'sudo' so the password injection fires."""
        from natshell.tools.registry import ToolResult

        # Model calls execute_shell with "apt install -y nmap" (no sudo)
        agent = _make_agent(
            [
                CompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            name="execute_shell",
                            arguments={"command": "apt install -y nmap"},
                        )
                    ],
                ),
                CompletionResult(content="nmap installed!"),
            ],
            safety_mode="warn",
        )

        # Mock the tool execution: first call returns sudo error, second succeeds
        sudo_error = ToolResult(
            exit_code=1,
            error="sudo: a terminal is required to read the password",
        )
        success = ToolResult(exit_code=0, output="installed nmap")
        agent.tools.execute = AsyncMock(side_effect=[sudo_error, success])

        # Password callback returns a password
        password_cb = AsyncMock(return_value="hunter2")

        events = []
        async for event in agent.handle_user_message(
            "install nmap",
            password_callback=password_cb,
        ):
            events.append(event)

        # Password callback should have been called
        password_cb.assert_called_once()

        # The retry call should have prepended sudo
        assert agent.tools.execute.call_count == 2
        retry_call = agent.tools.execute.call_args_list[1]
        # The retry args are passed as (name, arguments_dict)
        retry_args = retry_call[0][1] if len(retry_call[0]) > 1 else retry_call[1]
        assert retry_args.get("command", "").startswith("sudo ")

    async def test_retry_does_not_double_sudo(self):
        """When 'sudo apt install nmap' triggers a sudo error, the retry should
        NOT prepend a second 'sudo'."""
        from natshell.tools.registry import ToolResult

        agent = _make_agent(
            [
                CompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            name="execute_shell",
                            arguments={"command": "sudo apt install -y nmap"},
                        )
                    ],
                ),
                CompletionResult(content="done"),
            ],
            safety_mode="warn",
        )

        sudo_error = ToolResult(
            exit_code=1,
            error="sudo: no password was provided",
        )
        success = ToolResult(exit_code=0, output="installed")
        agent.tools.execute = AsyncMock(side_effect=[sudo_error, success])
        password_cb = AsyncMock(return_value="hunter2")

        async for _ in agent.handle_user_message(
            "install nmap",
            password_callback=password_cb,
        ):
            pass

        # Retry should use the original command (already has sudo)
        retry_args = agent.tools.execute.call_args_list[1][0][1]
        assert retry_args["command"] == "sudo apt install -y nmap"
        assert not retry_args["command"].startswith("sudo sudo")


# ─── Duplicate tool call detection ───────────────────────────────────────────


class TestDuplicateToolCallDetection:
    async def test_warning_at_3_identical_calls(self):
        """After 3 identical consecutive tool calls, a warning is injected."""
        responses = [
            CompletionResult(
                tool_calls=[
                    ToolCall(
                        id=str(i),
                        name="execute_shell",
                        arguments={"command": "npm run type-check"},
                    )
                ],
            )
            for i in range(4)
        ] + [CompletionResult(content="Done.")]

        agent = _make_agent(responses, max_steps=10, safety_mode="danger")
        await _collect_events(agent, "check types")

        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        # 3rd call should have the warning
        assert any(
            "identical arguments" in m["content"] and "DIFFERENT approach" in m["content"]
            for m in tool_msgs
        )

    async def test_abort_at_5_identical_calls(self):
        """After 5 identical consecutive tool calls, the agent recovers and gets one more LLM turn."""
        responses = [
            CompletionResult(
                tool_calls=[
                    ToolCall(
                        id=str(i),
                        name="execute_shell",
                        arguments={"command": "npm run type-check"},
                    )
                ],
            )
            for i in range(5)  # 5 identical calls triggers the dupe-abort
        ] + [CompletionResult(content="Task complete.")]

        agent = _make_agent(responses, max_steps=15, safety_mode="danger")
        events = await _collect_events(agent, "check types")

        # The loop should NOT hard-terminate with the old "stopped" message
        response_events = [e for e in events if e.type == EventType.RESPONSE]
        assert not any("repeated identical tool call" in e.data for e in response_events)

        # The warning should appear in the tool result message history
        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        assert any("STOP making this tool call" in m["content"] for m in tool_msgs)

        # The agent should have produced a normal final response
        assert any("Task complete." in e.data for e in response_events)

        # Tool was called exactly 5 times, not 6+
        assert len(tool_msgs) == 5

    async def test_counter_resets_on_different_call(self):
        """Counter resets when a different tool call is made."""
        responses = [
            # 3 identical calls (triggers warning but not abort)
            CompletionResult(
                tool_calls=[
                    ToolCall(id="1", name="execute_shell", arguments={"command": "echo a"})
                ],
            ),
            CompletionResult(
                tool_calls=[
                    ToolCall(id="2", name="execute_shell", arguments={"command": "echo a"})
                ],
            ),
            CompletionResult(
                tool_calls=[
                    ToolCall(id="3", name="execute_shell", arguments={"command": "echo a"})
                ],
            ),
            # Different call resets counter
            CompletionResult(
                tool_calls=[
                    ToolCall(id="4", name="execute_shell", arguments={"command": "echo b"})
                ],
            ),
            # 3 more of the original (triggers warning again but not abort)
            CompletionResult(
                tool_calls=[
                    ToolCall(id="5", name="execute_shell", arguments={"command": "echo a"})
                ],
            ),
            CompletionResult(
                tool_calls=[
                    ToolCall(id="6", name="execute_shell", arguments={"command": "echo a"})
                ],
            ),
            CompletionResult(
                tool_calls=[
                    ToolCall(id="7", name="execute_shell", arguments={"command": "echo a"})
                ],
            ),
            CompletionResult(content="Done."),
        ]

        agent = _make_agent(responses, max_steps=10, safety_mode="danger")
        events = await _collect_events(agent, "do stuff")

        # Should complete normally (no abort) because counter was reset
        response_events = [e for e in events if e.type == EventType.RESPONSE]
        assert len(response_events) == 1
        assert response_events[0].data == "Done."


# ─── Plan step budget enforcement ────────────────────────────────────────────


class TestPlanStepBudget:
    def test_set_step_limit(self):
        """set_step_limit directly overrides _max_steps."""
        agent = _make_agent(
            [CompletionResult(content="ok")],
            max_steps=15,
        )
        # _max_steps is set during initialize based on context
        original = agent._max_steps
        agent.set_step_limit(42)
        assert agent._max_steps == 42
        # Restore
        agent.set_step_limit(original)
        assert agent._max_steps == original

    async def test_set_step_limit_enforced_in_loop(self):
        """set_step_limit is respected by handle_user_message loop."""
        # Create 20 tool calls but limit to 5
        responses = [
            CompletionResult(
                tool_calls=[
                    ToolCall(
                        id=str(i),
                        name="execute_shell",
                        arguments={"command": f"echo step{i}"},
                    )
                ],
            )
            for i in range(20)
        ] + [CompletionResult(content="Done.")]

        agent = _make_agent(responses, max_steps=15, safety_mode="danger")
        agent.set_step_limit(5)
        events = await _collect_events(agent, "do stuff")

        # Should hit max steps at 5, not continue to 15 or 20
        response_events = [e for e in events if e.type == EventType.RESPONSE]
        assert len(response_events) == 1
        assert "maximum number of steps (5)" in response_events[0].data


# ─── Context compression ────────────────────────────────────────────────────


class TestCompressOldMessages:
    def test_compresses_write_file_args(self):
        """Old write_file tool call arguments have content elided."""
        agent = _make_agent([CompletionResult(content="ok")])
        agent.messages = [{"role": "system", "content": "sys"}]
        # Add 12 messages so oldest are outside the preserve window
        large_content = "x" * 2000
        for i in range(6):
            agent.messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": str(i), "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({
                            "path": f"/tmp/f{i}.ts",
                            "content": large_content,
                        }),
                    },
                }],
            })
            agent.messages.append({
                "role": "tool", "tool_call_id": str(i),
                "content": f"Wrote /tmp/f{i}.ts",
            })

        agent._compress_old_messages()

        # Oldest messages should be compressed
        first_tc = agent.messages[1]["tool_calls"][0]
        args = json.loads(first_tc["function"]["arguments"])
        assert "elided" in args["content"]
        assert len(args["content"]) < 100

        # Recent messages should be preserved
        last_tc = agent.messages[-2]["tool_calls"][0]
        args = json.loads(last_tc["function"]["arguments"])
        assert args["content"] == large_content

    def test_compresses_long_tool_results(self):
        """Old tool results over 800 chars are truncated."""
        agent = _make_agent([CompletionResult(content="ok")])
        agent.messages = [{"role": "system", "content": "sys"}]
        long_output = "\n".join(f"line {i}: some output" for i in range(50))
        for i in range(6):
            agent.messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": str(i), "type": "function",
                    "function": {
                        "name": "execute_shell",
                        "arguments": json.dumps({"command": f"cmd{i}"}),
                    },
                }],
            })
            agent.messages.append({
                "role": "tool", "tool_call_id": str(i),
                "content": long_output,
            })

        agent._compress_old_messages()

        # Old tool result should be truncated
        assert "elided" in agent.messages[2]["content"]
        assert len(agent.messages[2]["content"]) < len(long_output)

        # Recent tool result should be preserved
        assert agent.messages[-1]["content"] == long_output

    def test_no_compression_when_few_messages(self):
        """No compression when message count is below threshold."""
        agent = _make_agent([CompletionResult(content="ok")])
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        agent._compress_old_messages()
        # Nothing should change
        assert agent.messages[1]["content"] == "hello"
        assert agent.messages[2]["content"] == "hi"
