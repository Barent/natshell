"""Tests for the R2-1 TUI-token follow-up: routing streamed model tokens.

Three seams are pinned here:

* **Loop level** — an engine conforming to ``StreamingEngine`` is drained
  via ``stream_completion``; each ``StreamChunk`` surfaces in arrival order
  as a ``THINKING_TOKEN`` event (between ``THINKING`` and the first
  outcome); a failing stream falls back to the blocking ``chat_completion``
  before the recovery ladder is consulted, and a failure that survives
  both paths still reaches that ladder.  Engines without streaming keep
  the historical blocking path and emit no tokens.
* **Widget level** — ``ThinkingBlock`` accumulates deltas, keeps whatever
  has arrived so far copyable, mirrors the indicator's elapsed clock, and
  escapes markup in its body.
* **App level** — the first token promotes the ``ThinkingIndicator`` into
  a ``ThinkingBlock`` (timer not reset), and the terminal outcome event
  (``RESPONSE``/``PLANNING``) removes the placeholder while the canonical
  message is mounted — no duplication.  Headless ignores the token type
  entirely.
"""

from __future__ import annotations

from typing import Any

from natshell.agent.context import SystemContext
from natshell.agent.events import AgentEvent, EventType
from natshell.agent.loop import AgentLoop
from natshell.config import AgentConfig, SafetyConfig
from natshell.inference.engine import (
    CompletionResult,
    EngineInfo,
    StreamChunk,
    ToolCall,
)
from natshell.safety.classifier import SafetyClassifier
from natshell.tools.registry import create_default_registry

# ─── fakes ───────────────────────────────────────────────────────────────────


class FakeStreamingEngine:
    """A conforming streaming engine: yields chunks, then the result.

    ``results`` is popped per call so a run of several steps sees a
    different response each time.  ``fail_stream=True`` raises (before or
    mid-stream) so the loop must fall back to the blocking call, and
    ``fail_blocking`` makes that fallback raise too (→ recovery ladder).
    """

    def __init__(
        self,
        chunks: list[str],
        results: list[CompletionResult],
        fail_stream: bool = False,
        fail_mid_stream: bool = False,
        fail_blocking: bool = False,
    ) -> None:
        self._chunks = chunks
        self._results = list(results)
        self.fail_stream = fail_stream
        self.fail_mid_stream = fail_mid_stream
        self.fail_blocking = fail_blocking
        self.stream_calls = 0
        self.chat_calls = 0

    def _next_result(self) -> CompletionResult:
        assert self._results, "FakeStreamingEngine ran out of scripted results"
        return self._results.pop(0)

    def engine_info(self) -> EngineInfo:
        return EngineInfo(engine_type="local", model_name="fake-4b", n_ctx=32768)

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> CompletionResult:
        self.chat_calls += 1
        if self.fail_blocking:
            raise RuntimeError("boom")
        return self._next_result()

    async def stream_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        self.stream_calls += 1
        if self.fail_stream:
            raise RuntimeError("stream down")
        for i, c in enumerate(self._chunks):
            yield StreamChunk(text=c)
            if self.fail_mid_stream and i == len(self._chunks) - 1:
                raise RuntimeError("mid-stream crash")
        yield self._next_result()


class PlainBlockingEngine:
    """An engine WITHOUT stream_completion (like RemoteEngine): the loop
    must keep the historical blocking path."""

    def __init__(self, results: list[CompletionResult]) -> None:
        self._results = list(results)
        self.chat_calls = 0

    def engine_info(self) -> EngineInfo:
        return EngineInfo(engine_type="local", model_name="plain", n_ctx=32768)

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> CompletionResult:
        self.chat_calls += 1
        assert self._results, "PlainBlockingEngine ran out of scripted results"
        return self._results.pop(0)


def _make_agent(engine: Any) -> AgentLoop:
    tools = create_default_registry()
    safety = SafetyClassifier(
        SafetyConfig(mode="confirm", blocked=[]),
    )
    agent = AgentLoop(
        engine=engine,
        tools=tools,
        safety=safety,
        config=AgentConfig(max_steps=15, temperature=0.3, max_tokens=2048),
    )
    agent.initialize(SystemContext(hostname="t", distro="t", kernel="6.12", username="t"))
    return agent


async def _collect(agent: AgentLoop, message: str) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    async for event in agent.handle_user_message(
        message, confirm_callback=lambda tc: _auto_approve(tc)
    ):
        events.append(event)
    return events


async def _auto_approve(tc: ToolCall) -> bool:
    return True


# ─── loop-level token streaming ─────────────────────────────────────────────


class TestLoopTokenStreaming:
    async def test_tokens_stream_in_order_then_response(self):
        engine = FakeStreamingEngine(["Hel", "lo", "!"], [CompletionResult(content="Hello!")])
        agent = _make_agent(engine)
        events = await _collect(agent, "hi there")

        tokens = [e for e in events if e.type == EventType.THINKING_TOKEN]
        assert [t.data for t in tokens] == ["Hel", "lo", "!"]

        types = [e.type for e in events]
        assert (
            types.index(EventType.THINKING)
            < types.index(EventType.THINKING_TOKEN)
            < types.index(EventType.RESPONSE)
        )
        # Terminal result arrives through the same pipeline as before.
        assert engine.stream_calls == 1
        assert engine.chat_calls == 0
        response = [e for e in events if e.type == EventType.RESPONSE]
        assert response and response[0].data == "Hello!"

    async def test_streaming_tool_call_keeps_dispatch_events(self):
        engine = FakeStreamingEngine(
            ["run", " it"],
            [
                CompletionResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="1",
                            name="execute_shell",
                            arguments={"command": "echo hi"},
                        )
                    ],
                ),
            ],
        )
        agent = _make_agent(engine)
        events = await _collect(agent, "say hello with echo")

        types = [e.type for e in events]
        tokens = [e for e in events if e.type == EventType.THINKING_TOKEN]
        assert [t.data for t in tokens] == ["run", " it"]
        # Tools still dispatch exactly as before, after the tokens.
        assert types.index(EventType.THINKING_TOKEN) < types.index(EventType.EXECUTING)
        assert EventType.EXECUTING in types
        assert EventType.TOOL_RESULT in types

    async def test_stream_failure_falls_back_to_blocking(self):
        engine = FakeStreamingEngine(
            [],
            [CompletionResult(content="Hello!")],
            fail_stream=True,
        )
        agent = _make_agent(engine)
        events = await _collect(agent, "hi")

        types = [e.type for e in events]
        assert EventType.THINKING_TOKEN not in types
        assert types.count(EventType.RESPONSE) == 1
        assert engine.stream_calls == 1
        assert engine.chat_calls == 1
        responses = [e for e in events if e.type == EventType.RESPONSE]
        assert responses[0].data == "Hello!"

    async def test_partial_chunks_survive_mid_stream_crash(self):
        engine = FakeStreamingEngine(
            ["Hel"],
            [CompletionResult(content="Hello! (recovered)")],
            fail_mid_stream=True,
        )
        agent = _make_agent(engine)
        events = await _collect(agent, "hi")

        types = [e.type for e in events]
        tokens = [e for e in events if e.type == EventType.THINKING_TOKEN]
        assert [t.data for t in tokens] == ["Hel"]
        assert types.index(EventType.THINKING_TOKEN) < types.index(EventType.RESPONSE)
        responses = [e for e in events if e.type == EventType.RESPONSE]
        assert responses[0].data == "Hello! (recovered)"
        assert engine.chat_calls == 1

    async def test_both_paths_fail_reaches_recovery_ladder(self):
        engine = FakeStreamingEngine(
            [],
            [CompletionResult(content="unreachable")],
            fail_stream=True,
            fail_blocking=True,
        )
        agent = _make_agent(engine)
        events = await _collect(agent, "hi")

        types = [e.type for e in events]
        assert EventType.RESPONSE not in types
        errors = [e for e in events if e.type == EventType.ERROR]
        assert errors, "expected a recovery-banner ERROR event"
        assert "Inference error: boom" in errors[0].data

    async def test_plain_blocking_engine_emits_no_tokens(self):
        engine = PlainBlockingEngine([CompletionResult(content="Hello!")])
        agent = _make_agent(engine)
        events = await _collect(agent, "hi")

        types = [e.type for e in events]
        assert EventType.THINKING_TOKEN not in types
        assert EventType.RESPONSE in types
        assert engine.chat_calls == 1


# ─── widget level ────────────────────────────────────────────────────────────


class TestThinkingBlock:
    def test_append_accumulates_and_stays_copyable(self):
        from natshell.ui.widgets import ThinkingBlock

        block = ThinkingBlock()
        block.append("Hel")
        assert block.copyable_text == "Hel"
        block.append("lo!")
        assert block.copyable_text == "Hello!"

    def test_empty_chunk_is_noop(self):
        from natshell.ui.widgets import ThinkingBlock

        block = ThinkingBlock()
        block.append("")
        assert block.copyable_text == ""

    def test_elapsed_carried_from_indicator(self):
        from natshell.ui.widgets import ThinkingBlock

        block = ThinkingBlock(elapsed=45)  # tenths, like ThinkingIndicator._elapsed
        assert block._elapsed == 45
        header = block._header_markup()
        assert "4s" in header

    def test_body_escapes_markup(self):
        from rich.console import Group

        from natshell.ui.widgets import ThinkingBlock

        assert ThinkingBlock._format_body("[bold]x[/]") == "\\[bold]x\\[/]"
        # Fenced content goes through the same fence renderer as messages
        # (returns a Group renderable, not a naive str()).
        out = ThinkingBlock._format_body("```py\nprint(1)\n```")
        assert isinstance(out, Group)

    def test_header_reflects_typing_state(self):
        from natshell.ui.widgets import ThinkingBlock

        block = ThinkingBlock()
        assert "is thinking" in block._header_markup()
        block.append("hi")
        assert "is typing" in block._header_markup()


# ─── app-level routing (Textual pilot) ──────────────────────────────────────


class TestTuiTokenRouting:
    @staticmethod
    def _app():
        from natshell.agent.loop import AgentLoop
        from natshell.app import NatShellApp
        from natshell.config import AgentConfig
        from natshell.inference.engine import CompletionResult
        from natshell.safety.classifier import SafetyClassifier
        from natshell.tools.registry import create_default_registry

        engine = PlainBlockingEngine([CompletionResult(content="ok")])
        agent = AgentLoop(
            engine=engine,
            tools=create_default_registry(),
            safety=SafetyClassifier(SafetyConfig(mode="confirm", blocked=[])),
            config=AgentConfig(max_steps=15, temperature=0.3, max_tokens=2048),
        )
        agent.initialize(SystemContext(hostname="t", distro="t", kernel="6", username="t"))
        return NatShellApp(agent=agent, config=None, skip_permissions=True)

    @staticmethod
    def _ev(type_: EventType, data: Any = None) -> AgentEvent:
        return AgentEvent(type=type_, data=data)

    async def test_indicator_promotes_to_block_and_response_supersedes(self):
        from textual.containers import ScrollableContainer

        from natshell.ui.widgets import (
            AssistantMessage,
            ThinkingBlock,
            ThinkingIndicator,
        )

        app = self._app()
        async with app.run_test() as _pilot:
            conversation = app.query_one("#conversation", ScrollableContainer)
            tref: list = [None]
            await _pilot.pause()

            # THINKING → plain spinner, no block yet.
            app._render_agent_event(self._ev(EventType.THINKING), conversation, tref)
            await _pilot.pause()
            assert len(app.query(ThinkingIndicator)) == 1
            assert len(app.query(ThinkingBlock)) == 0

            # First token → spinner promoted into a block, carrying the clock.
            app._render_agent_event(self._ev(EventType.THINKING_TOKEN, "Hel"), conversation, tref)
            await _pilot.pause()
            assert len(app.query(ThinkingIndicator)) == 0
            blocks = app.query(ThinkingBlock)
            assert len(blocks) == 1
            assert blocks.first().copyable_text == "Hel"

            # More tokens grow the body in place (same block, timer intact).
            app._render_agent_event(self._ev(EventType.THINKING_TOKEN, "lo!"), conversation, tref)
            await _pilot.pause()
            assert len(app.query(ThinkingBlock)) == 1
            assert blocks.first().copyable_text == "Hello!"
            assert blocks.first()._elapsed >= 0

            # RESPONSE supersedes the placeholder — no duplication left mounted.
            app._render_agent_event(self._ev(EventType.RESPONSE, "Hello!"), conversation, tref)
            await _pilot.pause()
            assert len(app.query(ThinkingBlock)) == 0
            assert len(app.query(AssistantMessage)) == 1

    async def test_planning_supersedes_block(self):
        from textual.containers import ScrollableContainer

        from natshell.ui.widgets import (
            PlanningMessage,
            ThinkingBlock,
        )

        app = self._app()
        async with app.run_test() as _pilot:
            conversation = app.query_one("#conversation", ScrollableContainer)
            tref: list = [None]
            await _pilot.pause()

            app._render_agent_event(self._ev(EventType.THINKING), conversation, tref)
            app._render_agent_event(
                self._ev(EventType.THINKING_TOKEN, "let me "), conversation, tref
            )
            await _pilot.pause()
            assert len(app.query(ThinkingBlock)) == 1
            assert app.query(ThinkingBlock).first().copyable_text == "let me "

            app._render_agent_event(self._ev(EventType.PLANNING, "let me look"), conversation, tref)
            await _pilot.pause()
            assert len(app.query(ThinkingBlock)) == 0
            assert len(app.query(PlanningMessage)) == 1

    async def test_no_tokens_keeps_historical_indicator_flow(self):
        """Regression: the non-streaming path renders exactly as before —
        THINKING mounts a plain indicator, RESPONSE removes it, and no
        ThinkingBlock is ever created."""
        from textual.containers import ScrollableContainer

        from natshell.ui.widgets import (
            AssistantMessage,
            ThinkingBlock,
            ThinkingIndicator,
        )

        app = self._app()
        async with app.run_test() as _pilot:
            conversation = app.query_one("#conversation", ScrollableContainer)
            tref: list = [None]
            await _pilot.pause()

            app._render_agent_event(self._ev(EventType.THINKING), conversation, tref)
            await _pilot.pause()
            assert len(app.query(ThinkingIndicator)) == 1
            app._render_agent_event(self._ev(EventType.RESPONSE, "Hello!"), conversation, tref)
            await _pilot.pause()
            assert len(app.query(ThinkingBlock)) == 0
            assert len(app.query(ThinkingIndicator)) == 0
            assert len(app.query(AssistantMessage)) == 1

    async def test_token_without_prior_thinking_mounts_block(self):
        """Defensive path: a token that arrives with no placeholder still
        yields a live block instead of crashing."""
        from textual.containers import ScrollableContainer

        from natshell.ui.widgets import ThinkingBlock

        app = self._app()
        async with app.run_test() as _pilot:
            conversation = app.query_one("#conversation", ScrollableContainer)
            tref: list = [None]
            await _pilot.pause()

            app._render_agent_event(
                self._ev(EventType.THINKING_TOKEN, "surprise"), conversation, tref
            )
            await _pilot.pause()
            assert len(app.query(ThinkingBlock)) == 1
            assert app.query(ThinkingBlock).first().copyable_text == "surprise"
            assert tref[0] is not None


# ─── headless: token deltas are a documented no-op ──────────────────────────


class TestHeadlessTokenNoop:
    async def test_thinking_tokens_are_ignored(self):
        from natshell.headless import consume_events

        events = [
            AgentEvent(type=EventType.THINKING),
            AgentEvent(type=EventType.THINKING_TOKEN, data="Hel"),
            AgentEvent(type=EventType.THINKING_TOKEN, data="lo"),
            AgentEvent(type=EventType.RESPONSE, data="Hello!"),
        ]

        async def _stream():
            for e in events:
                yield e

        err_lines: list[str] = []
        responses: list[str] = []
        had_error = await consume_events(
            _stream(),
            emit=err_lines.append,
            on_response=responses.append,
        )
        assert not had_error
        assert responses == ["Hello!"]
        assert err_lines == []  # no per-token spam on the stderr line stream
