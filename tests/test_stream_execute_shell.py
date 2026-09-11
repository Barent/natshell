"""Tests for the streaming shell executor (R2-4).

``stream_execute_shell`` must be the streaming half of ``execute_shell``:
identical timeout, env, sudo ``-S``/``y\\n`` and stderr-scrub behaviour, plus
a forward of each stdout chunk to a callback as it arrives.  These tests pin
that shape: chunk ordering, terminal byte-parity with ``execute_shell``
(truncation + sudo scrub), the timeout 124 shape, sudo stdin transport, and
the missing-shell 127 shape.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

import natshell.tools.execute_shell as ex
from natshell.tools.execute_shell import (
    execute_shell,
    stream_execute_shell,
)


@pytest.fixture(autouse=True)
def _reset():
    yield
    ex.clear_sudo_password()


# ─── chunk ordering ──────────────────────────────────────────────────────────


class TestChunkOrdering:
    async def test_chunks_arrive_in_order_and_reassemble(self):
        """stdout is forwarded chunk-by-chunk, in order, and the pieces
        reassemble to the full output.  (Kept under the truncation budget so
        the terminal result is untruncated; truncation parity is pinned
        separately in TestParityWithBlocking.)"""
        chunks: list[str] = []
        result = await stream_execute_shell(
            "seq 1 500", on_chunk=chunks.append
        )
        # Each chunk is non-empty and in order.
        assert len(chunks) >= 1
        rejoined = "".join(chunks)
        # seq output is strictly ordered; rejoin must equal the terminal
        # (un-truncated) output byte-for-byte.
        expected = "\n".join(str(i) for i in range(1, 501)) + "\n"
        assert rejoined == expected
        # The terminal result agrees with what we streamed.
        assert result.output == expected
        assert not result.truncated
        assert result.exit_code == 0

    async def test_on_chunk_called_before_terminal_result(self):
        """Chunks must already have been delivered by the time the
        ToolResult is returned — the callback is the live channel."""
        seen = {"n": 0}

        def cb(chunk: str) -> None:
            seen["n"] += len(chunk)

        result = await stream_execute_shell("seq 1 1000", on_chunk=cb)
        # The callback saw the full payload.
        assert seen["n"] == len(result.output)
        assert seen["n"] > 0

    async def test_chunk_callback_receives_plain_str(self):
        """on_chunk always receives a decoded str, never bytes."""
        kinds = []

        def cb(chunk) -> None:
            kinds.append(type(chunk))

        await stream_execute_shell("echo abc", on_chunk=cb)
        assert all(k is str for k in kinds)


# ─── byte-parity with the blocking path ─────────────────────────────────────


class TestParityWithBlocking:
    async def test_stdout_stderr_exit_and_truncated_match(self):
        cmd = "echo out-a; echo err-a >&2; echo out-b; exit 4"
        a = await execute_shell(cmd, timeout=10)
        b = await stream_execute_shell(cmd, on_chunk=lambda c: None, timeout=10)
        assert a.output == b.output
        assert a.error == b.error
        assert a.exit_code == b.exit_code == 4
        assert a.truncated == b.truncated

    async def test_large_output_truncation_identical(self):
        # seq over a wide range exceeds the 4000-char budget → both paths
        # return the same head/tail-truncated text.
        a = await execute_shell("seq 1 200000")
        b = await stream_execute_shell("seq 1 200000", on_chunk=lambda c: None)
        assert a.truncated is True
        assert a.output == b.output
        assert a.truncated == b.truncated

    async def test_stderr_scrub_of_sudo_prompt_identical(self):
        # A command that emits a sudo password prompt on stderr must have it
        # scrubbed on both paths when a password is in play.  We force the
        # scrub without real sudo privileges by pointing the password cache
        # at a synthetic prompt line.
        ex.set_sudo_password("pw")
        cmd = "echo done; echo '[sudo] password for testuser:' >&2"
        a = await execute_shell(cmd, timeout=10)
        b = await stream_execute_shell(cmd, on_chunk=lambda c: None, timeout=10)
        # Both strip the prompt line but keep "done".
        assert "password for" not in a.error
        assert "password for" not in b.error
        assert a.error == b.error


# ─── timeout shape ───────────────────────────────────────────────────────────


class TestTimeout:
    async def test_timeout_returns_124_and_message(self):
        result = await stream_execute_shell("sleep 5", on_chunk=lambda c: None, timeout=1)
        assert result.exit_code == 124
        assert "timed out after 1 seconds" in result.error.lower()

    async def test_timeout_matches_blocking_path(self):
        a = await execute_shell("sleep 5", timeout=1)
        b = await stream_execute_shell("sleep 5", on_chunk=lambda c: None, timeout=1)
        assert a.exit_code == b.exit_code == 124
        assert a.error == b.error

    async def test_partial_output_not_lost_before_timeout(self):
        """Chunks emitted before the kill are still reflected in what streamed
        out (the live channel already delivered them to the TUI)."""
        got: list[str] = []
        # Prints some lines then sleeps past the timeout.
        cmd = "for i in $(seq 1 5); do echo line$i; sleep 1; done"
        result = await stream_execute_shell(cmd, on_chunk=got.append, timeout=1)
        assert result.exit_code == 124
        # At least the first line was streamed before the kill.
        assert "line1" in "".join(got)


# ─── sudo stdin transport ────────────────────────────────────────────────────


class TestSudoStdin:
    async def test_sudo_payload_reaches_child_stdin(self):
        """When the sudo path is active, the password/``y\\n`` payload is
        written to the child's stdin and the child sees it.  We exercise the
        real transport (``create_subprocess_exec`` + stdin PIPE) by pointing
        ``_prepare_sudo`` at a known payload and using ``cat`` as the reader."""
        orig = ex._prepare_sudo
        ex._prepare_sudo = lambda cmd: (cmd, "pw1\npw2\ny\n")
        try:
            result = await stream_execute_shell("cat", on_chunk=lambda c: None, timeout=5)
        finally:
            ex._prepare_sudo = orig
        # cat echoes back exactly the streamed payload.
        assert result.output == "pw1\npw2\ny\n"
        assert result.exit_code == 0

    async def test_sudo_injection_payload_shapes(self):
        """The pure payload-construction rules: package managers get the
        trailing ``y\\n`` block, other commands do not."""
        ex.set_sudo_password("pw")
        cmd, payload = ex._prepare_sudo("sudo apt install nmap")
        assert cmd == "sudo -S apt install nmap"
        assert payload == "pw\ny\ny\ny\n"

        cmd, payload = ex._prepare_sudo("printf x | sudo ls")
        assert cmd == "printf x | sudo -S ls"
        assert payload == "pw\n"

        # No sudo invocation → no stdin injection.
        cmd, payload = ex._prepare_sudo("ls -la")
        assert payload is None

    async def test_non_sudo_command_gets_devnull_stdin(self):
        # A non-sudo command must still run (stdin is /dev/null, not a pipe).
        result = await stream_execute_shell("echo fine", on_chunk=lambda c: None, timeout=5)
        assert result.exit_code == 0
        assert "fine" in result.output


# ─── missing shell ──────────────────────────────────────────────────────────


class TestMissingShell:
    async def test_missing_shell_returns_127(self):
        """If the shell binary cannot be spawned, mirror execute_shell's 127
        shape and message."""
        orig = asyncio.create_subprocess_exec

        async def _raise(*_a, **_k):
            raise FileNotFoundError("no bash")

        asyncio.create_subprocess_exec = _raise
        try:
            result = await stream_execute_shell("echo hi", on_chunk=lambda c: None)
        finally:
            asyncio.create_subprocess_exec = orig
        assert result.exit_code == 127
        assert "not found" in result.error.lower()
        # Windows would say PowerShell, POSIX says bash.
        import natshell.platform as plat

        expected = "PowerShell" if plat.is_windows() else "bash"
        assert expected in result.error


# ─── loop-level integration: stream_output routes chunks to events ──────────


class TestLoopStreamsExecuteShell:
    """The agent loop is the consumer that renders live chunks.  With
    ``stream_output=True`` an ``execute_shell`` call must surface TOOL_OUTPUT
    events (in arrival order, after EXECUTING, before TOOL_RESULT) and the
    terminal TOOL_RESULT must match what a blocking run would have produced.
    With the default (``stream_output=False``) the loop must NOT emit
    TOOL_OUTPUT and must run the historical ``tools.execute`` path — which is
    what every existing caller and the test suite relies on."""

    def _agent(self):
        from natshell.agent.context import SystemContext
        from natshell.agent.loop import AgentLoop
        from natshell.config import AgentConfig, SafetyConfig
        from natshell.inference.engine import CompletionResult, ToolCall
        from natshell.safety.classifier import SafetyClassifier
        from natshell.tools.registry import create_default_registry

        responses = [
            CompletionResult(tool_calls=[
                ToolCall(id="1", name="execute_shell", arguments={"command": "seq 1 20"})
            ]),
            CompletionResult(content="done"),
        ]
        engine = AsyncMock()
        engine.chat_completion = AsyncMock(side_effect=responses)
        tools = create_default_registry()
        safety = SafetyClassifier(SafetyConfig(mode="confirm", blocked=[]))
        agent = AgentLoop(
            engine=engine, tools=tools, safety=safety,
            config=AgentConfig(max_steps=15, temperature=0.3, max_tokens=2048),
        )
        agent.initialize(SystemContext(
            hostname="test", distro="Test", kernel="6.0", username="t"))
        return agent

    async def test_stream_output_emits_tool_output_events_in_order(self):
        from natshell.agent.loop import EventType

        agent = self._agent()
        events = []
        async for ev in agent.handle_user_message(
            "run seq", stream_output=True
        ):
            events.append(ev)

        types = [e.type for e in events]
        # At least one live chunk was streamed.
        assert EventType.TOOL_OUTPUT in types, f"no TOOL_OUTPUT in {types}"
        # Ordering: EXECUTING … one-or-more TOOL_OUTPUT … TOOL_RESULT … RESPONSE.
        first_exec = types.index(EventType.EXECUTING)
        first_out = types.index(EventType.TOOL_OUTPUT)
        first_result = types.index(EventType.TOOL_RESULT)
        resp = types.index(EventType.RESPONSE)
        assert first_exec < first_out < first_result < resp

        # The streamed chunks, in order, reassemble to the terminal stdout.
        chunks = [e.data for e in events if e.type == EventType.TOOL_OUTPUT]
        expected = "\n".join(str(i) for i in range(1, 21)) + "\n"
        assert "".join(chunks) == expected

        # The terminal result still carries the same output.
        result_ev = next(e for e in events if e.type == EventType.TOOL_RESULT)
        assert "20" in result_ev.tool_result.output

    async def test_default_does_not_stream(self):
        from natshell.agent.loop import EventType

        agent = self._agent()
        events = []
        async for ev in agent.handle_user_message("run seq"):  # stream_output default False
            events.append(ev)

        types = [e.type for e in events]
        # No live chunks on the non-streaming path.
        assert EventType.TOOL_OUTPUT not in types
        # …but the tool still ran and reported a result.
        assert EventType.TOOL_RESULT in types
        result_ev = next(e for e in events if e.type == EventType.TOOL_RESULT)
        assert "20" in result_ev.tool_result.output


# ─── TUI rendering: CommandBlock partial → final (Textual pilot) ────────────


class TestTUIRendersStream:
    """The TUI's TOOL_OUTPUT path: EXECUTING mounts a CommandBlock, each
    TOOL_OUTPUT appends live output via ``set_partial``, and TOOL_RESULT
    supersedes it with the authoritative final text via ``set_result``.
    Pinned through a real (headless) NatShellApp via Textual's pilot."""

    @staticmethod
    def _app():
        from natshell.agent.context import SystemContext
        from natshell.agent.loop import AgentLoop
        from natshell.app import NatShellApp
        from natshell.config import (
            AgentConfig,
            NatShellConfig,
            SafetyConfig,
        )
        from natshell.inference.engine import CompletionResult
        from natshell.safety.classifier import SafetyClassifier
        from natshell.tools.registry import create_default_registry

        engine = AsyncMock()
        engine.chat_completion = AsyncMock(side_effect=[CompletionResult(content="ok")])
        tools = create_default_registry()
        safety = SafetyClassifier(SafetyConfig(mode="confirm", blocked=[]))
        agent = AgentLoop(
            engine=engine, tools=tools, safety=safety,
            config=AgentConfig(max_steps=15, temperature=0.3, max_tokens=2048),
        )
        agent.initialize(SystemContext(
            hostname="t", distro="t", kernel="6", username="t"))
        return NatShellApp(agent=agent, config=NatShellConfig(), skip_permissions=True)

    async def test_partial_then_final(self):
        from natshell.agent.events import AgentEvent, EventType
        from natshell.inference.engine import ToolCall
        from natshell.tools.registry import ToolResult
        from natshell.ui.widgets import CommandBlock

        app = self._app()
        call = ToolCall(id="1", name="execute_shell", arguments={"command": "seq 1 3"})

        async with app.run_test() as pilot:
            from textual.containers import ScrollableContainer

            conversation = app.query_one("#conversation", ScrollableContainer)
            thinking_ref: list = [None]
            await pilot.pause()

            # EXECUTING mounts the block, TOOL_OUTPUT streams in, TOOL_RESULT
            # finalizes — exactly the order the loop yields them in.
            app._render_agent_event(
                AgentEvent(type=EventType.EXECUTING, tool_call=call),
                conversation, thinking_ref)
            app._render_agent_event(
                AgentEvent(type=EventType.TOOL_OUTPUT, tool_call=call, data="1\n"),
                conversation, thinking_ref)
            app._render_agent_event(
                AgentEvent(type=EventType.TOOL_OUTPUT, tool_call=call, data="2\n"),
                conversation, thinking_ref)
            # Live copy while still streaming shows whatever has arrived.
            live = app.query_one("#cmd-1", CommandBlock).copyable_text
            assert "1\n" in live and "2\n" in live
            app._render_agent_event(
                AgentEvent(
                    type=EventType.TOOL_RESULT, tool_call=call,
                    tool_result=ToolResult(output="1\n2\n3\n", exit_code=0),
                ),
                conversation, thinking_ref,
            )
            await pilot.pause()
            # Final supersedes the partial and carries the real (3rd) line.
            final = app.query_one("#cmd-1", CommandBlock).copyable_text
            assert final.endswith("1\n2\n3\n")
            assert "seq 1 3" in final
