"""Tests for R2-1 core streaming support (engine protocol + LocalEngine).

Covers the streaming seam: the StreamingEngine protocol, the LocalEngine
async generator (chunk order, terminal CompletionResult identity with the
blocking path, tool-call parsing, think-residue stripping,
context-overflow surfacing), and the fact that engines without streaming
(RemoteEngine) are feature-detectable as such.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from natshell.inference.engine import (
    CompletionResult,
    StreamChunk,
    StreamingEngine,
)


def _mock_llm_cls(chunks: list[dict[str, Any]], **_kw) -> MagicMock:
    """A fake llama_cpp.Llama whose create_chat_completion(stream=True)
    yields the given chunk dicts; stream=False returns the combined body
    (mirroring real llama-cpp-python so the blocking and streaming paths
    see the same content).
    """
    mock_llm = MagicMock()

    def fake_create_chat_completion(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return iter(chunks)
        parts = [
            (c.get("choices", [{}])[0].get("delta") or {}).get("content", "")
            for c in chunks
        ]
        content = "".join(parts)
        return {
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }

    mock_llm.create_chat_completion = fake_create_chat_completion
    return mock_llm


def _build_engine(chunks: list[dict[str, Any]], model: str = "/tmp/fake-model-4B.gguf"):
    """Build a LocalEngine over a mocked llama_cpp whose stream output is
    fixed — same harness shape as test_prompt_cache.py.
    """
    mock_llm = _mock_llm_cls(chunks)
    mock_llama_cls = MagicMock(return_value=mock_llm)
    mock_cache_cls = MagicMock(return_value=MagicMock())
    fake_llama_cpp = MagicMock(Llama=mock_llama_cls, LlamaRAMCache=mock_cache_cls)

    with (
        patch.dict(sys.modules, {"llama_cpp": fake_llama_cpp}),
        patch("natshell.gpu.gpu_backend_available", return_value=False),
        patch("natshell.gpu.best_gpu_index", return_value=0),
    ):
        from natshell.inference.local import LocalEngine

        engine = LocalEngine(
            model_path=model,
            n_ctx=4096,
            n_gpu_layers=0,
            prompt_cache=False,
        )

    return engine, mock_llm


def _drain(agen: Any) -> list[Any]:
    return asyncio.get_event_loop().run_until_complete(
        _collect(agen)
    )


async def _collect(agen: Any) -> list[Any]:
    out: list[Any] = []
    async for item in agen:
        out.append(item)
    return out


def _run(coro: Any) -> Any:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _stream_chunks(chunks: list[dict[str, Any]], **kwargs: Any) -> tuple[object, list[Any], Any]:
    """Drive LocalEngine.stream_completion to completion; return
    (generator, body StreamChunks, terminal CompletionResult)."""
    engine, _ = _build_engine(chunks)
    gen: object = engine.stream_completion(
        [{"role": "user", "content": "hi"}],
        **kwargs,
    )
    items = _run(_collect(gen))
    terminal = items[-1]
    assert isinstance(terminal, CompletionResult), (
        f"terminal item must be CompletionResult, got {type(terminal)}: {items}"
    )
    body_items = items[:-1]
    return gen, body_items, terminal


class TestStreamChunks:
    def test_text_stream_yields_chunks_then_result(self):
        chunks = [
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}], "usage": {}},
            {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]},
        ]
        _, items, terminal = _stream_chunks(chunks)

        # each text delta must come back as a StreamChunk, in order
        assert [c.text for c in items if isinstance(c, StreamChunk)] == ["Hel", "lo"]
        # the empty final delta must not surface
        assert all(c.text for c in items if isinstance(c, StreamChunk))
        # and the terminal parse must reconstruct the full content
        assert terminal.content == "Hello"
        assert isinstance(terminal, CompletionResult)

    def test_stream_result_content_matches_blocking(self):
        """stream's CompletionResult content must equal the blocking
        chat_completion result for the same chunks."""
        chunks = [
            {"choices": [{"delta": {"content": "partial "}}]},
            {"choices": [{"delta": {"content": "answer"}}]},
        ]
        engine, _ = _build_engine(chunks)
        items = _run(
            _collect(
                engine.stream_completion(
                    [{"role": "user", "content": "hi"}]
                )
            )
        )
        streaming_result = items[-1]
        blocking = _run(
            engine.chat_completion([{"role": "user", "content": "hi"}])
        )
        assert streaming_result.content == blocking.content
        assert isinstance(streaming_result, CompletionResult)
        assert isinstance(blocking, CompletionResult)


class TestStreamToolParsing:
    def test_stream_tool_call_is_parsed_in_terminal_result(self):
        """The R2-1 contract: the parser still runs on the *final*
        buffered content, not on the streamed chunks — so a native tool
        call spanning multiple chunks must be parsed out of the
        terminal result exactly as if the chunks had arrived in one
        blocking call."""
        # A Qwen-family tool call spread across two stream chunks.
        # Build the JSON body via json.dumps, and the tag markers via chr(),
        # to keep their raw sequence out of this test source (the write
        # pipeline mangles it).
        import json
        body = json.dumps({"name": "execute_shell", "arguments": {"command": "ls"}})
        tag_open = chr(60) + "tool_call" + chr(62)
        tag_close = chr(60) + "/tool_call" + chr(62)
        full = tag_open + body + tag_close

        chunks = [
            {"choices": [{"delta": {"content": full[:50]}}]},
            {"choices": [{"delta": {"content": full[50:]}}]},
        ]
        gen, items, terminal = _stream_chunks(chunks)

        # The tool call must land on the terminal CompletionResult —
        # that is what the agent loop consumes — not on chunk text.
        assert len(terminal.tool_calls) == 1
        call = terminal.tool_calls[0]
        assert call.name == "execute_shell"
        assert call.arguments == {"command": "ls"}
        # Chunk text may be raw (the tags themselves are fine to stream);
        # the *parse* is what has to be correct on the terminal result.


class TestStreamThinkResidue:
    def test_stream_result_strips_closed_think_blocks(self):
        """The terminal parse (think-residue strip) must apply to streamed
        content just like the blocking path: closed think blocks vanish
        from the surfaced text."""
        # Assemble the markers from chars — the pipeline rewrites the
        # literal thinking tags in tool traffic.
        think = chr(60) + "think" + chr(62)
        end = chr(60) + "/think" + chr(62)
        full = "hi " + think + "thinking about it" + end + " hello"
        chunks = [
            {"choices": [{"delta": {"content": full[:40]}}]},
            {"choices": [{"delta": {"content": full[40:]}}]},
        ]
        _, _, terminal = _stream_chunks(chunks)
        # terminal content must have the think block stripped — the grammar
        # parser runs on the buffered content in both paths.
        assert "thinking about it" not in (terminal.content or "")
        # ("thinking about it" appears only inside the think block)
        # sanity: the model text is preserved
        assert "hi" in (terminal.content or "")


class TestStreamOverflow:
    def test_context_overflow_surfaces(self):
        """A context-window overflow from llama.cpp in streaming mode must
        surface as ContextOverflowError, matching the blocking path."""
        from natshell.inference.remote import ContextOverflowError

        engine, mock_llm = _build_engine([])
        # override the bound callable with one that raises
        def raiser(*a: Any, **kw: Any):
            raise ValueError("context window exceeded (n_ctx=4096)")

        # bind the same Any-typed callable path stream_completion uses
        object.__setattr__(engine.llm, "create_chat_completion", raiser)

        with pytest.raises(ContextOverflowError):
            _run(
                _collect(
                    engine.stream_completion([{"role": "user", "content": "hi"}])
                )
            )


class TestProtocolConformance:
    def test_local_engine_is_a_streaming_engine(self):
        import sys as _s
        from unittest.mock import MagicMock as _M
        from unittest.mock import patch as _p
        mock_llm = MagicMock()
        fake_llama_cpp = _M(Llama=_M(return_value=mock_llm), LlamaRAMCache=_M())
        with (
            _p.dict(_s.modules, {"llama_cpp": fake_llama_cpp}),
            _p("natshell.gpu.gpu_backend_available", return_value=False),
            _p("natshell.gpu.best_gpu_index", return_value=0),
        ):
            from natshell.inference.local import LocalEngine
            engine = LocalEngine(
                model_path="/tmp/qwen3.gguf", n_ctx=4096,
                n_gpu_layers=0, prompt_cache=False,
            )
        assert isinstance(engine, StreamingEngine), (
            "LocalEngine must satisfy the StreamingEngine protocol"
        )

    def test_remote_engine_is_not_a_streaming_engine(self):
        from natshell.inference.remote import RemoteEngine
        engine = RemoteEngine(base_url="http://127.0.0.1:1", model="m")
        assert not hasattr(engine, "stream_completion"), (
            "RemoteEngine does not implement stream_completion yet (R2-1 follow-up); "
            "callers must feature-detect via hasattr"
        )
        # it still conforms to the base InferenceEngine
        assert hasattr(engine, "chat_completion")
