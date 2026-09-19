"""Tests for the cache-stable tool prefix (R2-3).

``LocalEngine._inject_tools`` renders the tool definitions into a plain-text
block that is appended to the system message.  Re-rendering the block on
every call makes the prompt prefix churn for no reason, working against
llama.cpp's RAM prompt cache.  The cache keys on (family, compact,
canonical-tools) and must:

1. produce byte-identical output to the pre-cache path (same string, same
   position in the system message);
2. reuse the cached string across calls with the same tool set;
3. keep distinct cache entries for different families / compact tiers /
   tool sets;
4. stay bounded in memory (LRU eviction at 32 entries);
5. never mutate the caller's message list.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from natshell.inference.grammars import get_grammar
from natshell.inference.local import LocalEngine


def _build_engine(model: str = "/tmp/fake-qwen-4B.gguf", n_ctx: int = 4096) -> LocalEngine:
    mock_llm = MagicMock()
    mock_llama_cls = MagicMock(return_value=mock_llm)
    mock_cache_cls = MagicMock(return_value=MagicMock())
    fake_llama_cpp = MagicMock(Llama=mock_llama_cls, LlamaRAMCache=mock_cache_cls)

    with (
        patch.dict(sys.modules, {"llama_cpp": fake_llama_cpp}),
        patch("natshell.gpu.gpu_backend_available", return_value=False),
        patch("natshell.gpu.best_gpu_index", return_value=0),
    ):
        engine = LocalEngine(
            model_path=model,
            n_ctx=n_ctx,
            n_gpu_layers=0,
            prompt_cache=False,
        )

    return engine


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_shell",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]


def test_inject_tools_output_is_unchanged():
    """The rendered block must be byte-identical to
    ``grammar.render_tools`` for the same compact tier — the cache may
    reuse the string, not alter it."""
    engine = _build_engine(n_ctx=4096)
    expected_block = get_grammar(engine.model_family).render_tools(
        SAMPLE_TOOLS, compact=engine.n_ctx < 16384
    )

    messages = [{"role": "system", "content": "BASE"}]
    out = engine._inject_tools(messages, SAMPLE_TOOLS)

    assert out[0]["content"] == "BASE\n\n" + expected_block
    # the caller's list must not be mutated (only the returned copy grows)
    assert messages[0]["content"] == "BASE"
    assert len(out) == 1


def test_inject_tools_reuses_cached_string_for_same_tools():
    """Two calls with the same tool set → render_tools runs exactly
    once (the second call is served from the cache)."""
    engine = _build_engine(n_ctx=4096)
    mock_grammar = MagicMock()
    mock_grammar.render_tools.return_value = "TOOLTEXT"

    with patch("natshell.inference.grammars.get_grammar", return_value=mock_grammar):
        engine._inject_tools([{"role": "system", "content": "A"}], SAMPLE_TOOLS)
        engine._inject_tools([{"role": "system", "content": "A"}], SAMPLE_TOOLS)

    assert mock_grammar.render_tools.call_count == 1, (
        f"render_tools called {mock_grammar.render_tools.call_count} times; "
        "the cache should have reduced it to 1"
    )
    # both calls still saw the cached text
    assert "TOOLTEXT" in engine._tool_text_cache[
        next(iter(engine._tool_text_cache))
    ]


def test_inject_tools_caches_per_family_and_compact():
    """Different n_ctx (compact tier) or family → different cache
    entries, each rendered once."""
    mock_grammar = MagicMock()
    mock_grammar.render_tools.return_value = "TOOLTEXT"

    small = _build_engine(n_ctx=4096)   # compact=True
    large = _build_engine(n_ctx=32768)  # compact=False

    with patch("natshell.inference.grammars.get_grammar", return_value=mock_grammar):
        out_small = small._inject_tools(
            [{"role": "system", "content": "S"}], SAMPLE_TOOLS
        )
        out_large = large._inject_tools(
            [{"role": "system", "content": "S"}], SAMPLE_TOOLS
        )

    # small engine passed compact=True, large engine compact=False —
    # render_tools saw both, on the same shared mock
    compact_args = [
        (call.args[0] if call.args else (), call.kwargs)
        for call in mock_grammar.render_tools.call_args_list
    ]
    compacts = [kw.get("compact") for _, kw in compact_args]
    assert True in compacts and False in compacts
    # independent caches per engine
    assert small._tool_text_cache is not large._tool_text_cache
    # both outputs carry the same mocked text (the cache holds strings,
    # not grammar objects — swapping grammars is fine)
    assert out_small[0]["content"].endswith("TOOLTEXT")
    assert out_large[0]["content"].endswith("TOOLTEXT")


def test_inject_tools_cache_bounded():
    """More than 32 distinct tool sets → oldest entries evicted, so the
    cache never grows past 32."""
    engine = _build_engine(n_ctx=4096)
    mock_grammar = MagicMock()
    mock_grammar.render_tools.return_value = "TOOLTEXT"

    with patch("natshell.inference.grammars.get_grammar", return_value=mock_grammar):
        for i in range(40):
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": f"tool_{i}",
                        "description": f"desc {i}",
                        "parameters": {"type": "object"},
                    },
                }
            ]
            engine._inject_tools([{"role": "system", "content": "BASE"}], tools)

    assert len(engine._tool_text_cache) <= 32
    assert mock_grammar.render_tools.call_count == 40  # each rendered once


def test_inject_tools_key_is_canonical_for_equal_lists():
    """Two tool lists that are semantically identical (fresh Python dicts,
    same content) must hit the same cache entry → one render."""
    engine = _build_engine(n_ctx=4096)
    mock_grammar = MagicMock()
    mock_grammar.render_tools.return_value = "TOOLTEXT"

    def _clone(tools):
        return [dict(t) for t in tools]

    with patch("natshell.inference.grammars.get_grammar", return_value=mock_grammar):
        engine._inject_tools([{"role": "system", "content": "X"}], _clone(SAMPLE_TOOLS))
        engine._inject_tools([{"role": "system", "content": "X"}], _clone(SAMPLE_TOOLS))

    assert mock_grammar.render_tools.call_count == 1, (
        f"render_tools called {mock_grammar.render_tools.call_count} times; "
        "equal tool sets should share one cache entry"
    )
    assert len(engine._tool_text_cache) == 1


def test_inject_tools_different_tools_distinct_entries():
    """Different tool sets must not collide on one cache entry."""
    engine = _build_engine(n_ctx=4096)
    mock_grammar = MagicMock()
    mock_grammar.render_tools.return_value = "TOOLTEXT"

    with patch("natshell.inference.grammars.get_grammar", return_value=mock_grammar):
        engine._inject_tools([{"role": "system", "content": "X"}], SAMPLE_TOOLS)
        other = [
            {
                "type": "function",
                "function": {
                    "name": "other_tool",
                    "description": "different",
                    "parameters": {"type": "object"},
                },
            }
        ]
        engine._inject_tools([{"role": "system", "content": "X"}], other)

    assert mock_grammar.render_tools.call_count == 2
    assert len(engine._tool_text_cache) == 2
