"""Tests for the context manager — token estimation, trimming, and summaries."""

from __future__ import annotations

import json

import pytest

from natshell.agent.context_manager import ContextManager, _MIN_RECENT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sys(content: str = "You are a helpful assistant.") -> dict:
    return {"role": "system", "content": content}


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _tool_call_msg(name: str, arguments: dict, tc_id: str = "tc1") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": tc_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments),
            },
        }],
    }


def _tool_result(content: str, tc_id: str = "tc1") -> dict:
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

class TestTokenEstimation:
    def test_approximate_estimation(self):
        """Fallback estimator uses len(text) // 3."""
        cm = ContextManager(context_budget=10000)
        msgs = [_sys("Hello world")]  # 11 chars → ~3 tokens
        tokens = cm.estimate_tokens(msgs)
        assert tokens == len("Hello world") // 3

    def test_exact_tokenizer(self):
        """When a tokenizer_fn is provided, use it."""
        cm = ContextManager(context_budget=10000, tokenizer_fn=lambda t: 42)
        msgs = [_sys("anything")]
        assert cm.estimate_tokens(msgs) == 42

    def test_tokenizer_fallback_on_error(self):
        """If the tokenizer raises, fall back to approximate."""
        def bad_tokenizer(text):
            raise RuntimeError("boom")

        cm = ContextManager(context_budget=10000, tokenizer_fn=bad_tokenizer)
        tokens = cm.estimate_tokens([_sys("Hello world")])
        assert tokens == len("Hello world") // 3

    def test_tool_call_arguments_counted(self):
        """Tool call arguments in assistant messages contribute to token count."""
        cm = ContextManager(context_budget=10000)
        msg = _tool_call_msg("execute_shell", {"command": "ls -la"})
        tokens = cm._msg_tokens(msg)
        assert tokens > 0


# ---------------------------------------------------------------------------
# No trimming needed
# ---------------------------------------------------------------------------

class TestNoTrimming:
    def test_within_budget(self):
        """Messages within budget pass through unchanged."""
        cm = ContextManager(context_budget=10000)
        msgs = [_sys(), _user("hello"), _assistant("hi")]
        result = cm.trim_messages(msgs)
        assert result == msgs

    def test_single_message(self):
        """A single system message passes through."""
        cm = ContextManager(context_budget=10000)
        msgs = [_sys()]
        assert cm.trim_messages(msgs) == msgs

    def test_empty_messages(self):
        """Empty list passes through."""
        cm = ContextManager(context_budget=10000)
        assert cm.trim_messages([]) == []


# ---------------------------------------------------------------------------
# Basic trimming
# ---------------------------------------------------------------------------

class TestTrimming:
    def test_system_prompt_preserved(self):
        """System prompt is always the first message after trimming."""
        # Very small budget forces trimming
        cm = ContextManager(context_budget=100)
        msgs = [_sys("sys")] + [_user(f"msg {i}" * 20) for i in range(20)]
        result = cm.trim_messages(msgs)
        assert result[0] == _sys("sys")

    def test_recent_messages_preserved(self):
        """The last _MIN_RECENT messages are always kept."""
        cm = ContextManager(context_budget=200)
        msgs = [_sys("s")]
        # Add many messages to force trimming
        for i in range(20):
            msgs.append(_user(f"message number {i} " * 10))
            msgs.append(_assistant(f"response number {i} " * 10))

        result = cm.trim_messages(msgs)
        # The last _MIN_RECENT messages from the original should be at the end
        original_recent = msgs[-_MIN_RECENT:]
        result_recent = result[-_MIN_RECENT:]
        assert result_recent == original_recent

    def test_summary_marker_inserted(self):
        """A summary marker is inserted after the system prompt when trimming occurs."""
        cm = ContextManager(context_budget=200)
        msgs = [_sys("s")]
        for i in range(20):
            msgs.append(_user(f"msg {i} " * 10))
            msgs.append(_assistant(f"resp {i} " * 10))

        result = cm.trim_messages(msgs)
        # Second message should be the summary marker
        assert result[1]["role"] == "system"
        assert "[Context note:" in result[1]["content"]

    def test_trimmed_count_tracked(self):
        """trimmed_count accumulates across multiple trim calls."""
        cm = ContextManager(context_budget=200)
        msgs = [_sys("s")]
        for i in range(20):
            msgs.append(_user(f"msg {i} " * 10))

        cm.trim_messages(msgs)
        first_count = cm.trimmed_count
        assert first_count > 0

        # Trim again
        cm.trim_messages(msgs)
        assert cm.trimmed_count > first_count


# ---------------------------------------------------------------------------
# Tool pair integrity
# ---------------------------------------------------------------------------

class TestToolPairIntegrity:
    def test_tool_pairs_not_split(self):
        """When trimming, tool call + result pairs are dropped together."""
        cm = ContextManager(context_budget=300)
        msgs = [_sys("s")]

        # Add several tool call pairs
        for i in range(10):
            tc_id = f"tc{i}"
            msgs.append(_tool_call_msg("execute_shell", {"command": f"cmd{i}"}, tc_id))
            msgs.append(_tool_result(f"output {i} " * 20, tc_id))

        # Add recent messages
        msgs.append(_user("final question"))
        msgs.append(_assistant("final answer"))

        result = cm.trim_messages(msgs)

        # Verify no orphan tool calls or results
        for j, msg in enumerate(result):
            if msg.get("tool_calls"):
                # Next message must be a tool result
                assert j + 1 < len(result), "Tool call at end without result"
                assert result[j + 1].get("role") == "tool", "Tool call not followed by tool result"
            if msg.get("role") == "tool" and j > 0:
                # Previous message should be a tool call (or system summary is OK)
                prev = result[j - 1]
                assert prev.get("tool_calls") or prev.get("role") == "system", \
                    "Orphan tool result without preceding tool call"


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_captures_user_messages(self):
        """Summary includes truncated user messages."""
        cm = ContextManager(context_budget=10000)
        dropped = [_user("What is the weather like today?")]
        summary = cm._build_summary(dropped)
        assert "User asked:" in summary
        assert "weather" in summary

    def test_summary_captures_commands(self):
        """Summary includes executed shell commands."""
        cm = ContextManager(context_budget=10000)
        dropped = [
            _tool_call_msg("execute_shell", {"command": "nmap -sn 192.168.1.0/24"}),
            _tool_result("Exit code: 0\nHost is up"),
        ]
        summary = cm._build_summary(dropped)
        assert "Ran: nmap" in summary
        assert "Exit code: 0" in summary

    def test_summary_captures_tool_names(self):
        """Summary includes non-shell tool names."""
        cm = ContextManager(context_budget=10000)
        dropped = [_tool_call_msg("read_file", {"path": "/etc/hosts"})]
        summary = cm._build_summary(dropped)
        assert "Called: read_file" in summary

    def test_summary_capped_at_500_chars(self):
        """Summary output is capped at 500 characters."""
        cm = ContextManager(context_budget=10000)
        dropped = [_user("x" * 200) for _ in range(30)]
        summary = cm._build_summary(dropped)
        assert len(summary) <= 503  # 500 + "..."

    def test_summary_max_15_facts(self):
        """Summary includes at most 15 facts."""
        cm = ContextManager(context_budget=10000)
        dropped = [_user(f"question {i}") for i in range(30)]
        summary = cm._build_summary(dropped)
        lines = [l for l in summary.split("\n") if l.startswith("- ")]
        assert len(lines) <= 15


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_only_system_and_recent(self):
        """When there are only system + recent messages, no trimming happens."""
        cm = ContextManager(context_budget=50)
        msgs = [_sys("s"), _user("hello"), _assistant("hi")]
        result = cm.trim_messages(msgs)
        # With only 2 non-system messages (< _MIN_RECENT), nothing to drop
        assert len(result) == len(msgs)

    def test_very_small_budget(self):
        """With a tiny budget, we still get system + summary + recent."""
        cm = ContextManager(context_budget=50)
        msgs = [_sys("s")]
        for i in range(20):
            msgs.append(_user(f"m{i}" * 20))
            msgs.append(_assistant(f"r{i}" * 20))

        result = cm.trim_messages(msgs)
        # At minimum: system + summary + recent window
        assert result[0]["role"] == "system"
        assert len(result) >= 2  # at least system + something

    def test_messages_with_empty_content(self):
        """Messages with empty content don't cause errors."""
        cm = ContextManager(context_budget=10000)
        msgs = [_sys(""), _user(""), _assistant("")]
        result = cm.trim_messages(msgs)
        assert len(result) == 3
