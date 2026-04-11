"""Tests for the context manager — token estimation, trimming, and summaries."""

from __future__ import annotations

import json

from natshell.agent.context_manager import _MIN_RECENT, ContextManager

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
        "tool_calls": [
            {
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


def _tool_result(content: str, tc_id: str = "tc1") -> dict:
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    def test_approximate_estimation(self):
        """Fallback estimator uses len(text) // 4."""
        cm = ContextManager(context_budget=10000)
        msgs = [_sys("Hello world")]  # 11 chars → ~2 tokens
        tokens = cm.estimate_tokens(msgs)
        assert tokens == len("Hello world") // 4

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
        assert tokens == len("Hello world") // 4

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
                assert prev.get("tool_calls") or prev.get("role") == "system", (
                    "Orphan tool result without preceding tool call"
                )


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_captures_user_messages(self):
        """Summary includes truncated user messages."""
        cm = ContextManager(context_budget=10000)
        dropped = [_user("What is the weather like today?")]
        summary = cm.build_summary(dropped)
        assert "User asked:" in summary
        assert "weather" in summary

    def test_summary_captures_commands(self):
        """Summary includes executed shell commands."""
        cm = ContextManager(context_budget=10000)
        dropped = [
            _tool_call_msg("execute_shell", {"command": "nmap -sn 192.168.1.0/24"}),
            _tool_result("Exit code: 0\nHost is up"),
        ]
        summary = cm.build_summary(dropped)
        assert "Ran: nmap" in summary
        assert "Exit code: 0" in summary

    def test_summary_captures_tool_names(self):
        """Summary includes read_file paths and generic tool names."""
        cm = ContextManager(context_budget=10000)
        dropped = [_tool_call_msg("read_file", {"path": "/etc/hosts"})]
        summary = cm.build_summary(dropped)
        assert "Read: /etc/hosts" in summary

    def test_summary_captures_file_changes(self):
        """Summary tracks files created/edited."""
        cm = ContextManager(context_budget=10000)
        dropped = [
            _tool_call_msg("write_file", {"path": "/tmp/foo.ts", "content": "x"}),
            _tool_result("Wrote /tmp/foo.ts"),
            _tool_call_msg("edit_file", {"path": "/tmp/bar.ts", "old_text": "a", "new_text": "b"}),
            _tool_result("Edited /tmp/bar.ts"),
        ]
        summary = cm.build_summary(dropped)
        assert "created: /tmp/foo.ts" in summary
        assert "edited: /tmp/bar.ts" in summary

    def test_summary_capped_at_800_chars(self):
        """Summary output is capped at 800 characters."""
        cm = ContextManager(context_budget=10000)
        dropped = [_user("x" * 200) for _ in range(30)]
        summary = cm.build_summary(dropped)
        assert len(summary) <= 803  # 800 + "..."

    def test_summary_max_12_actions(self):
        """Summary includes at most 12 action facts."""
        cm = ContextManager(context_budget=10000)
        dropped = [_user(f"question {i}") for i in range(30)]
        summary = cm.build_summary(dropped)
        lines = [line for line in summary.split("\n") if line.startswith("- ")]
        assert len(lines) <= 12


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


# ---------------------------------------------------------------------------
# Budget calibration from actual token counts
# ---------------------------------------------------------------------------


class TestCalibrateFromActual:
    def test_calibrate_shrinks_budget(self):
        """Budget shrinks proportionally when API reports more tokens than estimated."""
        cm = ContextManager(context_budget=10000)
        # Actual is 2x the estimate — budget should roughly halve
        cm.calibrate_from_actual(estimated_tokens=1000, actual_tokens=2000)
        assert cm.context_budget == 5000

    def test_calibrate_no_change_when_accurate(self):
        """Budget stays the same when estimate is within 15% of actual."""
        cm = ContextManager(context_budget=10000)
        cm.calibrate_from_actual(estimated_tokens=1000, actual_tokens=1100)
        assert cm.context_budget == 10000

    def test_calibrate_floor(self):
        """Budget never drops below 1024."""
        cm = ContextManager(context_budget=2000)
        # Actual is 10x the estimate — would shrink to 200, but floor is 1024
        cm.calibrate_from_actual(estimated_tokens=100, actual_tokens=1000)
        assert cm.context_budget == 1024

    def test_calibrate_floor_uses_initial_budget(self):
        """Budget floor is 40% of initial budget when that exceeds 1024."""
        cm = ContextManager(context_budget=5000)
        # Actual is 10x the estimate — would shrink to 500
        # Floor = max(1024, int(5000 * 0.4)) = max(1024, 2000) = 2000
        cm.calibrate_from_actual(estimated_tokens=100, actual_tokens=1000)
        assert cm.context_budget == 2000

    def test_calibrate_no_op_on_zero(self):
        """No-op when either value is zero."""
        cm = ContextManager(context_budget=10000)
        cm.calibrate_from_actual(estimated_tokens=0, actual_tokens=500)
        assert cm.context_budget == 10000
        cm.calibrate_from_actual(estimated_tokens=500, actual_tokens=0)
        assert cm.context_budget == 10000


# ---------------------------------------------------------------------------
# build_summary_with_refs (retrieval-augmented compaction)
# ---------------------------------------------------------------------------


class _FakeStore:
    """Lightweight stand-in for MemoryStore.put() in unit tests."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._counter = 0
        self.fail = False

    def put(self, **kwargs) -> str | None:
        if self.fail:
            return None
        self._counter += 1
        self.calls.append(kwargs)
        # Deterministic 64-char hex hash so tests can assert prefixes
        return f"{self._counter:064x}"


class TestBuildSummaryWithRefs:
    def test_existing_build_summary_unchanged(self):
        """The legacy build_summary path is not touched by the new method."""
        cm = ContextManager(context_budget=10000)
        msgs = [
            _user("hello"),
            _tool_call_msg("execute_shell", {"command": "ls"}),
            _tool_result("file.txt"),
        ]
        legacy = cm.build_summary(msgs)
        assert "Ran: ls" in legacy or "ls" in legacy

    def test_tool_pair_emits_single_ref(self):
        cm = ContextManager(context_budget=10000)
        store = _FakeStore()
        msgs = [
            _tool_call_msg("execute_shell", {"command": "nmap -sV 10.0.0.0/24"}),
            _tool_result("Exit code: 0\nfound 5 hosts"),
        ]
        text, hashes = cm.build_summary_with_refs(msgs, store.put, "sess1")
        assert len(hashes) == 1
        assert "[mem:" in text
        assert "execute_shell" in text
        assert "exit=0" in text
        # Only the bulky tool result should be stored — not the assistant stub.
        assert len(store.calls) == 1
        assert store.calls[0]["role"] == "tool"
        assert "found 5 hosts" in store.calls[0]["content"]
        assert store.calls[0]["tool_name"] == "execute_shell"

    def test_pair_aware_iteration_handles_consecutive_pairs(self):
        cm = ContextManager(context_budget=10000)
        store = _FakeStore()
        msgs = [
            _tool_call_msg("read_file", {"path": "/a"}, tc_id="tc1"),
            _tool_result("contents of a", tc_id="tc1"),
            _tool_call_msg("read_file", {"path": "/b"}, tc_id="tc2"),
            _tool_result("contents of b", tc_id="tc2"),
        ]
        text, hashes = cm.build_summary_with_refs(msgs, store.put, "sess1")
        assert len(hashes) == 2
        assert text.count("[mem:") == 2
        # Each tool call's result is stored once.
        assert len(store.calls) == 2

    def test_files_changed_section_preserved(self):
        cm = ContextManager(context_budget=10000)
        store = _FakeStore()
        msgs = [
            _tool_call_msg("write_file", {"path": "/tmp/foo.py"}),
            _tool_result("File written"),
        ]
        text, _ = cm.build_summary_with_refs(msgs, store.put, "sess1")
        assert "Files changed:" in text
        assert "created: /tmp/foo.py" in text

    def test_user_message_short_no_ref(self):
        cm = ContextManager(context_budget=10000)
        store = _FakeStore()
        msgs = [_user("short question")]
        text, hashes = cm.build_summary_with_refs(msgs, store.put, "sess1")
        assert hashes == []
        assert "User asked: short question" in text
        assert len(store.calls) == 0

    def test_user_message_long_gets_ref(self):
        cm = ContextManager(context_budget=10000)
        store = _FakeStore()
        long_q = "x" * 500
        msgs = [_user(long_q)]
        text, hashes = cm.build_summary_with_refs(msgs, store.put, "sess1")
        assert len(hashes) == 1
        assert "[mem:" in text
        assert "user_message" in text

    def test_inspection_hint_when_many_refs(self):
        cm = ContextManager(context_budget=10000)
        store = _FakeStore()
        msgs = []
        for i in range(4):
            msgs.append(_tool_call_msg("execute_shell", {"command": f"cmd{i}"}, tc_id=f"t{i}"))
            msgs.append(_tool_result(f"output{i}", tc_id=f"t{i}"))
        text, hashes = cm.build_summary_with_refs(msgs, store.put, "sess1")
        assert len(hashes) == 4
        assert "recall_memory" in text

    def test_inspection_hint_absent_when_few_refs(self):
        cm = ContextManager(context_budget=10000)
        store = _FakeStore()
        msgs = [
            _tool_call_msg("execute_shell", {"command": "ls"}),
            _tool_result("a\nb\nc"),
        ]
        text, _ = cm.build_summary_with_refs(msgs, store.put, "sess1")
        assert "recall_memory" not in text

    def test_store_failure_falls_back_to_plain_action(self):
        cm = ContextManager(context_budget=10000)
        store = _FakeStore()
        store.fail = True
        msgs = [
            _tool_call_msg("execute_shell", {"command": "ls"}),
            _tool_result("output"),
        ]
        text, hashes = cm.build_summary_with_refs(msgs, store.put, "sess1")
        assert hashes == []
        assert "[mem:" not in text
        assert "Called: execute_shell" in text

    def test_ref_line_capped(self):
        cm = ContextManager(context_budget=10000)
        store = _FakeStore()
        # Long command should still produce a capped ref line
        long_cmd = "echo " + "x" * 500
        msgs = [
            _tool_call_msg("execute_shell", {"command": long_cmd}),
            _tool_result("Exit code: 0"),
        ]
        text, _ = cm.build_summary_with_refs(msgs, store.put, "sess1")
        # Find the [mem:...] line and check its length
        for line in text.splitlines():
            if "[mem:" in line:
                assert len(line) < 200  # well under any reasonable cap

    def test_extract_exit_code(self):
        assert ContextManager._extract_exit_code("Exit code: 0\nfoo") == 0
        assert ContextManager._extract_exit_code("Exit code: 127\nfoo") == 127
        assert ContextManager._extract_exit_code("no exit line") is None
        assert ContextManager._extract_exit_code("Exit code: nan") is None

    def test_tool_call_preview(self):
        assert ContextManager._tool_call_preview(
            "execute_shell", '{"command": "ls -la"}'
        ) == '"ls -la"'
        assert ContextManager._tool_call_preview(
            "read_file", '{"path": "/etc/hosts"}'
        ) == '"/etc/hosts"'
        assert ContextManager._tool_call_preview("unknown", "") == ""

    def test_assistant_text_is_stored(self):
        cm = ContextManager(context_budget=10000)
        store = _FakeStore()
        msgs = [_assistant("I will now run the test suite to verify.")]
        text, hashes = cm.build_summary_with_refs(msgs, store.put, "sess1")
        assert len(hashes) == 1
        assert any(c["role"] == "assistant" for c in store.calls)
