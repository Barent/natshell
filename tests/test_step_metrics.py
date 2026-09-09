"""Tests for natshell.agent.step_metrics.

Covers the per-step outcome handlers and the per-run stats carrier
extracted from ``AgentLoop.handle_user_message`` (R1-7).  The event texts
and the two-branch degenerate / three-branch token-limit behaviour are
pinned here at the module level; the loop-level integration is pinned by
``tests/test_agent.py`` (``TestDegenerateAgentLoop``,
``TestTruncatedResponse``, ``test_max_steps_cutoff``).
"""

from __future__ import annotations

import re
import time

import pytest

from natshell.agent.events import EventType
from natshell.agent.step_metrics import (
    RunStats,
    StepControl,
    StepOutcome,
    build_metrics,
    build_run_stats,
    handle_degenerate_output,
    handle_token_limit,
    strip_think_residue,
)
from natshell.inference.engine import CompletionResult


def _make_result(
    content: str | None = "ok",
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> CompletionResult:
    return CompletionResult(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics builders
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildMetrics:
    def test_response_time_only(self):
        m = build_metrics(_make_result(), 250)
        assert m == {"response_time_ms": 250}

    def test_tokens_and_tps(self):
        m = build_metrics(_make_result(completion_tokens=200), 2000)
        assert m["completion_tokens"] == 200
        assert m["tokens_per_sec"] == pytest.approx(100.0)
        assert m["response_time_ms"] == 2000

    def test_zero_elapsed_skips_tps(self):
        m = build_metrics(_make_result(completion_tokens=10), 0)
        assert "tokens_per_sec" not in m

    def test_prompt_tokens_included(self):
        m = build_metrics(_make_result(prompt_tokens=55), 10)
        assert m["prompt_tokens"] == 55


class TestBuildRunStats:
    def test_full_stats(self):
        s = build_run_stats(4, 1000, 900, 100, 50)
        assert s["steps"] == 4
        assert s["total_wall_ms"] == 1000
        assert s["total_inference_ms"] == 900
        assert s["total_prompt_tokens"] == 100
        assert s["total_completion_tokens"] == 50
        assert s["total_tokens"] == 150
        assert s["avg_tokens_per_sec"] == pytest.approx(50 / 0.9)

    def test_no_tokens_no_totals(self):
        s = build_run_stats(1, 10, 5, 0, 0)
        assert "total_tokens" not in s
        assert "avg_tokens_per_sec" not in s

    def test_zero_inference_no_tps(self):
        s = build_run_stats(1, 10, 0, 0, 20)
        assert "avg_tokens_per_sec" not in s


# ─────────────────────────────────────────────────────────────────────────────
# RunStats carrier
# ─────────────────────────────────────────────────────────────────────────────


class TestRunStats:
    def test_accumulate_folds_everything(self):
        stats = RunStats(t0=1000.0)
        stats.accumulate(_make_result(prompt_tokens=10, completion_tokens=5), 100)
        stats.accumulate(_make_result(prompt_tokens=7, completion_tokens=3), 200)
        assert stats.inference_ms == 300
        assert stats.prompt_tokens == 17
        assert stats.completion_tokens == 8
        assert stats.last_elapsed_ms == 200

    def test_accumulate_tolerates_none_token_counts(self):
        stats = RunStats(t0=0.0)
        stats.accumulate(_make_result(), 42)
        assert stats.inference_ms == 42
        assert stats.prompt_tokens == 0
        assert stats.completion_tokens == 0

    def test_run_stats_uses_t0_for_wall_time(self):
        stats = RunStats(t0=1_000_000_000.0)
        stats.accumulate(_make_result(), 333)
        s = stats.run_stats(3, now=1_000_000_500.0)
        assert s["total_wall_ms"] == 500_000  # 500 s gap
        assert s["steps"] == 3
        assert s["total_inference_ms"] == 333


# ─────────────────────────────────────────────────────────────────────────────
# Degenerate-output handler
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleDegenerateOutput:
    def test_compacted_retries(self):
        outcome = handle_degenerate_output(
            _make_result(None), compact_stats={"compacted": True}
        )
        assert outcome.control is StepControl.RETRY
        assert [e.type for e in outcome.events] == [EventType.ERROR]
        assert "retrying" in outcome.events[0].data
        assert "degenerate" in outcome.events[0].data

    def test_not_compacted_stops(self):
        outcome = handle_degenerate_output(
            _make_result(None), compact_stats={"compacted": False}
        )
        assert outcome.control is StepControl.STOP
        assert [e.type for e in outcome.events] == [EventType.ERROR]
        assert "/clear" in outcome.events[0].data


# ─────────────────────────────────────────────────────────────────────────────
# Token-limit (finish_reason == "length") handler
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleTokenLimit:
    def test_partial_answer_surfaced_with_warning(self):
        stats = RunStats(t0=0.0)
        stats.accumulate(_make_result(), 50)
        now = time.monotonic()
        outcome = handle_token_limit(
            CompletionResult(content="final visible words"),
            steps_used=1,
            stats=stats,
            now=now,
        )
        assert outcome.control is StepControl.STOP
        assert outcome.partial == "final visible words"
        types = [e.type for e in outcome.events]
        assert types == [EventType.RESPONSE, EventType.ERROR]
        assert outcome.events[0].data == "final visible words"
        # First step: no RUN_STATS epilogue (matches the original inline code)
        assert EventType.RUN_STATS not in types

    def test_partial_answer_includes_run_stats_from_step_two(self):
        stats = RunStats(t0=1000.0)
        stats.accumulate(_make_result(prompt_tokens=4), 20)
        outcome = handle_token_limit(
            CompletionResult(content="still talking"),
            steps_used=2,
            stats=stats,
            now=1000.5,
        )
        assert outcome.control is StepControl.STOP
        types = [e.type for e in outcome.events]
        assert types == [EventType.RESPONSE, EventType.ERROR, EventType.RUN_STATS]
        assert outcome.events[2].metrics["steps"] == 2
        assert outcome.events[2].metrics["total_inference_ms"] == 20

    def test_all_tokens_spent_thinking_yields_error_only(self):
        stats = RunStats(t0=0.0)
        stats.accumulate(_make_result(), 9)
        # Build the think-tag input from codepoints (the pipeline mangles the
        # raw literal in tool traffic; the file-based tests use the real one).
        t_open = chr(60) + "think" + chr(62)
        t_close = chr(60) + "/think" + chr(62)
        thinking = t_open + "thinking hard about everything" + t_close
        outcome = handle_token_limit(
            CompletionResult(content=thinking),
            steps_used=3,
            stats=stats,
            now=time.monotonic(),
        )
        assert outcome.control is StepControl.STOP
        assert outcome.partial is None
        assert [e.type for e in outcome.events] == [EventType.ERROR]
        assert "truncated" in outcome.events[0].data

    def test_empty_content_yields_error_only(self):
        stats = RunStats(t0=0.0)
        outcome = handle_token_limit(
            CompletionResult(content=None),
            steps_used=1,
            stats=stats,
            now=time.monotonic(),
        )
        assert outcome.control is StepControl.STOP
        assert outcome.partial is None
        assert [e.type for e in outcome.events] == [EventType.ERROR]
        assert "reset" in outcome.events[0].data


# ─────────────────────────────────────────────────────────────────────────────
# Think-stripping — must stay byte-identical to the original inline regexes
# ─────────────────────────────────────────────────────────────────────────────


class TestStripThinkResidue:
    def test_closed_block_removed(self):
        t_open = chr(60) + "think" + chr(62)
        t_close = chr(60) + "/think" + chr(62)
        raw = t_open + "pondering" + t_close + "answer"
        assert strip_think_residue(raw) == "answer"

    def test_unclosed_trailing_block_removed(self):
        t_open = chr(60) + "think" + chr(62)
        raw = t_open + "dangling to the end"
        assert strip_think_residue(raw) == ""

    def test_partial_after_trailing_open_stripped(self):
        t_open = chr(60) + "think" + chr(62)
        raw = "visible" + t_open + "then never closed"
        assert strip_think_residue(raw) == "visible"

    def test_plain_text_untouched(self):
        assert strip_think_residue("plain text") == "plain text"

    def test_empty_input(self):
        assert strip_think_residue("") == ""

    def test_newlines_inside_block(self):
        t_open = chr(60) + "think" + chr(62)
        t_close = chr(60) + "/think" + chr(62)
        raw = "a\nb" + t_open + "x\ny" + t_close + "c\nd"
        assert strip_think_residue(raw) == "a\nbc\nd"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
