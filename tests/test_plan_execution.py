"""Tests for plan execution prompt building and /exeplan integration."""

from __future__ import annotations

import textwrap

import pytest

from natshell.agent.plan import Plan, PlanStep, parse_plan_text
from natshell.app import _build_step_prompt


# ─── _build_step_prompt ──────────────────────────────────────────────────────


class TestBuildStepPrompt:
    """Test the per-step prompt builder."""

    def _make_plan(self) -> Plan:
        return parse_plan_text(textwrap.dedent("""\
            # Fix Tetris

            ## Fix piece shapes in theme.cpp

            Edit theme.cpp and fix the I-piece shape array.

            ```cpp
            int shapes[4] = {1, 2, 3, 4};
            ```

            ## Initialize timing

            Add `lastDropTime = 0;` after `updateDropInterval();`

            ## Add scoring

            Implement the score counter.
        """))

    def test_first_step_no_previously_completed(self):
        plan = self._make_plan()
        prompt = _build_step_prompt(plan.steps[0], plan, [])
        assert "Previously completed" not in prompt
        assert "step 1 of 3" in prompt

    def test_later_step_includes_summaries(self):
        plan = self._make_plan()
        summaries = ["1. Fix piece shapes in theme.cpp \u2713"]
        prompt = _build_step_prompt(plan.steps[1], plan, summaries)
        assert "Previously completed" in prompt
        assert "Fix piece shapes" in prompt
        assert "\u2713" in prompt
        assert "step 2 of 3" in prompt

    def test_prompt_includes_full_body(self):
        plan = self._make_plan()
        prompt = _build_step_prompt(plan.steps[0], plan, [])
        assert "```cpp" in prompt
        assert "int shapes[4]" in prompt

    def test_do_not_read_plan_files_directive(self):
        plan = self._make_plan()
        prompt = _build_step_prompt(plan.steps[0], plan, [])
        assert "do not read any plan files" in prompt

    def test_step_title_in_prompt(self):
        plan = self._make_plan()
        prompt = _build_step_prompt(plan.steps[2], plan, [])
        assert "## Add scoring" in prompt

    def test_multiple_summaries(self):
        plan = self._make_plan()
        summaries = [
            "1. Fix piece shapes in theme.cpp \u2713",
            "2. Initialize timing \u2713",
        ]
        prompt = _build_step_prompt(plan.steps[2], plan, summaries)
        assert "Fix piece shapes" in prompt
        assert "Initialize timing" in prompt
        assert "step 3 of 3" in prompt


# ─── /exeplan in SLASH_COMMANDS ──────────────────────────────────────────────


class TestExeplanInSlashCommands:
    def test_exeplan_listed(self):
        from natshell.app import SLASH_COMMANDS
        commands = [cmd for cmd, _ in SLASH_COMMANDS]
        assert "/exeplan" in commands
