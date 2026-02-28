"""Tests for the markdown plan parser."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from natshell.agent.plan import Plan, PlanStep, parse_plan_file, parse_plan_text


# ─── parse_plan_text ─────────────────────────────────────────────────────────


class TestParsePlanText:
    """Test plan parsing from raw text."""

    def test_basic_plan(self):
        text = textwrap.dedent("""\
            # My Plan

            Some preamble text.

            ## Fix the bug

            Edit file.py and change X to Y.

            ## Add tests

            Write tests for the fix.
        """)
        plan = parse_plan_text(text)
        assert plan.title == "My Plan"
        assert "preamble" in plan.preamble.lower()
        assert len(plan.steps) == 2
        assert plan.steps[0].title == "Fix the bug"
        assert plan.steps[1].title == "Add tests"

    def test_step_numbering_is_1_based(self):
        text = textwrap.dedent("""\
            ## First step
            body1
            ## Second step
            body2
            ## Third step
            body3
        """)
        plan = parse_plan_text(text)
        assert plan.steps[0].number == 1
        assert plan.steps[1].number == 2
        assert plan.steps[2].number == 3

    def test_step_n_colon_format(self):
        """Parse ## Step 1: Title format."""
        text = textwrap.dedent("""\
            # Plan

            ## Step 1: Fix shapes

            Fix the shapes.

            ## Step 2: Add timing

            Add timing code.
        """)
        plan = parse_plan_text(text)
        assert len(plan.steps) == 2
        assert plan.steps[0].title == "Fix shapes"
        assert plan.steps[1].title == "Add timing"

    def test_plain_heading_format(self):
        """Parse plain ## Title format without step numbers."""
        text = textwrap.dedent("""\
            ## Initialize project

            Run npm init.

            ## Configure linting

            Add eslint config.
        """)
        plan = parse_plan_text(text)
        assert len(plan.steps) == 2
        assert plan.steps[0].title == "Initialize project"
        assert plan.steps[1].title == "Configure linting"

    def test_numbered_dot_format(self):
        """Parse ## 1. Title format."""
        text = textwrap.dedent("""\
            # Plan

            ## 1. Fix shapes

            Fix them.

            ## 2. Add timing

            Add it.
        """)
        plan = parse_plan_text(text)
        assert len(plan.steps) == 2
        assert plan.steps[0].title == "Fix shapes"
        assert plan.steps[1].title == "Add timing"

    def test_preamble_extraction(self):
        text = textwrap.dedent("""\
            # Plan Title

            This is the preamble with context.
            It has multiple lines.

            ## First step

            Do something.
        """)
        plan = parse_plan_text(text)
        assert "preamble with context" in plan.preamble
        assert "multiple lines" in plan.preamble

    def test_code_blocks_preserved(self):
        text = textwrap.dedent("""\
            ## Add function

            Add this code:

            ```cpp
            void init() {
                timer = 0;
            }
            ```

            After the existing code.
        """)
        plan = parse_plan_text(text)
        assert len(plan.steps) == 1
        assert "```cpp" in plan.steps[0].body
        assert "void init()" in plan.steps[0].body
        assert "```" in plan.steps[0].body

    def test_no_steps_raises_value_error(self):
        text = "# Just a title\n\nNo step headings here.\n"
        with pytest.raises(ValueError, match="No steps found"):
            parse_plan_text(text)

    def test_missing_h1_uses_default_title(self):
        text = textwrap.dedent("""\
            ## First step

            Do something.
        """)
        plan = parse_plan_text(text)
        assert plan.title == "Untitled Plan"

    def test_empty_preamble(self):
        text = textwrap.dedent("""\
            # Title

            ## Step one

            Body here.
        """)
        plan = parse_plan_text(text)
        assert plan.preamble == ""

    def test_step_body_stripped(self):
        text = textwrap.dedent("""\
            ## Step one


            Body with extra whitespace.


        """)
        plan = parse_plan_text(text)
        assert plan.steps[0].body == "Body with extra whitespace."

    def test_single_step(self):
        text = textwrap.dedent("""\
            # Single Step Plan

            ## Do the thing

            Just one step.
        """)
        plan = parse_plan_text(text)
        assert len(plan.steps) == 1
        assert plan.steps[0].number == 1

    def test_many_steps(self):
        lines = ["# Big Plan"]
        for i in range(10):
            lines.append(f"\n## Step {i+1}: Task {i+1}\n\nBody for step {i+1}.")
        text = "\n".join(lines)
        plan = parse_plan_text(text)
        assert len(plan.steps) == 10
        assert plan.steps[9].number == 10
        assert plan.steps[9].title == f"Task 10"


# ─── parse_plan_file ─────────────────────────────────────────────────────────


class TestParsePlanFile:
    """Test file-based plan parsing."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_plan_file("/nonexistent/path/plan.md")

    def test_reads_file(self, tmp_path: Path):
        plan_file = tmp_path / "plan.md"
        plan_file.write_text(textwrap.dedent("""\
            # Test Plan

            ## Step 1: Do something

            Do it.

            ## Step 2: Verify

            Check it.
        """))
        plan = parse_plan_file(str(plan_file))
        assert plan.title == "Test Plan"
        assert len(plan.steps) == 2

    def test_tilde_expansion(self, tmp_path: Path, monkeypatch):
        plan_file = tmp_path / "plan.md"
        plan_file.write_text("## Step one\n\nBody.\n")
        monkeypatch.setenv("HOME", str(tmp_path))
        plan = parse_plan_file("~/plan.md")
        assert len(plan.steps) == 1
