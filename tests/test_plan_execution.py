"""Tests for plan execution prompt building and /exeplan integration."""

from __future__ import annotations

import textwrap
from pathlib import Path

from natshell.agent.plan import Plan, parse_plan_file, parse_plan_text
from natshell.agent.plan_executor import _build_step_prompt, _shallow_tree

# ─── _build_step_prompt ──────────────────────────────────────────────────────


class TestBuildStepPrompt:
    """Test the per-step prompt builder."""

    def _make_plan(self, source_dir: Path | None = None) -> Plan:
        plan = parse_plan_text(
            textwrap.dedent("""\
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
        """)
        )
        plan.source_dir = source_dir
        return plan

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
        assert "Do NOT read the plan file" in prompt

    def test_termination_directive(self):
        plan = self._make_plan()
        prompt = _build_step_prompt(plan.steps[0], plan, [])
        assert "IMMEDIATELY provide a short text summary" in prompt
        assert "STOP" in prompt
        assert "Do not modify files not mentioned in this step" in prompt

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

    def test_includes_project_layout(self, tmp_path: Path):
        """When source_dir is set, the prompt includes a directory tree."""
        # Create a mock project structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "game.cpp").write_text("")
        (tmp_path / "src" / "theme.cpp").write_text("")
        (tmp_path / "Makefile").write_text("")

        plan = self._make_plan(source_dir=tmp_path)
        prompt = _build_step_prompt(plan.steps[0], plan, [])
        assert "Project layout:" in prompt
        assert "src/" in prompt
        assert "game.cpp" in prompt
        assert "Makefile" in prompt

    def test_no_layout_without_source_dir(self):
        """When source_dir is None, no project layout is included."""
        plan = self._make_plan(source_dir=None)
        prompt = _build_step_prompt(plan.steps[0], plan, [])
        assert "Project layout:" not in prompt

    def test_includes_preamble_when_present(self):
        """When plan has a preamble, it is injected into the step prompt."""
        plan = self._make_plan()
        # The fixture plan has title "Fix Tetris" but no preamble text
        plan.preamble = "Tech stack: C++17 with SDL2. Use snake_case naming."
        prompt = _build_step_prompt(plan.steps[0], plan, [])
        assert "Project context:" in prompt
        assert "C++17 with SDL2" in prompt
        assert "snake_case" in prompt

    def test_no_preamble_when_empty(self):
        """When plan preamble is empty, no project context section appears."""
        plan = self._make_plan()
        plan.preamble = ""
        prompt = _build_step_prompt(plan.steps[0], plan, [])
        assert "Project context:" not in prompt

    def test_budget_guidance_in_prompt(self):
        """Step prompt includes budget guidance with the step limit."""
        plan = self._make_plan()
        prompt = _build_step_prompt(plan.steps[0], plan, [], max_steps=25)
        assert "25 tool calls" in prompt
        assert "kill it before finishing" in prompt

    def test_budget_uses_custom_max_steps(self):
        """Budget guidance reflects the max_steps parameter."""
        plan = self._make_plan()
        prompt = _build_step_prompt(plan.steps[0], plan, [], max_steps=30)
        assert "30 tool calls" in prompt

    def test_parse_plan_file_sets_source_dir(self, tmp_path: Path):
        """parse_plan_file populates source_dir from the file's parent."""
        plan_file = tmp_path / "FIXPLAN.md"
        plan_file.write_text("## Step one\n\nBody.\n")
        plan = parse_plan_file(str(plan_file))
        assert plan.source_dir == tmp_path


# ─── _shallow_tree ───────────────────────────────────────────────────────────


class TestShallowTree:
    def test_basic_tree(self, tmp_path: Path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.cpp").write_text("")
        (tmp_path / "Makefile").write_text("")
        tree = _shallow_tree(tmp_path)
        assert "src/" in tree
        assert "main.cpp" in tree
        assert "Makefile" in tree

    def test_hidden_files_excluded(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "README.md").write_text("")
        tree = _shallow_tree(tmp_path)
        assert ".git" not in tree
        assert "README.md" in tree

    def test_depth_limit(self, tmp_path: Path):
        (tmp_path / "a" / "b" / "c").mkdir(parents=True)
        (tmp_path / "a" / "b" / "c" / "deep.txt").write_text("")
        tree = _shallow_tree(tmp_path, max_depth=2)
        assert "a/" in tree
        assert "b/" in tree
        # c/ is at depth 3, should not appear
        assert "deep.txt" not in tree


# ─── /exeplan in SLASH_COMMANDS ──────────────────────────────────────────────


# ─── AgentConfig plan_max_steps ──────────────────────────────────────────────


class TestPlanMaxSteps:
    def test_plan_max_steps_default(self):
        from natshell.config import AgentConfig

        config = AgentConfig()
        assert config.plan_max_steps == 35

    def test_plan_max_steps_independent_of_max_steps(self):
        from natshell.config import AgentConfig

        config = AgentConfig(max_steps=10, plan_max_steps=30)
        assert config.max_steps == 10
        assert config.plan_max_steps == 30


# ─── /exeplan in SLASH_COMMANDS ──────────────────────────────────────────────


class TestExeplanInSlashCommands:
    def test_exeplan_listed(self):
        from natshell.app import SLASH_COMMANDS

        commands = [cmd for cmd, _ in SLASH_COMMANDS]
        assert "/exeplan" in commands
