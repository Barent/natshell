"""Tests for plan execution prompt building and /exeplan integration."""

from __future__ import annotations

import textwrap
from pathlib import Path

from natshell.agent.plan import Plan, parse_plan_file, parse_plan_text
from natshell.agent.plan_executor import _build_plan_prompt, _build_step_prompt, _shallow_tree

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
        assert "STOP making tool calls" in prompt
        assert "SUMMARY:" in prompt
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


# ─── _build_plan_prompt ──────────────────────────────────────────────────────


class TestBuildPlanPrompt:
    def test_contains_prohibition_text(self):
        """Plan prompt should contain tool restriction text."""
        prompt = _build_plan_prompt("build a REST API", "project/\n  src/\n  package.json")
        assert "Do NOT run shell commands with execute_shell" in prompt
        assert "Do NOT modify existing files with edit_file" in prompt
        assert "Do NOT execute code with run_code" in prompt
        assert "For research use read_file, list_directory, search_files" in prompt
        assert "write_file to create PLAN.md" in prompt

    def test_contains_user_description(self):
        prompt = _build_plan_prompt("build a REST API", "project/")
        assert "build a REST API" in prompt

    def test_contains_directory_tree(self):
        tree = "project/\n  src/\n  Makefile"
        prompt = _build_plan_prompt("test", tree)
        assert "Makefile" in prompt


# ─── /exeplan in SLASH_COMMANDS ──────────────────────────────────────────────


# ─── _effective_plan_max_steps ────────────────────────────────────────────────


class TestEffectivePlanMaxSteps:
    """Test context-aware plan step scaling."""

    def test_scaling_table(self):
        from natshell.agent.plan_executor import _effective_plan_max_steps

        assert _effective_plan_max_steps(4096) == 20
        assert _effective_plan_max_steps(8192) == 30
        assert _effective_plan_max_steps(16384) == 35
        assert _effective_plan_max_steps(32768) == 45
        assert _effective_plan_max_steps(65536) == 55
        assert _effective_plan_max_steps(131072) == 65
        assert _effective_plan_max_steps(262144) == 65

    def test_explicit_override_respected(self):
        from natshell.agent.plan_executor import _effective_plan_max_steps

        # Non-default value should be returned as-is regardless of n_ctx
        assert _effective_plan_max_steps(262144, configured=50) == 50
        assert _effective_plan_max_steps(4096, configured=50) == 50

    def test_default_value_triggers_scaling(self):
        from natshell.agent.plan_executor import _effective_plan_max_steps

        # Default (35) should be auto-scaled
        assert _effective_plan_max_steps(4096, configured=35) == 20
        assert _effective_plan_max_steps(131072, configured=35) == 65


# ─── Enhanced _build_plan_prompt ─────────────────────────────────────────────


class TestBuildPlanPromptEnhanced:
    """Test the enhanced plan generation prompt."""

    def test_research_phase_present(self):
        prompt = _build_plan_prompt("build an API", "project/")
        assert "PHASE 1: RESEARCH" in prompt

    def test_preamble_spec_present(self):
        prompt = _build_plan_prompt("build an API", "project/")
        assert "PREAMBLE SPECIFICATION" in prompt

    def test_step_template_present(self):
        prompt = _build_plan_prompt("build an API", "project/")
        assert "STEP BODY TEMPLATE" in prompt
        assert "**Goal**" in prompt
        assert "**Files**" in prompt
        assert "**Details**" in prompt
        assert "**Depends on**" in prompt
        assert "**Verify**" in prompt

    def test_verification_guidance_present(self):
        prompt = _build_plan_prompt("build an API", "project/")
        assert "FINAL STEP GUIDANCE" in prompt

    def test_greenfield_guidance_present(self):
        prompt = _build_plan_prompt("build an API", "project/")
        assert "GREENFIELD PROJECT GUIDANCE" in prompt

    def test_fetch_url_mentioned(self):
        prompt = _build_plan_prompt("build an API", "project/")
        assert "fetch_url" in prompt

    def test_compact_mode_for_small_context(self):
        prompt = _build_plan_prompt("build an API", "project/", n_ctx=4096)
        assert "COMPACT MODE" in prompt
        assert "THOROUGH MODE" not in prompt

    def test_thorough_mode_for_large_context(self):
        prompt = _build_plan_prompt("build an API", "project/", n_ctx=262144)
        assert "THOROUGH MODE" in prompt
        assert "COMPACT MODE" not in prompt

    def test_budget_hint_present(self):
        prompt = _build_plan_prompt("build an API", "project/", n_ctx=32768)
        assert "approximately" in prompt
        assert "tool calls" in prompt

    def test_example_step_present(self):
        """The prompt includes a concrete example of a well-structured step."""
        prompt = _build_plan_prompt("build an API", "project/")
        assert "authentication middleware" in prompt


# ─── Enhanced _build_step_prompt ─────────────────────────────────────────────


class TestBuildStepPromptEnhanced:
    """Test the enhanced step execution prompt with file tracking."""

    def _make_plan(self) -> Plan:
        return parse_plan_text(
            textwrap.dedent("""\
            # Test Plan

            ## Create auth module

            CREATE `src/auth.py`

            ## Wire up routes

            MODIFY `src/app.py`
        """)
        )

    def test_completed_files_appear_when_nonempty(self):
        plan = self._make_plan()
        files = ["src/auth.py (created in step 1)"]
        prompt = _build_step_prompt(
            plan.steps[1], plan, [], completed_files=files
        )
        assert "Files created/modified by previous steps:" in prompt
        assert "src/auth.py (created in step 1)" in prompt

    def test_completed_files_absent_when_empty(self):
        plan = self._make_plan()
        prompt = _build_step_prompt(
            plan.steps[0], plan, [], completed_files=[]
        )
        assert "Files created/modified by previous steps:" not in prompt

    def test_completed_files_absent_when_none(self):
        plan = self._make_plan()
        prompt = _build_step_prompt(
            plan.steps[0], plan, [], completed_files=None
        )
        assert "Files created/modified by previous steps:" not in prompt

    def test_backward_compatible_without_completed_files(self):
        """Calling without completed_files kwarg still works."""
        plan = self._make_plan()
        prompt = _build_step_prompt(plan.steps[0], plan, [])
        assert "step 1 of 2" in prompt
        assert "Files created/modified" not in prompt

    def test_tool_guidance_present(self):
        plan = self._make_plan()
        prompt = _build_step_prompt(plan.steps[0], plan, [])
        assert "write_file" in prompt
        assert "edit_file" in prompt
        assert "Tool usage:" in prompt


class TestExeplanInSlashCommands:
    def test_exeplan_listed(self):
        from natshell.app import SLASH_COMMANDS

        commands = [cmd for cmd, _ in SLASH_COMMANDS]
        assert "/exeplan" in commands


# ─── _annotate_file ───────────────────────────────────────────────────────────


class TestAnnotateFile:
    """Test the file annotation helper for cross-step context injection."""

    def test_annotate_tsx_exports_and_classes(self):
        from natshell.agent.plan_executor import _annotate_file

        content = '''
export function HUD() {
  return <div className="text-neon-cyan bg-neon-pink">...</div>
}
export const Score = () => <span className="neon-yellow">0</span>
'''
        result = _annotate_file("src/HUD.tsx", content)
        assert "exports: HUD, Score" in result
        # Generic class extraction — any hyphenated token > 3 chars
        assert "text-neon-cyan" in result
        assert "bg-neon-pink" in result
        assert "neon-yellow" in result

    def test_annotate_non_source_returns_empty(self):
        from natshell.agent.plan_executor import _annotate_file

        result = _annotate_file("README.md", "# Hello")
        assert result == ""

    def test_annotate_skips_short_utility_classes(self):
        from natshell.agent.plan_executor import _annotate_file

        # Short tokens without hyphens like "flex", "p-4" are excluded
        content = (
            'export function Foo() {'
            ' return <div className="flex p-4 btn-primary">...</div> }'
        )
        result = _annotate_file("Foo.tsx", content)
        assert "exports: Foo" in result
        # "flex" has no hyphen, "p-4" is <=3 chars — both excluded
        assert "flex" not in result
        # "btn-primary" has a hyphen and is >3 chars — included
        assert "btn-primary" in result

    def test_annotate_js_file(self):
        from natshell.agent.plan_executor import _annotate_file

        content = "export const helper = () => {};\nexport function util() {}"
        result = _annotate_file("utils.js", content)
        assert "exports: helper, util" in result

    def test_annotate_non_js_extension_returns_empty(self):
        from natshell.agent.plan_executor import _annotate_file

        content = "export function Foo() {}"
        assert _annotate_file("module.py", content) == ""
        assert _annotate_file("style.css", content) == ""

    def test_annotate_deduplicates_classes(self):
        from natshell.agent.plan_executor import _annotate_file

        content = (
            '<div className="btn-pink btn-cyan">'
            '<span className="btn-pink btn-yellow"></span>'
            "</div>"
        )
        result = _annotate_file("Card.tsx", content)
        # btn-pink should appear only once
        assert result.count("btn-pink") == 1

    def test_annotate_limits_exports_to_five(self):
        from natshell.agent.plan_executor import _annotate_file

        exports = "\n".join(f"export function Fn{i}() {{}}" for i in range(10))
        result = _annotate_file("big.tsx", exports)
        # Should only list up to 5 exports
        listed = result.split("exports: ")[1].split(";")[0].split(", ")
        assert len(listed) <= 5

    def test_annotate_limits_classes_to_ten(self):
        from natshell.agent.plan_executor import _annotate_file

        classes = " ".join(f"cls-token-{i}" for i in range(20))
        content = f'<div className="{classes}"></div>'
        result = _annotate_file("many.tsx", content)
        listed = result.split("classes: ")[1].split(", ")
        assert len(listed) <= 10


# ─── _build_step_prompt summary injection ─────────────────────────────────────


class TestBuildStepPromptSummaryDirective:
    """SUMMARY: instruction is appended to step prompts."""

    def _make_plan(self) -> "Plan":
        return parse_plan_text(
            textwrap.dedent("""\
            # My Plan

            ## Do something

            Create a file.
        """)
        )

    def test_summary_directive_present(self):
        plan = self._make_plan()
        prompt = _build_step_prompt(plan.steps[0], plan, [])
        assert "SUMMARY:" in prompt
        assert "one sentence" in prompt

    def test_summary_directive_format_line(self):
        plan = self._make_plan()
        prompt = _build_step_prompt(plan.steps[0], plan, [])
        assert "must be the last line" in prompt


# ─── _build_step_prompt n_ctx summary gating ──────────────────────────────────


class TestBuildStepPromptSummaryGating:
    """Expanded summaries are gated on context window size."""

    def _make_plan(self) -> "Plan":
        return parse_plan_text(
            textwrap.dedent("""\
            # Plan

            ## Step A

            Do A.

            ## Step B

            Do B.
        """)
        )

    def test_expanded_summary_shown_at_large_context(self):
        plan = self._make_plan()
        summaries = ["1. Step A \u2713 \u2014 Created auth module with JWT tokens"]
        prompt = _build_step_prompt(
            plan.steps[1], plan, summaries, n_ctx=32768
        )
        assert "Created auth module with JWT tokens" in prompt

    def test_expanded_summary_truncated_at_small_context(self):
        plan = self._make_plan()
        summaries = ["1. Step A \u2713 \u2014 Created auth module with JWT tokens"]
        prompt = _build_step_prompt(
            plan.steps[1], plan, summaries, n_ctx=8192
        )
        assert "Created auth module with JWT tokens" not in prompt
        assert "Step A" in prompt

    def test_plain_summary_unaffected_at_small_context(self):
        plan = self._make_plan()
        summaries = ["1. Step A \u2713"]
        prompt = _build_step_prompt(
            plan.steps[1], plan, summaries, n_ctx=8192
        )
        assert "Step A" in prompt

    def test_emdash_in_title_preserved_at_small_context(self):
        """An em-dash within the step *title* must not be split off."""
        plan = self._make_plan()
        # Title contains an em-dash, no appended summary detail
        summaries = ["1. Setup \u2014 auth \u2713"]
        prompt = _build_step_prompt(
            plan.steps[1], plan, summaries, n_ctx=8192
        )
        # The whole title should survive — no detail to strip
        assert "Setup \u2014 auth \u2713" in prompt

    def test_emdash_in_title_with_summary_strips_detail_only(self):
        """When title has em-dash AND there is an appended summary,
        only the detail after the status marker's em-dash is removed;
        the em-dash inside the title survives."""
        plan = self._make_plan()
        summaries = [
            "1. Setup \u2014 auth \u2713 \u2014 Created JWT middleware"
        ]
        prompt = _build_step_prompt(
            plan.steps[1], plan, summaries, n_ctx=8192
        )
        # Title including its em-dash survives
        assert "Setup \u2014 auth" in prompt
        # Summary detail is stripped
        assert "Created JWT middleware" not in prompt
