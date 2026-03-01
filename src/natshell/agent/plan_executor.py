"""Plan execution helpers — pure functions with no TUI dependencies."""

from __future__ import annotations

from pathlib import Path

from natshell.agent.plan import Plan, PlanStep


def _shallow_tree(directory: Path, max_depth: int = 2) -> str:
    """Build a compact directory tree string (2 levels deep, no hidden files)."""
    lines: list[str] = [f"{directory}/"]

    def _walk(path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        entries = [e for e in entries if not e.name.startswith(".")]
        for i, entry in enumerate(entries):
            connector = "\u2514\u2500 " if i == len(entries) - 1 else "\u251c\u2500 "
            extension = "   " if i == len(entries) - 1 else "\u2502  "
            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                _walk(entry, prefix + extension, depth + 1)
            else:
                lines.append(f"{prefix}{connector}{entry.name}")

    _walk(directory, "  ", 1)
    return "\n".join(lines)


def _build_step_prompt(
    step: PlanStep, plan: Plan, completed_summaries: list[str], max_steps: int = 25
) -> str:
    """Build a focused prompt for a single plan step.

    Includes the step body, one-line summaries of completed steps,
    project directory tree, preamble context, budget guidance,
    and a directive to not read plan files.
    """
    parts = [f"Execute this task (step {step.number} of {len(plan.steps)}):"]

    # Include project directory structure so the model doesn't waste steps discovering it
    if plan.source_dir and plan.source_dir.is_dir():
        tree = _shallow_tree(plan.source_dir)
        parts.append(f"\nProject layout:\n{tree}")

    # Include preamble context (tech stack, conventions, key interfaces)
    if plan.preamble:
        parts.append(f"\nProject context:\n{plan.preamble}")

    if completed_summaries:
        parts.append("\nPreviously completed:")
        for summary in completed_summaries:
            parts.append(f"  {summary}")

    parts.append(f"\n## {step.title}\n")
    parts.append(step.body)
    parts.append(
        f"\nYou have {max_steps} tool calls for this step. Prioritize core implementation over "
        "validation. Do not re-read files you just wrote. Do not start long-running servers "
        "unless the step specifically requires it. If you must start a server for testing, "
        "kill it before finishing."
    )
    parts.append(
        "\nExecute this step now. All instructions are above \u2014 do not read any plan files."
    )

    return "\n".join(parts)


def _build_plan_prompt(description: str, directory_tree: str) -> str:
    """Build a prompt that instructs the model to generate a PLAN.md file."""
    parts = [
        "Generate a multi-step plan file called PLAN.md in the current directory.",
        "",
        "User's request:",
        description,
        "",
        "Current directory:",
        directory_tree,
        "",
        "Plan format — the file MUST use this exact markdown structure:",
        "",
        "  # Plan Title",
        "  ",
        "  Preamble with shared context: tech stack, file structure, naming",
        "  conventions, key interfaces. This context carries across all steps.",
        "  ",
        "  ## Step 1: Title describing what this step does",
        "  ",
        "  Detailed instructions. Include exact file paths, function names,",
        "  data structures. A small LLM executes each step independently.",
        "  ",
        "  ## Step 2: Next step title",
        "  ...",
        "",
        "Rules:",
        "- First line: # heading (plan title)",
        "- Preamble BEFORE the first ## heading with shared context",
        "- Each step: ## heading (not ###)",
        "- Each step: 1-3 files, achievable in under 15 tool calls",
        "- Include exact names: functions, variables, file paths, types",
        "- Order by dependency — foundations first",
        "- 3-10 steps depending on complexity",
        "",
        "Execution quality:",
        "- Each step is executed by a small LLM with NO memory of previous steps",
        "  (only one-line summaries). Include ALL context needed in each step.",
        "- When a file depends on config from another file (e.g., import style",
        "  matching package.json \"type\"), state the dependency explicitly.",
        "- Test steps must specify: framework setup, state isolation between tests,",
        "  and the exact validation command.",
        "- Do not include \"start the server and verify in browser\" as validation.",
        "  Use test commands or scripts that start/stop automatically.",
        "",
        "Quality expectations:",
        "- READ the actual source files before writing about them. Use read_file",
        "  on key modules — do not summarize from file names alone.",
        "- Look for architecture docs (README, CONTRIBUTING, design docs, etc.)",
        "  in the project root and read them to understand documented conventions",
        "  and design decisions before proposing changes.",
        "- Be specific: reference exact file paths, function names, line ranges,",
        "  and concrete problems or patterns found.",
        "- Suggestions must be actionable and evidence-based, not generic advice",
        '  like "add more robust validation" or "improve error handling".',
        "- If reviewing code, cite the specific code that needs improvement and",
        "  explain why, not just which module it's in.",
        "",
        "First examine the directory with list_directory. Read key source files",
        "and any project documentation (README, design docs, etc.) to understand",
        "the codebase before writing PLAN.md.",
    ]
    return "\n".join(parts)
