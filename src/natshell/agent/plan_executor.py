"""Plan execution helpers — pure functions with no TUI dependencies."""

from __future__ import annotations

from pathlib import Path

from natshell.agent.plan import Plan, PlanStep

_DEFAULT_PLAN_MAX_STEPS = 35


def _effective_plan_max_steps(n_ctx: int, configured: int = _DEFAULT_PLAN_MAX_STEPS) -> int:
    """Scale plan step budget based on context window size.

    Only auto-scales when *configured* is the default (35); an explicit user
    override is respected as-is.
    """
    if configured != _DEFAULT_PLAN_MAX_STEPS:
        return configured
    if n_ctx >= 131072:
        return 65
    elif n_ctx >= 65536:
        return 55
    elif n_ctx >= 32768:
        return 45
    elif n_ctx >= 16384:
        return 35
    elif n_ctx >= 8192:
        return 30
    return 20


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
    step: PlanStep,
    plan: Plan,
    completed_summaries: list[str],
    max_steps: int = 25,
    *,
    completed_files: list[str] | None = None,
    n_ctx: int = 4096,
) -> str:
    """Build a focused prompt for a single plan step.

    Includes the step body, one-line summaries of completed steps,
    project directory tree, preamble context, budget guidance,
    file change tracking, and a directive to not read plan files.

    Args:
        step: The current plan step to execute.
        plan: The full parsed plan (for preamble, source_dir, total steps).
        completed_summaries: One-line summaries of previously completed steps.
        max_steps: Tool call budget for this step.
        completed_files: Paths modified/created by previous steps with annotations.
        n_ctx: Engine context window size — controls detail level.
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

    # Cross-step file change tracking
    if completed_files:
        parts.append("\nFiles created/modified by previous steps:")
        for entry in completed_files:
            parts.append(f"  {entry}")

    parts.append(f"\n## {step.title}\n")
    parts.append(step.body)
    parts.append(
        f"\nYou have {max_steps} tool calls for this step. Prioritize core implementation over "
        "validation. Do not re-read files you just wrote. Do not start long-running servers "
        "unless the step specifically requires it. If you must start a server for testing, "
        "kill it before finishing."
    )

    # Tool guidance
    parts.append(
        "\nTool usage:"
        "\n- Use write_file to create new files."
        "\n- Use edit_file for targeted changes to existing files (read first)."
        "\n- Read files before editing them to get exact content for old_text matching."
    )

    parts.append(
        "\nIMPORTANT — scope and termination rules:"
        "\n- Execute ONLY the task described above. Do not modify files not mentioned in this step."
        "\n- Do NOT re-examine, refactor, or 'improve' files you already wrote."
        "\n- Do NOT read the plan file (PLAN.md or similar)."
        "\n- When the step is complete, IMMEDIATELY provide a short text summary of what you did"
        " and STOP. Do not continue with additional tool calls after the work is done."
    )

    return "\n".join(parts)


def _build_plan_prompt(description: str, directory_tree: str, *, n_ctx: int = 4096) -> str:
    """Build a prompt that instructs the model to generate a PLAN.md file.

    Args:
        description: The user's natural-language request.
        directory_tree: A shallow directory tree of the current project.
        n_ctx: The engine's context window size, used to adapt research
               depth and step detail level.
    """
    # Determine context tier
    thorough = n_ctx >= 32768
    # Budget hint: approximate tool calls available
    if n_ctx >= 131072:
        budget = 65
    elif n_ctx >= 65536:
        budget = 55
    elif n_ctx >= 32768:
        budget = 45
    elif n_ctx >= 16384:
        budget = 35
    elif n_ctx >= 8192:
        budget = 30
    else:
        budget = 20
    research_pct = int(budget * 0.4)

    parts = [
        "Generate a multi-step plan file called PLAN.md in the current directory.",
        "",
        "User's request:",
        description,
        "",
        "Current directory:",
        directory_tree,
        "",
        # ── Research phase ────────────────────────────────────────────────
        "═══ PHASE 1: RESEARCH ═══",
        "",
        f"You have approximately {budget} tool calls. Budget ~{research_pct} for research,",
        "~10% for writing PLAN.md, and reserve the rest as buffer.",
        "",
        "Determine whether this is a new project or modifications to existing code.",
        "",
        "For EXISTING projects (directory has source files):",
        "- list_directory to see project structure",
        "- Read README, architecture docs, CONTRIBUTING, or design docs if they exist",
    ]
    if thorough:
        parts.append("- Read the 5-10 most relevant source files for the request")
    else:
        parts.append("- Read the 2-3 most relevant source files for the request")
    parts += [
        "- Read build config (package.json, pyproject.toml, Cargo.toml, etc.) to understand dependencies",
        "",
        "For NEW projects (empty or near-empty directory):",
        "- list_directory to confirm the project is new/empty",
        "- Analyze the user's request for technology choices (frameworks, languages, libraries)",
        "- Use fetch_url to look up official documentation for mentioned frameworks/libraries",
        "  (e.g., FastAPI quickstart, Express.js API reference, React project structure guides)",
        "- Use fetch_url to research best practices for the stack (recommended project structure,",
        "  common patterns, configuration approaches)",
        "- If the request mentions APIs/services to integrate, fetch their documentation",
        "",
        "For BOTH:",
        "- Read any project docs present, and use search_files/fetch_url to resolve ambiguities",
        "  in the user's request",
        "",
    ]
    if not thorough:
        parts += [
            "COMPACT MODE (small context window): conserve context for plan writing.",
            "- Research 2-3 key files maximum",
            "- Minimize fetch_url — use only for the primary framework",
            "- Keep code snippets short in step bodies",
            "- Target 3-6 steps maximum",
            "",
        ]
    else:
        parts += [
            "THOROUGH MODE (large context window): maximize research quality.",
            "- Research 5-10 source files, fetch multiple documentation URLs",
            "- Steps can span 2-4 files each, include substantial code snippets",
            "- Include detailed type signatures, import chains, error handling patterns",
            "- 3-10 steps (more granularity encouraged)",
            "- Use fetch_url aggressively for new projects",
            "",
        ]

    parts += [
        # ── Plan format ───────────────────────────────────────────────────
        "═══ PHASE 2: WRITE PLAN.MD ═══",
        "",
        "Plan format — the file MUST use this exact markdown structure:",
        "",
        "  # Plan Title",
        "  ",
        "  Preamble with shared context (see specification below).",
        "  ",
        "  ## Step 1: Title describing what this step does",
        "  ",
        "  Detailed step body (see step template below).",
        "  ",
        "  ## Step 2: Next step title",
        "  ...",
        "",
        # ── Preamble specification ────────────────────────────────────────
        "PREAMBLE SPECIFICATION — the preamble MUST include:",
        "- Language, framework, and version (discovered from research or project files)",
        "- Build / test / run commands",
        "- Project structure / directory layout to create (new project) or follow (existing)",
        "- Import style and module conventions (ESM vs CJS, absolute vs relative imports, etc.)",
        "- Key file paths that multiple steps will reference",
        "- Configuration patterns (env vars, config files, etc.)",
        "- For new projects: package dependencies to install, with versions from docs",
        "",
        "CRITICAL: The preamble is the ONLY shared context between steps. A step cannot",
        "reference information that is not in its own body or the preamble. If you learned",
        "something from fetch_url during research, encode that knowledge into the preamble",
        "or the relevant step body — do not assume the executing model knows it.",
        "",
        # ── Step body template ────────────────────────────────────────────
        "STEP BODY TEMPLATE — each step MUST include these sections:",
        "",
        "**Goal**: One sentence describing the outcome of this step.",
        "",
        "**Files**: Each file listed with its action:",
        "  CREATE `path/to/file.py` — use write_file",
        "  MODIFY `path/to/file.py` — use edit_file (read first)",
        "  READ `path/to/file.py` — read for context only",
        "",
        "**Details**: Exact function names, signatures, imports, and code snippets —",
        "explicit enough for a small LLM (4B parameters) to implement correctly.",
        "",
        "**Depends on**: Files from earlier steps that this step reads or imports.",
        "",
        "**Verify**: A command or check to confirm the step succeeded",
        "(e.g., `python -c \"import mymodule\"` or `npm test`).",
        "",
        "Example of a well-structured step:",
        "",
        "  ## Step 2: Create authentication middleware",
        "  ",
        "  **Goal**: Add JWT token validation middleware for protected routes.",
        "  ",
        "  **Files**:",
        "  CREATE `src/middleware/auth.py`",
        "  MODIFY `src/app.py`",
        "  READ `src/config.py`",
        "  ",
        "  **Details**:",
        "  In `src/middleware/auth.py`:",
        "  - Import `jwt` from PyJWT, `Request` from fastapi",
        "  - Define `async def verify_token(request: Request) -> dict`",
        "  - Read `Authorization` header, strip `Bearer ` prefix",
        "  - Decode with `jwt.decode(token, SECRET_KEY, algorithms=[\"HS256\"])`",
        "  - Return payload dict on success, raise `HTTPException(401)` on failure",
        "  ",
        "  In `src/app.py`:",
        "  - Add `from src.middleware.auth import verify_token`",
        "  - Add `Depends(verify_token)` to the `/api/protected` route",
        "  ",
        "  **Depends on**: `src/config.py` (created in step 1) for SECRET_KEY",
        "  ",
        "  **Verify**: `python -c \"from src.middleware.auth import verify_token\"`",
        "",
        # ── Greenfield scaffolding guidance ───────────────────────────────
        "GREENFIELD PROJECT GUIDANCE (if this is a new project):",
        "- Step 1 should be project scaffolding: create directory structure, package manifest",
        "  (package.json / pyproject.toml / Cargo.toml) with dependencies and versions discovered",
        "  during research, and configuration files (.env, tsconfig.json, etc.)",
        "- Include an explicit step to install dependencies (npm install, pip install -e ., etc.)",
        "  if the plan creates a package manifest",
        "- Embed specific version numbers and API patterns learned from fetch_url into step details",
        "  — do not rely on the model's training data (which may be outdated)",
        "",
        # ── Verification step guidance ────────────────────────────────────
        "FINAL STEP GUIDANCE:",
        "- The last step should be a verification/integration step that runs the build,",
        "  test suite, or linter. If no test framework exists, at minimum verify files",
        "  can be parsed/imported (e.g., `python -c \"import mypackage\"`).",
        "",
        # ── Rules ─────────────────────────────────────────────────────────
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
        "IMPORTANT restrictions during plan generation:",
        "- Do NOT run shell commands with execute_shell.",
        "- Do NOT modify existing files with edit_file.",
        "- Do NOT execute code with run_code.",
        "- ONLY use read_file, list_directory, search_files, fetch_url, and git_tool to examine the codebase.",
        "- When done, use write_file to create PLAN.md and nothing else.",
        "",
        "First examine the directory with list_directory. Read key source files",
        "and any project documentation (README, design docs, etc.) to understand",
        "the codebase before writing PLAN.md.",
    ]
    return "\n".join(parts)
