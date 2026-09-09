"""Plan execution helpers — pure functions with no TUI dependencies.

The prompt *templates* (plan generation, per-step, verify-fix retry) were
extracted into :mod:`natshell.agent.plan_prompts` (R1-8) and are
re-exported here so every existing import path keeps working.
"""

from __future__ import annotations

from natshell.agent.plan import Plan

# Re-exported from natshell.agent.plan_prompts (moved in R1-8) so the
# historical import paths (app.py, headless.py, tests) keep working.
from natshell.agent.plan_prompts import (  # noqa: E402
    VERIFY_FIX_BUDGET,
    _build_plan_prompt,
    _build_step_prompt,
    _build_verify_fix_prompt,
    _shallow_tree,
)
from natshell.scaling import PLAN_MAX_STEPS_TABLE, scale_for_context

__all__ = [
    "VERIFY_FIX_BUDGET",
    "_build_plan_prompt",
    "_build_step_prompt",
    "_build_verify_fix_prompt",
    "_shallow_tree",
    "_effective_plan_max_steps",
    "validate_plan",
]

_DEFAULT_PLAN_MAX_STEPS = 35


def _effective_plan_max_steps(n_ctx: int, configured: int = _DEFAULT_PLAN_MAX_STEPS) -> int:
    """Scale plan step budget based on context window size.

    Only auto-scales when *configured* is the default (35); an explicit user
    override is respected as-is.
    """
    if configured != _DEFAULT_PLAN_MAX_STEPS:
        return configured
    return scale_for_context(n_ctx, PLAN_MAX_STEPS_TABLE, 20)


def validate_plan(plan: Plan) -> list[str]:
    """Check a parsed plan for common quality issues.

    Returns a list of warning strings. Empty list = no issues.
    Does not block execution — warnings are informational.
    """
    warnings: list[str] = []

    numbers = [s.number for s in plan.steps]
    if len(numbers) != len(set(numbers)):
        dupes = [n for n in numbers if numbers.count(n) > 1]
        warnings.append(f"Duplicate step numbers: {sorted(set(dupes))}")

    no_verify = [s.number for s in plan.steps if not s.verification]
    if no_verify:
        warnings.append(
            f"Steps without **Verify:** commands: {no_verify} "
            f"(verification will be skipped for these steps)"
        )

    large = [s.number for s in plan.steps if len(s.body) > 3000]
    if large:
        warnings.append(f"Steps with very large bodies (>3000 chars): {large}")

    expected = list(range(1, len(plan.steps) + 1))
    if numbers != expected:
        warnings.append(f"Non-sequential step numbers: {numbers} (expected {expected})")

    return warnings
