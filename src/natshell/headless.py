"""Headless (non-interactive) mode — single-shot CLI execution.

Usage: natshell --headless "scan my network" | grep host
       natshell --plan "build a REST API"
       natshell --exeplan PLAN.md --danger-fast

RESPONSE text goes to stdout (pipeable), everything else to stderr.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from natshell.agent.loop import AgentLoop, EventType


async def run_headless(
    agent: AgentLoop,
    prompt: str,
    auto_approve: bool = False,
) -> int:
    """Run a single prompt through the agent, printing results to stdout/stderr.

    Args:
        agent: Initialized AgentLoop instance.
        prompt: The user's natural language request.
        auto_approve: If True, auto-approve all confirmation prompts.
                      If False, auto-decline (safer default).

    Returns:
        Exit code (0 = success, 1 = error).
    """

    async def _confirm_callback(tool_call: Any) -> bool:
        if auto_approve:
            _err(f"[auto-approved] {tool_call.name}: {tool_call.arguments}")
            return True
        _err(f"[declined — use --danger-fast to auto-approve] {tool_call.name}")
        return False

    had_error = False

    async for event in agent.handle_user_message(
        prompt,
        confirm_callback=_confirm_callback,
    ):
        match event.type:
            case EventType.RESPONSE:
                # Final text → stdout (pipeable)
                print(event.data, flush=True)

            case EventType.TOOL_RESULT:
                # Tool output → stderr for debugging
                if event.tool_result and event.tool_result.output:
                    _err(event.tool_result.output)
                if event.tool_result and event.tool_result.error:
                    _err(f"[stderr] {event.tool_result.error}")

            case EventType.PLANNING:
                _err(f"[thinking] {event.data}")

            case EventType.EXECUTING:
                if event.tool_call:
                    _err(f"[executing] {event.tool_call.name}")

            case EventType.BLOCKED:
                if event.tool_call:
                    _err(f"[BLOCKED] {event.tool_call.name}: {event.tool_call.arguments}")

            case EventType.ERROR:
                _err(f"[error] {event.data}")
                had_error = True

            case EventType.CONFIRM_NEEDED:
                pass  # Handled by _confirm_callback

            case EventType.QUEUED_MESSAGE:
                pass  # Headless is single-shot; queuing doesn't apply

            case EventType.RUN_STATS:
                if event.metrics:
                    steps = event.metrics.get("steps", "?")
                    wall = event.metrics.get("total_wall_ms", 0)
                    _err(f"[stats] {steps} steps in {wall}ms")

    if had_error:
        return 1
    return 0


async def run_headless_plan(
    agent: AgentLoop,
    description: str,
    auto_approve: bool = False,
) -> int:
    """Generate a PLAN.md from a natural language description.

    Mirrors app.py:run_plan_generation but outputs to stderr/stdout.

    Returns:
        Exit code (0 = PLAN.md created, 1 = error).
    """
    from natshell.agent.plan import parse_plan_file
    from natshell.agent.plan_executor import _build_plan_prompt, _shallow_tree
    from natshell.tools.registry import PLAN_SAFE_TOOLS

    async def _confirm_callback(tool_call: Any) -> bool:
        if auto_approve:
            _err(f"[auto-approved] {tool_call.name}: {tool_call.arguments}")
            return True
        _err(f"[declined — use --danger-fast to auto-approve] {tool_call.name}")
        return False

    agent.clear_history()
    tree = _shallow_tree(Path.cwd())
    try:
        n_ctx = agent.engine.engine_info().n_ctx or 4096
    except (AttributeError, TypeError):
        n_ctx = 4096
    prompt = _build_plan_prompt(description, tree, n_ctx=n_ctx)

    had_error = False
    try:
        async for event in agent.handle_user_message(
            prompt,
            confirm_callback=_confirm_callback,
            tool_filter=PLAN_SAFE_TOOLS,
            skip_intent_detection=True,
        ):
            match event.type:
                case EventType.RESPONSE:
                    _err(f"[response] {event.data}")
                case EventType.TOOL_RESULT:
                    if event.tool_result and event.tool_result.output:
                        _err(event.tool_result.output)
                    if event.tool_result and event.tool_result.error:
                        _err(f"[stderr] {event.tool_result.error}")
                case EventType.PLANNING:
                    _err(f"[thinking] {event.data}")
                case EventType.EXECUTING:
                    if event.tool_call:
                        _err(f"[executing] {event.tool_call.name}")
                case EventType.ERROR:
                    _err(f"[error] {event.data}")
                    had_error = True
                case EventType.RUN_STATS:
                    if event.metrics:
                        steps = event.metrics.get("steps", "?")
                        wall = event.metrics.get("total_wall_ms", 0)
                        _err(f"[stats] {steps} steps in {wall}ms")
    except Exception as e:
        _err(f"[error] Plan generation failed: {e}")
        return 1

    if had_error:
        return 1

    # Check if PLAN.md was created
    plan_path = Path.cwd() / "PLAN.md"
    if plan_path.exists():
        try:
            plan = parse_plan_file(str(plan_path))
            print(f"Plan written: {plan.title} ({len(plan.steps)} steps)", flush=True)
            for step in plan.steps:
                print(f"  {step.number}. {step.title}", flush=True)
        except (ValueError, FileNotFoundError):
            print(f"Plan written to {plan_path}", flush=True)
        return 0

    _err("[error] PLAN.md was not created")
    return 1


async def run_headless_exeplan(
    agent: AgentLoop,
    plan_path: str,
    auto_approve: bool = False,
) -> int:
    """Execute a plan file step by step.

    Mirrors app.py:run_plan but outputs to stderr/stdout.

    Returns:
        Exit code (0 = all steps passed, 1 = any failures).
    """
    from natshell.agent.plan import parse_plan_file
    from natshell.agent.plan_executor import (
        _build_step_prompt,
        _effective_plan_max_steps,
    )

    try:
        plan = parse_plan_file(plan_path)
    except FileNotFoundError as e:
        _err(f"[error] {e}")
        return 1
    except ValueError as e:
        _err(f"[error] Parse error: {e}")
        return 1

    async def _confirm_callback(tool_call: Any) -> bool:
        if auto_approve:
            _err(f"[auto-approved] {tool_call.name}: {tool_call.arguments}")
            return True
        _err(f"[declined — use --danger-fast to auto-approve] {tool_call.name}")
        return False

    _err(f"Executing plan: {plan.title} ({len(plan.steps)} steps)")
    plan_t0 = time.monotonic()

    try:
        n_ctx = agent.engine.engine_info().n_ctx or 4096
    except (AttributeError, TypeError):
        n_ctx = 4096

    completed_summaries: list[str] = []
    completed_files: list[str] = []
    completed_count = 0
    failed_count = 0

    for step in plan.steps:
        _err(f"\n{'=' * 60}")
        _err(f"Step {step.number}/{len(plan.steps)}: {step.title}")
        _err(f"{'=' * 60}")

        agent.clear_history()

        plan_max = _effective_plan_max_steps(n_ctx, agent.config.plan_max_steps)
        original_max = agent.config.max_steps
        agent.config.max_steps = plan_max
        effective_max = max(getattr(agent, "_max_steps", plan_max), plan_max)

        prompt = _build_step_prompt(
            step,
            plan,
            completed_summaries,
            max_steps=effective_max,
            completed_files=completed_files or None,
            n_ctx=n_ctx,
        )

        hit_max_steps = False
        step_files: list[str] = []

        try:
            async for event in agent.handle_user_message(
                prompt,
                confirm_callback=_confirm_callback,
            ):
                match event.type:
                    case EventType.RESPONSE:
                        _err(f"[response] {event.data}")
                        if event.data and "maximum number of steps" in event.data:
                            hit_max_steps = True
                    case EventType.TOOL_RESULT:
                        if event.tool_result and event.tool_result.output:
                            _err(event.tool_result.output)
                        if event.tool_result and event.tool_result.error:
                            _err(f"[stderr] {event.tool_result.error}")
                        # Track file changes
                        if (
                            event.tool_call
                            and event.tool_call.name in ("write_file", "edit_file")
                            and event.tool_result
                            and event.tool_result.exit_code == 0
                        ):
                            path = event.tool_call.arguments.get("path", "")
                            if path:
                                action = (
                                    "created"
                                    if event.tool_call.name == "write_file"
                                    else "modified"
                                )
                                step_files.append(
                                    f"{path} ({action} in step {step.number})"
                                )
                    case EventType.PLANNING:
                        _err(f"[thinking] {event.data}")
                    case EventType.EXECUTING:
                        if event.tool_call:
                            _err(f"[executing] {event.tool_call.name}")
                    case EventType.BLOCKED:
                        if event.tool_call:
                            _err(
                                f"[BLOCKED] {event.tool_call.name}: "
                                f"{event.tool_call.arguments}"
                            )
                    case EventType.ERROR:
                        _err(f"[error] {event.data}")
                    case EventType.RUN_STATS:
                        if event.metrics:
                            steps = event.metrics.get("steps", "?")
                            wall = event.metrics.get("total_wall_ms", 0)
                            _err(f"[stats] {steps} steps in {wall}ms")

        except Exception as e:
            _err(f"[error] Step {step.number} failed: {e}")
            failed_count += 1
            completed_summaries.append(f"{step.number}. {step.title} ✗")
            agent.config.max_steps = original_max
            continue

        finally:
            agent.config.max_steps = original_max

        completed_files.extend(step_files)

        if hit_max_steps:
            failed_count += 1
            completed_summaries.append(f"{step.number}. {step.title} ⚠ (partial)")
            _err(f"[partial] Step {step.number} hit max steps")
        else:
            completed_count += 1
            completed_summaries.append(f"{step.number}. {step.title} ✓")
            _err(f"[done] Step {step.number} complete")

    skipped_count = len(plan.steps) - completed_count - failed_count
    wall_ms = int((time.monotonic() - plan_t0) * 1000)

    # Print summary to stdout
    print(f"\nPlan complete: {plan.title}", flush=True)
    print(f"  Passed: {completed_count}", flush=True)
    print(f"  Failed: {failed_count}", flush=True)
    if skipped_count:
        print(f"  Skipped: {skipped_count}", flush=True)
    print(f"  Time: {wall_ms}ms", flush=True)

    return 1 if failed_count > 0 else 0


def _err(msg: str) -> None:
    """Print a message to stderr."""
    print(msg, file=sys.stderr, flush=True)
