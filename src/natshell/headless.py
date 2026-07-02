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


def make_confirm_callback(auto_approve: bool):
    """Build the standard headless confirmation callback."""

    async def _confirm_callback(tool_call: Any) -> bool:
        if auto_approve:
            _err(f"[auto-approved] {tool_call.name}: {tool_call.arguments}")
            return True
        _err(f"[declined — use --danger-fast to auto-approve] {tool_call.name}")
        return False

    return _confirm_callback


async def consume_events(
    stream,
    *,
    emit=None,
    on_response=None,
    on_tool_result=None,
    on_executing=None,
    on_stats=None,
) -> bool:
    """Shared event loop for all headless runners.

    emit: line sink (defaults to _err; exeplan passes its step logger).
    on_response: handler for the final RESPONSE text (required).
    on_tool_result: optional extra hook, called after standard output logging.
    on_executing: optional replacement for the default "[executing] name" line.
    Returns True if any ERROR event was seen.
    """
    if emit is None:
        emit = _err
    had_error = False
    async for event in stream:
        match event.type:
            case EventType.RESPONSE:
                on_response(event.data)
            case EventType.TOOL_RESULT:
                if event.tool_result and event.tool_result.output:
                    emit(event.tool_result.output)
                if event.tool_result and event.tool_result.error:
                    emit(f"[stderr] {event.tool_result.error}")
                if on_tool_result:
                    on_tool_result(event)
            case EventType.PLANNING:
                emit(f"[thinking] {event.data}")
            case EventType.EXECUTING:
                if event.tool_call:
                    if on_executing:
                        on_executing(event)
                    else:
                        emit(f"[executing] {event.tool_call.name}")
            case EventType.BLOCKED:
                if event.tool_call:
                    emit(f"[BLOCKED] {event.tool_call.name}: {event.tool_call.arguments}")
            case EventType.ERROR:
                emit(f"[error] {event.data}")
                had_error = True
            case EventType.RUN_STATS:
                if event.metrics:
                    steps = event.metrics.get("steps", "?")
                    wall = event.metrics.get("total_wall_ms", 0)
                    emit(f"[stats] {steps} steps in {wall}ms")
                    if on_stats:
                        on_stats(event.metrics)
    return had_error


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

    had_error = await consume_events(
        agent.handle_user_message(
            prompt,
            confirm_callback=make_confirm_callback(auto_approve),
        ),
        on_response=lambda data: print(data, flush=True),
    )

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

    agent.clear_history()
    tree = _shallow_tree(Path.cwd())
    try:
        n_ctx = agent.engine.engine_info().n_ctx or 4096
    except (AttributeError, TypeError):
        n_ctx = 4096
    prompt = _build_plan_prompt(description, tree, n_ctx=n_ctx)

    had_error = False
    try:
        had_error = await consume_events(
            agent.handle_user_message(
                prompt,
                confirm_callback=make_confirm_callback(auto_approve),
                tool_filter=PLAN_SAFE_TOOLS,
                skip_intent_detection=True,
            ),
            on_response=lambda data: _err(f"[response] {data}"),
        )
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
    resume: bool = False,
) -> int:
    """Execute a plan file step by step.

    Mirrors app.py:run_plan but outputs to stderr/stdout.
    Supports post-step verification, one-retry fix, state persistence,
    and resume from the last failed step.

    Returns:
        Exit code (0 = all steps passed, 1 = any failures).
    """
    from datetime import datetime, timezone

    from natshell.agent.plan import parse_plan_file
    from natshell.agent.plan_executor import (
        _build_step_prompt,
        _effective_plan_max_steps,
        validate_plan,
    )
    from natshell.agent.plan_state import (
        PlanState,
        StepResult,
        get_resume_point,
        load_plan_state,
        save_plan_state,
        state_path_for_plan,
    )

    try:
        plan = parse_plan_file(plan_path)
    except FileNotFoundError as e:
        _err(f"[error] {e}")
        return 1
    except ValueError as e:
        _err(f"[error] Parse error: {e}")
        return 1

    confirm = make_confirm_callback(auto_approve)

    _err(f"Executing plan: {plan.title} ({len(plan.steps)} steps)")

    plan_warnings = validate_plan(plan)
    for w in plan_warnings:
        _err(f"[plan-warning] {w}")

    plan_t0 = time.monotonic()

    try:
        n_ctx = agent.engine.engine_info().n_ctx or 4096
    except (AttributeError, TypeError):
        n_ctx = 4096

    # --- Resume or fresh start ---
    state_file = state_path_for_plan(Path(plan_path).resolve())
    completed_summaries: list[str] = []
    completed_files: list[str] = []
    completed_count = 0
    failed_count = 0
    resume_idx = 0
    prior_results: list[StepResult] = []

    if resume and state_file.exists():
        try:
            prev_state = load_plan_state(state_file)
            resume_idx = get_resume_point(prev_state)
            completed_summaries = list(prev_state.completed_summaries)
            completed_files = list(prev_state.completed_files)
            completed_count = sum(
                1 for r in prev_state.step_results if r.status == "passed"
            )
            failed_count = sum(
                1 for r in prev_state.step_results
                if r.status in ("failed", "partial")
            )
            prior_results = list(prev_state.step_results[:resume_idx])
            _err(
                f"Resuming from step {resume_idx + 1} "
                f"({completed_count} passed, {failed_count} failed)"
            )
            # Reset failed count — we're retrying from the failure point
            failed_count = 0
        except (ValueError, KeyError) as e:
            _err(f"[warn] Could not load state file, starting fresh: {e}")
            resume_idx = 0

    state = PlanState(
        plan_file=plan_path,
        plan_title=plan.title,
        total_steps=len(plan.steps),
        step_results=list(prior_results),
        completed_summaries=list(completed_summaries),
        completed_files=list(completed_files),
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    for step in plan.steps:
        # Skip already-completed steps on resume
        if step.number - 1 < resume_idx:
            _err(f"[skip] Step {step.number}: {step.title} (already completed)")
            continue

        _err(f"\n{'=' * 60}")
        _err(f"Step {step.number}/{len(plan.steps)}: {step.title}")
        _err(f"{'=' * 60}")

        agent.clear_history()

        # plan_max is a FLOOR — large-context models get their
        # full context-scaled budget when it exceeds the plan floor.
        plan_max = _effective_plan_max_steps(n_ctx, agent.config.plan_max_steps)
        original_max = agent.config.max_steps
        effective_max = max(plan_max, agent._effective_max_steps(n_ctx))
        agent.config.max_steps = effective_max
        agent.set_step_limit(effective_max)

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
        step_metrics: dict[str, Any] = {}
        step_log: list[str] = []
        step_tool_count = 0
        step_t0 = time.monotonic()
        verify_attempts = 0

        def _log(msg: str) -> None:
            step_log.append(msg)
            _err(msg)

        try:
            def _on_response(data: str) -> None:
                nonlocal hit_max_steps
                _log(f"[response] {data}")
                if data and "maximum number of steps" in data:
                    hit_max_steps = True

            def _on_tool_result(event: Any) -> None:
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
                        step_files.append(f"{path} ({action} in step {step.number})")

            def _on_executing(event: Any) -> None:
                nonlocal step_tool_count
                step_tool_count += 1
                elapsed = int(time.monotonic() - step_t0)
                _log(
                    f"[executing] {event.tool_call.name} "
                    f"({step_tool_count}/{effective_max} calls, {elapsed}s)"
                )

            def _on_stats(metrics: dict[str, Any]) -> None:
                nonlocal step_metrics
                step_metrics = metrics

            await consume_events(
                agent.handle_user_message(prompt, confirm_callback=confirm),
                emit=_log,
                on_response=_on_response,
                on_tool_result=_on_tool_result,
                on_executing=_on_executing,
                on_stats=_on_stats,
            )

        except Exception as e:
            _log(f"[error] Step {step.number} failed: {e}")
            failed_count += 1
            completed_summaries.append(f"{step.number}. {step.title} ✗")
            state.step_results.append(StepResult(
                number=step.number, title=step.title, status="failed",
                summary=completed_summaries[-1], files_changed=step_files,
                error=str(e),
                wall_ms=step_metrics.get("total_wall_ms", 0),
                inference_ms=step_metrics.get("total_inference_ms", 0),
                prompt_tokens=step_metrics.get("total_prompt_tokens", 0),
                completion_tokens=step_metrics.get("total_completion_tokens", 0),
                tool_calls=step_metrics.get("steps", 0),
                verify_attempts=verify_attempts,
                log_lines=step_log,
            ))
            state.completed_summaries = list(completed_summaries)
            state.completed_files = list(completed_files)
            save_plan_state(state, state_file)
            agent.config.max_steps = original_max
            agent.set_step_limit(agent._effective_max_steps(n_ctx))
            continue

        finally:
            agent.config.max_steps = original_max
            agent.set_step_limit(agent._effective_max_steps(n_ctx))

        completed_files.extend(step_files)

        # --- Update status ---
        if hit_max_steps:
            failed_count += 1
            completed_summaries.append(f"{step.number}. {step.title} ⚠ (partial)")
            _err(f"[partial] Step {step.number} hit max steps")
            status = "partial"
        else:
            completed_count += 1
            completed_summaries.append(f"{step.number}. {step.title} ✓")
            _err(f"[done] Step {step.number} complete")
            status = "passed"

        # --- Persist state ---
        state.step_results.append(StepResult(
            number=step.number, title=step.title, status=status,
            summary=completed_summaries[-1], files_changed=step_files,
            wall_ms=step_metrics.get("total_wall_ms", 0),
            inference_ms=step_metrics.get("total_inference_ms", 0),
            prompt_tokens=step_metrics.get("total_prompt_tokens", 0),
            completion_tokens=step_metrics.get("total_completion_tokens", 0),
            tool_calls=step_metrics.get("steps", 0),
            verify_attempts=verify_attempts,
            log_lines=step_log,
        ))
        state.completed_summaries = list(completed_summaries)
        state.completed_files = list(completed_files)
        save_plan_state(state, state_file)

    skipped_count = len(plan.steps) - completed_count - failed_count
    wall_ms = int((time.monotonic() - plan_t0) * 1000)

    # Compute aggregates
    state.total_wall_ms = sum(r.wall_ms for r in state.step_results)
    state.total_inference_ms = sum(r.inference_ms for r in state.step_results)
    state.total_prompt_tokens = sum(r.prompt_tokens for r in state.step_results)
    state.total_completion_tokens = sum(r.completion_tokens for r in state.step_results)
    state.total_tool_calls = sum(r.tool_calls for r in state.step_results)

    # Final state save
    state.finished_at = datetime.now(timezone.utc).isoformat()
    save_plan_state(state, state_file)

    # Print summary to stdout
    print(f"\nPlan complete: {plan.title}", flush=True)
    print(f"  Passed: {completed_count}", flush=True)
    print(f"  Failed: {failed_count}", flush=True)
    if skipped_count:
        print(f"  Skipped: {skipped_count}", flush=True)
    print(f"  Time: {wall_ms}ms", flush=True)
    total_tokens = state.total_prompt_tokens + state.total_completion_tokens
    if total_tokens:
        print(
            f"  Tokens: {total_tokens:,} "
            f"({state.total_prompt_tokens:,}p + {state.total_completion_tokens:,}c)",
            flush=True,
        )
    if state.total_wall_ms:
        mins = state.total_wall_ms / 60_000
        print(f"  Inference time: {mins:.1f} min", flush=True)

    return 1 if failed_count > 0 else 0


def _err(msg: str) -> None:
    """Print a message to stderr."""
    print(msg, file=sys.stderr, flush=True)
