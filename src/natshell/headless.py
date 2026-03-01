"""Headless (non-interactive) mode — single-shot CLI execution.

Usage: natshell --headless "scan my network" | grep host

RESPONSE text goes to stdout (pipeable), everything else to stderr.
"""

from __future__ import annotations

import sys
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

    had_response = False
    had_error = False

    async for event in agent.handle_user_message(
        prompt,
        confirm_callback=_confirm_callback,
    ):
        match event.type:
            case EventType.RESPONSE:
                # Final text → stdout (pipeable)
                print(event.data, flush=True)
                had_response = True

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

            case EventType.RUN_STATS:
                if event.metrics:
                    steps = event.metrics.get("steps", "?")
                    wall = event.metrics.get("total_wall_ms", 0)
                    _err(f"[stats] {steps} steps in {wall}ms")

    if had_error and not had_response:
        return 1
    return 0


def _err(msg: str) -> None:
    """Print a message to stderr."""
    print(msg, file=sys.stderr, flush=True)
