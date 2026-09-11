"""Agent event types — shared vocabulary of the ReAct loop and the TUI.

Extracted from :mod:`natshell.agent.loop` so helper modules (e.g.
:mod:`natshell.agent.sudo_retry`) and the UI can consume events without
importing the runtime.  ``natshell.agent.loop`` re-exports both names so
existing imports keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from natshell.inference.engine import ToolCall
from natshell.tools.registry import ToolResult


class EventType(Enum):
    THINKING = "thinking"
    PLANNING = "planning"  # Model's text before tool calls
    EXECUTING = "executing"  # About to run a tool
    TOOL_OUTPUT = "tool_output"  # Live output chunk from a streaming tool (R2-4)
    TOOL_RESULT = "tool_result"  # Result from a tool
    CONFIRM_NEEDED = "confirm_needed"  # Awaiting user confirmation
    BLOCKED = "blocked"  # Command was blocked
    RESPONSE = "response"  # Final text response from model
    ERROR = "error"  # Something went wrong
    RUN_STATS = "run_stats"  # Cumulative stats for the full agent run
    QUEUED_MESSAGE = "queued_message"  # User message injected mid-run
    PLAN_STEP = "plan_step"  # Plan step divider (start/update)
    PLAN_COMPLETE = "plan_complete"  # Entire plan finished


@dataclass
class AgentEvent:
    """An event yielded by the agent loop for the TUI to render."""

    type: EventType
    data: Any = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    metrics: dict[str, Any] | None = None
