"""Centralised runtime limits that scale with the context window."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolLimits:
    """Mutable limits container shared across all tools.

    Default values match the smallest context-window tier (<=4K).
    The agent loop overwrites these once the engine's *n_ctx* is known.
    """

    max_output_chars: int = 4000
    read_file_lines: int = 200
