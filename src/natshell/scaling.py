"""Context-window tier scaling — single source of truth for n_ctx → value ladders.

Each table maps a minimum context size to a scaled value, sorted descending.
``scale_for_context`` returns the value for the largest threshold that *n_ctx*
meets or exceeds, or *default* when none match.
"""

from __future__ import annotations

ScaleTable = tuple[tuple[int, int], ...]


def scale_for_context(n_ctx: int, table: ScaleTable, default: int) -> int:
    """Return the scaled value for *n_ctx* from *table*, else *default*."""
    for threshold, value in table:
        if n_ctx >= threshold:
            return value
    return default


# Agent step budget (loop.py _effective_max_steps)
MAX_STEPS_TABLE: ScaleTable = (
    (1048576, 200),
    (524288, 150),
    (262144, 120),
    (131072, 60),
    (32768, 50),
    (16384, 35),
    (8192, 25),
)

# Shell output truncation (loop.py _effective_max_output_chars)
MAX_OUTPUT_CHARS_TABLE: ScaleTable = (
    (1048576, 128000),
    (524288, 96000),
    (262144, 64000),
    (131072, 32000),
    (65536, 16000),
    (32768, 12000),
    (16384, 8000),
)

# read_file default line count (loop.py _effective_read_file_lines)
READ_FILE_LINES_TABLE: ScaleTable = (
    (1048576, 8000),
    (524288, 6000),
    (262144, 4000),
    (131072, 3000),
    (65536, 2000),
    (32768, 1000),
    (16384, 500),
)

# Working-memory character budget (working_memory.py effective_memory_chars)
MEMORY_CHARS_TABLE: ScaleTable = (
    (524288, 32000),
    (262144, 24000),
    (131072, 16000),
    (65536, 12000),
    (32768, 8000),
)

# Per-plan-step agent budget (plan_executor.py _effective_plan_max_steps)
PLAN_MAX_STEPS_TABLE: ScaleTable = (
    (1048576, 120),
    (524288, 90),
    (131072, 65),
    (65536, 55),
    (32768, 45),
    (16384, 35),
    (8192, 30),
)
