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


# ──────────────────────────────────────────────────────────────────────────────
# R2-6 feedback half — run-history driven autotuning (pure policy, no I/O).
# Consumers supply *records* (from
# :meth:`natshell.agent.run_metrics.RunMetricsStore.load_recent`); we only
# decide whether ``max_tokens`` should grow for the next run.
# ──────────────────────────────────────────────────────────────────────────────


def advise_max_tokens(
    current: int | None,
    records: list[dict],
    *,
    window: int = 8,
    min_truncated: int = 1,
    max_multiplier: float = 1.4,
    min_increase: int = 1000,
    max_value: int = 65536,
) -> int | None:
    """Return the ``max_tokens`` to use for the next run.

    Reads the *most recent* ``window`` runs and, when at least
    *min_truncated* of them were cut off mid-generation (the recording half
    stamps ``truncated: true`` on a run when any step hit
    ``finish_reason == "length"``), grow the budget by a bounded factor so
    the model has more room to complete its answer next time.

    The growth is bounded in three independent ways — each an independent
    gate that can veto it:

    1.  *max_multiplier* caps the per-run growth (default ``×1.4``);
    2.  *min_increase* caps the absolute increase (default ``+1000`` tokens);
    3.  *max_value* caps the absolute ceiling (default ``65536`` — the same
        hard cap the loop's :meth:`AgentLoop._effective_max_tokens` already
        applies).

    ``None`` / 0 ``current`` returns unchanged; an empty ``records`` returns
    unchanged.  Pure: no I/O, no log noise, no state mutation.
    """
    if not records or current is None or current <= 0:
        return current if current is not None else 0
    recent = records[-window:] if window > 0 else records
    truncated = 0
    for rec in recent:
        # ``truncated`` is stamped by the loop onto the run-metrics record.
        # ``finish_reason`` is not recorded (varies per step, not run) so we
        # rely on the aggregate flag.
        if rec.get("truncated"):
            truncated += 1
    if truncated < max(1, min_truncated):
        return current
    candidate = int(current * max_multiplier)
    candidate = min(candidate, current + min_increase)
    candidate = min(candidate, max_value)
    if candidate <= current:
        return current
    return candidate
