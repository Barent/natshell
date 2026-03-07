"""Plan execution state persistence — tracks step completion for resume support."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class StepResult:
    """Result of executing a single plan step."""

    number: int
    title: str
    status: str  # "passed", "failed", "partial"
    summary: str  # e.g. "1. Fix shapes ✓"
    files_changed: list[str] = field(default_factory=list)
    error: str | None = None
    wall_ms: int = 0
    inference_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    verify_attempts: int = 0
    log_lines: list[str] = field(default_factory=list)


@dataclass
class PlanState:
    """Persistent state for a plan execution run."""

    plan_file: str
    plan_title: str
    total_steps: int
    step_results: list[StepResult] = field(default_factory=list)
    completed_summaries: list[str] = field(default_factory=list)
    completed_files: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str | None = None
    total_wall_ms: int = 0
    total_inference_ms: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tool_calls: int = 0


def state_path_for_plan(plan_path: Path) -> Path:
    """Return the state file path for a given plan file.

    Example: PLAN.md -> PLAN.state.json
    """
    return plan_path.with_suffix(".state.json")


def save_plan_state(state: PlanState, path: Path) -> None:
    """Save plan state as JSON. Uses atomic write (tmp + rename)."""
    data = json.dumps(asdict(state), indent=2)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(str(tmp), str(path))


def load_plan_state(path: Path) -> PlanState:
    """Load plan state from a JSON file.

    Raises:
        FileNotFoundError: If the state file doesn't exist.
        ValueError: If the JSON is malformed or missing required fields.
    """
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed state file: {e}") from e

    try:
        step_results = [StepResult(**sr) for sr in raw.pop("step_results", [])]
        return PlanState(step_results=step_results, **raw)
    except (TypeError, KeyError) as e:
        raise ValueError(f"Invalid state file structure: {e}") from e


def get_resume_point(state: PlanState) -> int:
    """Return the 0-based index of the first non-passed step.

    If all steps passed, returns len(step_results) (resume after last).
    If no results, returns 0.
    """
    for i, result in enumerate(state.step_results):
        if result.status != "passed":
            return i
    return len(state.step_results)
