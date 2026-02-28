"""Markdown plan parser — splits a plan file into discrete executable steps."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PlanStep:
    """A single step extracted from a markdown plan."""

    number: int  # 1-based
    title: str  # e.g. "Fix piece shapes in theme.cpp"
    body: str  # Full markdown body including code blocks


@dataclass
class Plan:
    """A parsed multi-step plan."""

    title: str  # From # heading
    preamble: str  # Text before first step
    steps: list[PlanStep] = field(default_factory=list)
    source_dir: Path | None = None  # Directory containing the plan file


# Matches ## headings, optionally prefixed with "Step N:" or "N."
_STEP_HEADING_RE = re.compile(
    r"^##\s+"
    r"(?:(?:Step\s+)?(\d+)[.:]\s*)?"  # Optional "Step N:" or "N."
    r"(.+)$"
)


def parse_plan_file(path: str | Path) -> Plan:
    """Parse a markdown plan file into a Plan with discrete steps.

    The file should have:
    - An optional # heading (used as plan title)
    - Optional preamble text before the first ## heading
    - One or more ## headings, each starting a new step

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If no ## headings (steps) are found.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Plan file not found: {path}")

    text = path.read_text()
    plan = parse_plan_text(text)
    plan.source_dir = path.parent
    return plan


def parse_plan_text(text: str) -> Plan:
    """Parse plan text (already loaded) into a Plan.

    Raises:
        ValueError: If no ## headings (steps) are found.
    """
    lines = text.splitlines()

    title = "Untitled Plan"
    preamble_lines: list[str] = []
    steps: list[PlanStep] = []
    current_title: str | None = None
    current_body_lines: list[str] = []
    in_preamble = True

    for line in lines:
        # Check for H1 title (only take the first one)
        if in_preamble and line.startswith("# ") and not line.startswith("## "):
            title = line[2:].strip()
            continue

        # Check for H2 step heading
        m = _STEP_HEADING_RE.match(line)
        if m:
            # Flush previous step
            if current_title is not None:
                steps.append(
                    PlanStep(
                        number=len(steps) + 1,
                        title=current_title,
                        body="\n".join(current_body_lines).strip(),
                    )
                )
            elif in_preamble:
                pass  # preamble ends when first ## heading is found

            in_preamble = False
            current_title = m.group(2).strip()
            current_body_lines = []
            continue

        if in_preamble:
            preamble_lines.append(line)
        else:
            current_body_lines.append(line)

    # Flush last step
    if current_title is not None:
        steps.append(
            PlanStep(
                number=len(steps) + 1,
                title=current_title,
                body="\n".join(current_body_lines).strip(),
            )
        )

    if not steps:
        raise ValueError("No steps found — expected at least one ## heading")

    preamble = "\n".join(preamble_lines).strip()

    return Plan(title=title, preamble=preamble, steps=steps)
