"""Tests for plan state persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from natshell.agent.plan_state import (
    PlanState,
    StepResult,
    get_resume_point,
    load_plan_state,
    save_plan_state,
    state_path_for_plan,
)

# ─── state_path_for_plan ────────────────────────────────────────────────────


class TestStatePath:
    def test_state_path_for_plan(self):
        assert state_path_for_plan(Path("/tmp/PLAN.md")) == Path("/tmp/PLAN.state.json")

    def test_state_path_preserves_directory(self):
        p = state_path_for_plan(Path("/home/user/project/PLAN.md"))
        assert p.parent == Path("/home/user/project")
        assert p.name == "PLAN.state.json"


# ─── save / load roundtrip ──────────────────────────────────────────────────


class TestSaveLoad:
    def test_roundtrip(self, tmp_path: Path):
        state = PlanState(
            plan_file="PLAN.md",
            plan_title="Test Plan",
            total_steps=3,
            step_results=[
                StepResult(number=1, title="First", status="passed", summary="1. First ✓"),
                StepResult(
                    number=2, title="Second", status="failed",
                    summary="2. Second ✗", error="boom",
                ),
            ],
            completed_summaries=["1. First ✓", "2. Second ✗"],
            completed_files=["src/a.py (created in step 1)"],
            started_at="2026-01-01T00:00:00Z",
        )
        path = tmp_path / "state.json"
        save_plan_state(state, path)

        loaded = load_plan_state(path)
        assert loaded.plan_title == "Test Plan"
        assert loaded.total_steps == 3
        assert len(loaded.step_results) == 2
        assert loaded.step_results[0].status == "passed"
        assert loaded.step_results[1].error == "boom"
        assert loaded.completed_summaries == ["1. First ✓", "2. Second ✗"]
        assert loaded.completed_files == ["src/a.py (created in step 1)"]

    def test_load_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_plan_state(tmp_path / "nope.json")

    def test_load_malformed_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("{invalid json")
        with pytest.raises(ValueError):
            load_plan_state(bad)

    def test_atomic_write_no_tmp_leftover(self, tmp_path: Path):
        state = PlanState(
            plan_file="PLAN.md", plan_title="T", total_steps=1,
            started_at="2026-01-01T00:00:00Z",
        )
        path = tmp_path / "state.json"
        save_plan_state(state, path)
        assert path.exists()
        assert not Path(str(path) + ".tmp").exists()


# ─── get_resume_point ────────────────────────────────────────────────────────


class TestGetResumePoint:
    def test_all_passed(self):
        state = PlanState(
            plan_file="P", plan_title="T", total_steps=3,
            step_results=[
                StepResult(number=i, title=f"S{i}", status="passed", summary="")
                for i in range(1, 4)
            ],
        )
        assert get_resume_point(state) == 3

    def test_first_failed(self):
        state = PlanState(
            plan_file="P", plan_title="T", total_steps=2,
            step_results=[
                StepResult(number=1, title="S1", status="failed", summary=""),
            ],
        )
        assert get_resume_point(state) == 0

    def test_middle_failed(self):
        state = PlanState(
            plan_file="P", plan_title="T", total_steps=3,
            step_results=[
                StepResult(number=1, title="S1", status="passed", summary=""),
                StepResult(number=2, title="S2", status="passed", summary=""),
                StepResult(number=3, title="S3", status="partial", summary=""),
            ],
        )
        assert get_resume_point(state) == 2

    def test_empty_results(self):
        state = PlanState(plan_file="P", plan_title="T", total_steps=3)
        assert get_resume_point(state) == 0
