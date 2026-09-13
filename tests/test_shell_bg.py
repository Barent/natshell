"""Tests for the shell_bg (background shell) tool — R2-4 background half.

Covers the launch / tail / kill lifecycle, the on-disk handle JSON + log,
the session-scoped live-registry + daemon reaper, and the safety classifier
rules. Real child processes are spawned (``bash -c``) inside a bg dir
redirected to pytest's ``tmp_path``. Only the classifier is exercised against
real config patterns.
"""

from __future__ import annotations

import json
import os
import stat
import time

import pytest

import natshell.tools.shell_bg as sbg
from natshell.config import SafetyConfig
from natshell.safety.classifier import Risk, SafetyClassifier
from natshell.tools.registry import (
    PLAN_SAFE_TOOLS,
    SMALL_CONTEXT_TOOLS,
    create_default_registry,
)
from natshell.tools.shell_bg import (
    DEFINITION,
    _kill_group,
    _launch,
    _pid_alive,
    _tail,
    clean_orphans,
    reset_registry,
    shell_bg,
)

# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def bg(tmp_path, monkeypatch):
    """Redirect the bg dir into tmp_path and clear the in-memory registry.

    Yields the pre-created bg dir. Every test that launches a real child
    should kill it / let the reaper finish it before the test exits; the
    fixture teardown is a last-resort sweep that SIGTERMs any live child of
    this test (best-effort, never an assertion) and drops the registry.
    """
    d = tmp_path / "bg"
    d.mkdir(exist_ok=True)
    monkeypatch.setattr(sbg, "_data_dir", lambda: tmp_path)
    reset_registry()
    yielded = d
    try:
        yield yielded
    finally:
        # Last-resort sweep: kill any child we launched (best-effort) — the
        # in-memory registry may have already dropped it if the reaper ran.
        with sbg._LOCK:
            live = list(sbg._PROCS.items())
        for hid, proc in live:
            try:
                _kill_group(proc.pid)
            except Exception:
                pass
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass
        reset_registry()
        # Leave the tmp dir alone: tests may still read from it.


def _launch_and_wait_finished(cmd: str, timeout: float = 10.0) -> str:
    """Launch a command, wait until the reaper marks it finished, return the handle."""
    res = _launch(cmd)
    assert res.exit_code == 0, res.error
    hid = _extract_handle(res.output)
    _wait_finished(hid, timeout=timeout)
    return hid


def _extract_handle(output: str) -> str:
    assert "handle : " in output, output
    return output.split("handle : ")[1].split("\n")[0].strip()


def _wait_finished(hid: str, timeout: float = 10.0) -> dict:
    """Poll the handle JSON until ``finished`` is true (reaper wrote it)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = sbg._read_handle(hid)
        if data and data.get("finished"):
            return data
        time.sleep(0.05)
    raise AssertionError(f"handle {hid} did not finish in {timeout}s")


def _launch_running(cmd: str = "bash -c 'sleep 30; echo done'") -> str:
    """Launch a long-running command; return the handle without touching it."""
    res = _launch(cmd)
    assert res.exit_code == 0, res.error
    return _extract_handle(res.output)


# ── definition / registration ────────────────────────────────────────────────


class TestDefinition:
    def test_name_and_actions(self):
        assert DEFINITION.name == "shell_bg"
        actions = DEFINITION.parameters["properties"]["action"]["enum"]
        assert actions == ["launch", "tail", "kill"]
        assert "action" in DEFINITION.parameters["required"]

    def test_registered_in_default_registry(self):
        assert "shell_bg" in create_default_registry().tool_names

    def test_schema_generated(self):
        registry = create_default_registry()
        names = [s["function"]["name"] for s in registry.get_tool_schemas()]
        assert "shell_bg" in names

    def test_not_in_small_context_tools(self):
        # The plan: ≤8K windows keep today's surface.
        assert "shell_bg" not in SMALL_CONTEXT_TOOLS

    def test_not_in_plan_safe_tools(self):
        assert "shell_bg" not in PLAN_SAFE_TOOLS


# ── launch ───────────────────────────────────────────────────────────────────


class TestLaunch:
    def test_writes_handle_and_log(self, bg):
        hid = _launch_running()
        data = sbg._read_handle(hid)
        assert data is not None
        assert len(hid) == 16
        assert data["pid"] is not None
        assert data["finished"] is False
        log = bg / data["log"]
        assert log.exists()
        assert data["command"] == "bash -c 'sleep 30; echo done'"

    def test_dir_permissions_0700(self, bg):
        _launch_running()
        mode = stat.S_IMODE(os.stat(bg).st_mode)
        assert mode == 0o700

    def test_launch_requires_command(self, bg):
        res = _launch("")
        assert res.exit_code == 1
        assert "command" in res.error.lower()

    async def test_launch_async_dispatch_requires_command(self, bg):
        res = await shell_bg("launch")
        assert res.exit_code == 1
        assert "command" in res.error.lower()


# ── tail ─────────────────────────────────────────────────────────────────────


class TestTail:
    def test_tail_unknown_handle(self, bg):
        res = _tail("deadbeefcafe0001")
        assert res.exit_code == 1
        assert "Unknown background handle" in res.error

    def test_tail_running_then_finished(self, bg):
        hid = _launch_running()
        # Allow the child to (attempt to) start — the reaper is still waiting.
        time.sleep(0.3)
        res = _tail(hid)
        assert res.exit_code == 0
        assert "running" in res.output
        assert hid in res.output

        # Kill + reap, then tail must report finished.
        assert sbg._kill(hid).exit_code == 0
        _wait_finished(hid)
        r2 = _tail(hid)
        assert "finished" in r2.output
        assert "exit code" in r2.output

    def test_tail_truncates_to_lines(self, bg):
        hid = _launch_and_wait_finished("bash -c 'for i in 1 2 3 4 5; do echo line_$i; done'")
        r = _tail(hid, lines=2)
        assert r.exit_code == 0
        # Last 2 lines are line_4 and line_5.
        assert "line_4" in r.output
        assert "line_5" in r.output
        assert "line_1" not in r.output

    def test_tail_coerces_bad_lines(self, bg):
        hid = _launch_running()
        try:
            # Non-integer lines should fall back to the default, not crash.
            r = _tail(hid, "not-an-int")  # type: ignore[arg-type]
            assert r.exit_code == 0
        finally:
            # Ensure the child doesn't linger.
            sbg._kill(hid)


# ── kill ─────────────────────────────────────────────────────────────────────


class TestKill:
    def test_kill_running_process(self, bg):
        hid = _launch_running("bash -c 'trap \":\" TERM; sleep 30'")
        # Give the child a moment to actually receive the trap registration.
        time.sleep(0.4)
        res = sbg._kill(hid)
        assert res.exit_code == 0, res.error
        assert "Terminated" in res.output
        data = _wait_finished(hid)
        assert data["finished"] is True
        assert data["exit_code"] is not None

    def test_kill_already_finished(self, bg):
        hid = _launch_and_wait_finished("bash -c 'exit 7'")
        r = sbg._kill(hid)
        assert "finished" in r.output
        # Second kill is idempotent.
        r2 = sbg._kill(hid)
        assert ("not running" in r2.output) or ("finished" in r2.output)

    def test_kill_unknown_handle(self, bg):
        res = sbg._kill("deadbeefcafe0001")
        assert res.exit_code == 1
        assert "Unknown" in res.error

    def test_kill_group_invalid_pid(self):
        assert _kill_group(0) is False
        assert _kill_group(None) is False
        assert _kill_group("123") is False
        assert _kill_group(True) is False


# ── async / CLI entrypoint ───────────────────────────────────────────────────


class TestEntryPoint:
    async def test_unknown_action(self, bg):
        res = await shell_bg("frobnicate")
        assert res.exit_code == 1
        assert "Unknown shell_bg action" in res.error

    async def test_launch_dispatch(self, bg):
        res = await shell_bg("launch", command="bash -c 'echo hi'")
        assert res.exit_code == 0
        assert "handle" in res.output

    async def test_dispatch_unknown_action_rejected(self, bg):
        res = await shell_bg("nope")
        assert res.exit_code == 1


# ── liveness / cleanup ───────────────────────────────────────────────────────


class TestLivenessAndCleanup:
    def test_pid_alive_none_pid(self):
        assert _pid_alive(0) is None
        assert _pid_alive(None) is None

    def test_pid_alive_self_true(self):
        # Our own pid is always alive (and ours, so not PermissionError).
        assert _pid_alive(os.getpid()) is True

    def test_clean_orphans_sweeps_old_keeps_new(self, bg):
        # Create an old handle (48 h old) and a new one (now).
        old_hid = "a" * 16
        new_hid = "b" * 16
        now = time.time()
        for hid, age in ((old_hid, 48 * 3600), (new_hid, 0)):
            (bg / f"{hid}.json").write_text(json.dumps({"created_ts": now - age}))
            (bg / f"{hid}.log").write_text("x")
        # Sweep at the 24 h threshold: old one removed, new one kept.
        swept = clean_orphans(max_age_hours=24)
        assert swept == 1
        assert not (bg / f"{old_hid}.json").exists()
        assert not (bg / f"{old_hid}.log").exists()
        assert (bg / f"{new_hid}.json").exists()

    def test_clean_orphans_skips_live(self, bg):
        # A handle that is "live" in this session must not be swept.
        hid = "c" * 16
        (bg / f"{hid}.json").write_text(json.dumps({"created_ts": time.time() - 48 * 3600}))
        (bg / f"{hid}.log").write_text("x")
        # Register a live marker without a real Popen — the registry only
        # stores a dict of Popen; the presence of the key is what matters.
        with sbg._LOCK:
            sbg._PROCS[hid] = object()  # type: ignore[assignment]
        swept = clean_orphans(max_age_hours=24)
        assert swept == 0
        assert (bg / f"{hid}.json").exists()
        with sbg._LOCK:
            sbg._PROCS.pop(hid, None)

    def test_clean_orphans_missing_dir(self, tmp_path, monkeypatch):
        # A bg dir that doesn't exist is a valid, swept-zero state.
        monkeypatch.setattr(sbg, "_data_dir", lambda: tmp_path / "nowhere")
        assert clean_orphans() == 0


# ── classifier ───────────────────────────────────────────────────────────────


def _classifier(mode: str = "confirm") -> SafetyClassifier:
    return SafetyClassifier(
        SafetyConfig(
            mode=mode,
            always_confirm=[r"^rm\s", r"^dd\s"],
            blocked=[r"^rm\s+-[rR]f\s+/\s*$"],
        )
    )


class TestClassifierShellBg:
    def test_launch_blocked_stays_blocked(self):
        c = _classifier()
        r = c.classify_tool_call("shell_bg", {"action": "launch", "command": "rm -rf /"})
        assert r == Risk.BLOCKED

    def test_launch_confirm(self):
        c = _classifier()
        r = c.classify_tool_call("shell_bg", {"action": "launch", "command": "rm file"})
        assert r == Risk.CONFIRM

    def test_launch_safe(self):
        c = _classifier()
        r = c.classify_tool_call("shell_bg", {"action": "launch", "command": "echo hi"})
        assert r == Risk.SAFE

    def test_tail_confirm_by_default(self):
        c = _classifier()
        r = c.classify_tool_call("shell_bg", {"action": "tail", "handle": "abc"})
        assert r == Risk.CONFIRM

    def test_kill_confirm_by_default(self):
        c = _classifier()
        r = c.classify_tool_call("shell_bg", {"action": "kill", "handle": "abc"})
        assert r == Risk.CONFIRM

    def test_danger_tail_and_kill_safe(self):
        c = _classifier(mode="danger")
        assert c.classify_tool_call("shell_bg", {"action": "tail"}) == Risk.SAFE
        assert c.classify_tool_call("shell_bg", {"action": "kill"}) == Risk.SAFE

    def test_danger_launch_confirms_downgrade(self):
        c = _classifier(mode="danger")
        r = c.classify_tool_call("shell_bg", {"action": "launch", "command": "rm file"})
        assert r == Risk.SAFE

    def test_danger_launch_blocked_stays_blocked(self):
        c = _classifier(mode="danger")
        r = c.classify_tool_call("shell_bg", {"action": "launch", "command": "rm -rf /"})
        assert r == Risk.BLOCKED
