"""Background shell — launch a long-running command detached, tail its log, kill it.

R2-4 (background half).  ``execute_shell`` is bounded by a timeout and returns
only once the command ends; for servers, watchers, long builds or anything the
user wants to *start and then check on later*, this tool provides a
``launch` / `tail` / `kill`` trio over a small on-disk handle directory.

Design (one tool, three actions — keeps the prompt surface to a single
definition rather than three):

* ``launch(command)``   → spawn ``bash -c command`` detached (``start_new_session``
  → its own process group, so a single ``killpg`` takes down the whole tree),
  stdout+stderr appended to ``<id>.log``.  Returns a 16-hex *handle*.
* ``tail(handle, lines)`` → current status (running / finished+exit code) plus the
  last ``lines`` of the log.
* ``kill(handle)``      → ``killpg`` SIGTERM, escalate to SIGKILL after 2 s.

Lifetime is *session-scoped*: a live :class:`subprocess.Popen` is kept in an
in-memory registry (``_PROCS``) for the duration of the NatShell process, and a
daemon watcher thread reaps each child and stamps the handle JSON with
``finished=True`` + ``exit_code``.  Survived across a NatShell restart, status is
recovered from the handle JSON (``pid``-based) and ``kill`` falls back to
``killpg``/``kill`` by pid.  Orphans from sessions >24 h old are swept at
startup by :func:`clean_orphans`.

Security: the safety classifier classifies ``launch`` *exactly* like
``execute_shell`` (so a BLOCKED command such as ``rm -rf /`` stays BLOCKED even
though it's "detached"), and ``tail``/``kill`` fall through to the fails-closed
``CONFIRM`` default.  The handle directory is ``0o700``.  Environment passed to
the child is the same sensitive-filtered env ``execute_shell`` uses.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from natshell.platform import data_dir as _data_dir
from natshell.platform import is_windows
from natshell.tools.execute_shell import _filtered_env
from natshell.tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# How long a handle (by ``started_ts``) may age before it's swept as orphaned.
_ORPHAN_AGE_SECONDS = 24 * 60 * 60  # 24 h
# How long to let a killed process group react to SIGTERM before SIGKILL.
_KILL_GRACE_SECONDS = 2.0
# Default / max number of trailing log lines ``tail`` returns.
_DEFAULT_LINES = 50
_MAX_LINES = 1000
# How long (s) a single ``tail``/status liveness probe may take before giving up.
_LIVENESS_PROBE_SECONDS = 0.5


# ── Storage ─────────────────────────────────────────────────────────────────


def bg_dir() -> Path:
    """The background-handle directory (``<data_dir>/bg``)."""
    return _data_dir() / "bg"


def _ensure_dir() -> Path:
    """Create (``0o700``) and return the handle directory."""
    d = bg_dir()
    d.mkdir(parents=True, exist_ok=True)
    try:
        d.chmod(0o700)
    except OSError:
        pass
    return d


def _handle_path(handle: str) -> Path:
    return _ensure_dir() / f"{handle}.json"


def _log_path(handle: str) -> Path:
    return _ensure_dir() / f"{handle}.log"


def _read_handle(handle: str) -> dict[str, Any] | None:
    """Read a handle JSON, or ``None`` if it doesn't exist / is unreadable."""
    p = _handle_path(handle)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_handle(handle: str, data: dict[str, Any]) -> None:
    """Atomically write a handle JSON (tmp + ``os.replace``).  Holds
    :data:`_RW_LOCK` for the I/O so a concurrent reaper update can't interleave
    read-modify-write and clobber our fields."""
    with _RW_LOCK:
        p = _handle_path(handle)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, p)


def _update_handle(handle: str, updates: dict[str, Any]) -> None:
    """Merge ``updates`` into the on-disk handle JSON (read-modify-write).

    Serialized by :data:`_RW_LOCK` so the daemon reaper — which runs in a
    separate thread and re-reads the file just before it writes — can never
    stomp a field this path set.
    """
    with _RW_LOCK:
        data = _read_handle(handle) or {}
        data.update(updates)
        p = _handle_path(handle)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, p)


def _new_handle_id() -> str:
    """16-hex handle id, collision-checked against the on-disk directory."""
    d = _ensure_dir()
    names = {p.stem for p in d.glob("*.json")}
    while True:
        hid = secrets.token_hex(8)
        if hid not in names:
            return hid


# ── In-memory process registry (session-scoped) ────────────────────────────

# handle_id -> subprocess.Popen for a *live* launch in this NatShell process.
_PROCS: dict[str, subprocess.Popen] = {}
_LOCK = threading.Lock()

# Serializes *writers* of the handle JSON (the daemon reaper thread and any
# in-thread code path, e.g. _kill).  Both do read-modify-write; without it the
# reaper's "blind" write wins the last-writer race and clobbers fields the
# kill path just set.  Acquired only around the (short) file I/O, so it can't
# deadlock on the reaper's `proc.wait()` which runs *outside* this lock.
_RW_LOCK = threading.Lock()


def _register_proc(handle: str, proc: subprocess.Popen) -> None:
    with _LOCK:
        _PROCS[handle] = proc


def _pop_proc(handle: str) -> subprocess.Popen | None:
    with _LOCK:
        return _PROCS.pop(handle, None)


def reset_registry() -> None:
    """Drop all live processes tracked in this session (used by tests)."""
    with _LOCK:
        _PROCS.clear()


# ── Liveness / status ──────────────────────────────────────────────────────


def _pid_alive(pid: int | None) -> bool | None:
    """Best-effort liveness of ``pid``.

    Returns ``True``/``False`` when it can tell, ``None`` when it can't (e.g.
    no pid recorded, or a cross-user probe that raises ``PermissionError``).
    """
    if not pid:
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    except OSError:
        return None


def _poll_or_timeout(proc: subprocess.Popen) -> int | None:
    """Return the exit code if the child has exited within the probe window,
    else ``None`` (still running).  Never blocks past the window."""
    try:
        return proc.wait(timeout=_LIVENESS_PROBE_SECONDS)
    except subprocess.TimeoutExpired:
        return None


def _current_status(handle: str) -> tuple[str, str, int | None]:
    """Return ``(status_line, liveness, exit_code)`` for a handle.

    Prefers the live :class:`Popen` (session-scoped), then the JSON's recorded
    ``finished``/``exit_code``, then a pid liveness probe for post-restart handles.
    """
    data = _read_handle(handle)
    hid = handle
    pid = data.get("pid") if data else None
    exit_code = data.get("exit_code") if data else None
    finished = bool(data.get("finished")) if data else False

    proc = _procd_proc(hid)
    if proc is not None:
        rc = _poll_or_timeout(proc)
        if rc is not None:
            return f"finished (exit code {rc})", "finished", rc
        return f"running (pid {proc.pid})", "running", None

    # No live handle — rely on the persisted JSON, then a pid probe.
    if finished:
        return f"finished (exit code {exit_code})", "finished", exit_code
    live = _pid_alive(pid)
    if live is True:
        return f"running (pid {pid})", "running", None
    if live is False:
        # Process exited but we didn't reap it (NatShell died first). Infer 0
        # vs "unknown" — we can't read the child's exit status without waitpid.
        return "finished (exit code unknown — process reaped)", "finished", None
    # live is None: can't tell (no pid / permission).
    return "status unknown" if pid is None else f"unknown (pid {pid})", "unknown", None


def _procd_proc(handle: str) -> subprocess.Popen | None:
    with _LOCK:
        return _PROCS.get(handle)


# ── launch ─────────────────────────────────────────────────────────────────


def _reap(handle: str, proc: subprocess.Popen) -> None:
    """Daemon-thread body: wait for the child, stamp the handle, drop the ref."""
    try:
        rc = proc.wait()
    except Exception:  # pragma: no cover - defensive; never crash a worker
        rc = -1
    _update_handle(
        handle,
        {
            "finished": True,
            "exit_code": rc,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "finished_ts": time.time(),
        },
    )
    _pop_proc(handle)


def _launch(command: str) -> ToolResult:
    """Spawn a detached background child and record its handle."""
    command = (command or "").strip()
    if not command:
        return ToolResult(error="launch requires a non-empty 'command'.", exit_code=1)

    try:
        _ensure_dir()
        handle = _new_handle_id()
    except OSError as e:
        return ToolResult(error=f"Could not create background dir: {e}", exit_code=1)

    log = _log_path(handle)
    env = _filtered_env()
    env["LC_ALL"] = "C"

    # Build the same argv execute_shell uses (bash -c on POSIX, powershell on
    # Windows) so environment and shell behaviour stay in lockstep.
    if is_windows():
        shell_cmd: list[str] = [
            "powershell", "-NoProfile", "-NonInteractive", "-Command", command,
        ]
        creation_kwargs: dict[str, Any] = {
            "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            "stdin": subprocess.DEVNULL,
        }
    else:
        shell_cmd = ["bash", "-c", command]
        creation_kwargs = {
            "start_new_session": True,
            "stdin": subprocess.DEVNULL,
        }

    now = time.time()
    try:
        # stdout+stderr → the log file; we close our handle immediately so the
        # child's inherited fd is the sole writer and we never block on it.
        with open(log, "a", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                shell_cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=os.getcwd(),
                **creation_kwargs,
            )
    except FileNotFoundError:
        shell_name = "PowerShell" if is_windows() else "bash"
        return ToolResult(
            output="",
            error=f"{shell_name} not found. Is it installed?",
            exit_code=127,
        )
    except Exception as e:
        return ToolResult(
            output="",
            error=f"Failed to launch: {type(e).__name__}: {e}",
            exit_code=1,
        )

    data: dict[str, Any] = {
        "handle": handle,
        "command": command,
        "pid": proc.pid,
        "created_ts": now,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "finished": False,
        "exit_code": None,
        "log": log.name,
    }
    _write_handle(handle, data)
    _register_proc(handle, proc)

    # Detach: reap on the daemon thread, then let it run independently.
    watcher = threading.Thread(
        target=_reap, args=(handle, proc), name=f"shell-bg-{handle}", daemon=True
    )
    watcher.start()

    return ToolResult(
        output=(
            f"Launched background process.\n"
            f"handle : {handle}\n"
            f"pid    : {proc.pid}\n"
            f"log    : {log}\n"
            f"Command is running detached. Use shell_bg(action='tail', "
            f"handle='{handle}') to check on it, or shell_bg(action='kill', "
            f"handle='{handle}') to stop it."
        )
    )


# ── tail ───────────────────────────────────────────────────────────────────


def _tail(handle: str, lines: int = _DEFAULT_LINES) -> ToolResult:
    data = _read_handle(handle)
    if data is None:
        return ToolResult(
            error=f"Unknown background handle: {handle!r}. "
            "Launch it first, or check the handle id.",
            exit_code=1,
        )

    try:
        lines = max(1, min(int(lines), _MAX_LINES))
    except (TypeError, ValueError):
        lines = _DEFAULT_LINES

    status_line, _, _ = _current_status(handle)

    log = _ensure_dir() / f"{data.get('log', f'{handle}.log')}"
    tail_text = ""
    if log.exists():
        try:
            with open(log, "r", encoding="utf-8", errors="replace") as f:
                lines_out = [line.rstrip("\n") for line in f.readlines()[-lines:]]
            tail_text = "\n".join(lines_out)
        except OSError as e:
            tail_text = f"(could not read log: {e})"
    else:
        tail_text = "(no output yet)"

    out = [
        f"handle : {handle}",
        f"command: {data.get('command', '')}",
        f"status : {status_line}",
        f"pid    : {data.get('pid')}",
        f"started: {data.get('created_at', '')}",
        "",
        f"--- last {lines} lines of output ---",
        tail_text or "(empty)",
    ]
    return ToolResult(output="\n".join(out))


# ── kill ───────────────────────────────────────────────────────────────────


def _kill(handle: str) -> ToolResult:
    data = _read_handle(handle)
    if data is None:
        return ToolResult(
            error=f"Unknown background handle: {handle!r}.",
            exit_code=1,
        )
    pid = data.get("pid")
    if not pid and is_windows():
        return ToolResult(error="No pid recorded for this handle.", exit_code=1)

    # Prefer the live Popen (session-scoped) — its child is in its own group.
    proc = _procd_proc(handle)
    killed = False
    if proc is not None:
        rc = _poll_or_timeout(proc)
        if rc is None:  # still running → kill the whole session/process group
            killed = _kill_group(pid)
        else:
            _update_handle(
                handle,
                {"finished": True, "exit_code": rc, "finished_ts": time.time()},
            )
            _pop_proc(handle)
            return ToolResult(
                output=f"Handle {handle} was already finished (exit code {rc})."
            )
    else:
        # Post-restart or no live handle: fall back to a pid/group kill.
        live = _pid_alive(pid)
        if live is not True:
            suffix = " (already finished)" if data.get("finished") else ""
            return ToolResult(output=f"Handle {handle} is not running{suffix}.")
        killed = _kill_group(pid)

    if not killed:
        return ToolResult(
            error=f"Could not signal {handle!r} (pid {pid}).",
            exit_code=1,
        )
    # Record the kill.  The daemon reaper (still attached to this handle) will
    # merge its own exit-code/finished fields atomically via _update_handle; this
    # path sets `killed` first so that write carries through.
    _update_handle(handle, {"killed": True, "finished_ts": time.time()})
    return ToolResult(
        output=f"Terminated background handle {handle} (pid {pid})."
    )


def _kill_group(pid: Any) -> bool:
    """SIGTERM the process group, escalate to SIGKILL after the grace window.

    POSIX: ``killpg`` the child's group.  Windows: ``TerminateProcess`` via
    ``os.kill(pid, SIGTERM)`` then ``SIGKILL``.  Returns True iff the signal
    was delivered.  A non-integer/missing pid is a hard failure (False).
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False

    def _signal(sig: int) -> bool:
        try:
            if not is_windows():
                os.killpg(os.getpgid(pid), sig)
            else:
                os.kill(pid, sig)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False

    if not _signal(signal.SIGTERM):
        return False
    deadline = time.monotonic() + _KILL_GRACE_SECONDS
    while time.monotonic() < deadline:
        if _pid_alive(pid) is False:
            return True
        time.sleep(0.1)
    # Grace window elapsed — escalate.
    _signal(signal.SIGKILL)
    return True


# ── cleanup ────────────────────────────────────────────────────────────────


def clean_orphans(max_age_hours: int = 24) -> int:
    """Sweep background handles older than ``max_age_hours``.

    Called at startup.  Only touches the on-disk handle/log pair for entries
    whose ``created_ts`` (falling back to file mtime) is older than the window
    and that are not a live child of *this* process.  Returns the number of
    handles swept.  Never raises.
    """
    max_age = max(1, max_age_hours) * 3600
    d = bg_dir()
    if not d.exists():
        return 0
    now = time.time()
    swept = 0
    for handle_file in d.glob("*.json"):
        try:
            hid = handle_file.stem
            with _LOCK:
                live = hid in _PROCS
            if live:
                continue
            data = None
            try:
                data = json.loads(handle_file.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                pass
            ts = data.get("created_ts") if data else None
            if ts is None:
                try:
                    ts = handle_file.stat().st_mtime
                except OSError:
                    continue
            if now - float(ts) < max_age:
                continue
            handle_file.unlink(missing_ok=True)
            d.joinpath(f"{hid}.log").unlink(missing_ok=True)
            swept += 1
        except OSError:
            continue
    return swept


# ── Tool definition + entrypoint ───────────────────────────────────────────


DEFINITION = ToolDefinition(
    name="shell_bg",
    description=(
        "Manage long-running background shell processes. 'launch' starts a command "
        "detached (survives beyond this tool call) and returns a handle; 'tail' "
        "shows its current status (running / finished + exit code) and recent "
        "output; 'kill' stops it. Use for servers, watchers, long builds, or anything "
        "you want to start now and check on later. For short, bounded commands use "
        "execute_shell instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["launch", "tail", "kill"],
                "description": "Which operation: launch (start), tail (check), kill (stop).",
            },
            "command": {
                "type": "string",
                "description": "The command to run in the background. Required for 'launch'.",
            },
            "handle": {
                "type": "string",
                "description": "The 16-hex handle returned by launch. "
                "Required for 'tail' and 'kill'.",
            },
            "lines": {
                "type": "integer",
                "description": "Number of trailing log lines to show with 'tail'. Default 50.",
            },
        },
        "required": ["action"],
    },
)


async def shell_bg(
    action: str,
    command: str = "",
    handle: str = "",
    lines: int = _DEFAULT_LINES,
) -> ToolResult:
    """Dispatch one of the three background-shell actions."""
    action = (action or "").strip().lower()
    if action == "launch":
        if not (command or "").strip():
            return ToolResult(
                error="shell_bg launch requires a 'command'.",
                exit_code=1,
            )
        return _launch(command)
    if action == "tail":
        if not (handle or "").strip():
            return ToolResult(
                error="shell_bg tail requires the 'handle' returned by launch.",
                exit_code=1,
            )
        return _tail(handle, lines)
    if action == "kill":
        if not (handle or "").strip():
            return ToolResult(
                error="shell_bg kill requires the 'handle' returned by launch.",
                exit_code=1,
            )
        return _kill(handle)
    return ToolResult(
        error=f"Unknown shell_bg action: {action!r}. Use 'launch', 'tail', or 'kill'.",
        exit_code=1,
    )
