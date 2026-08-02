"""Git integration tool — structured access to common git operations."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess

from natshell.tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

DEFINITION = ToolDefinition(
    name="git_tool",
    description=(
        "Perform common git operations in the current repository. "
        "Supported operations: status, diff, log, branch, commit, stash. "
        "Read-only operations (status, diff, log, branch) are safe; "
        "mutating operations (commit, stash) require confirmation. "
        "Prefer this over execute_shell for git tasks — it returns "
        "clean, structured output."
    ),
    parameters={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["status", "diff", "log", "branch", "commit", "stash"],
                "description": "The git operation to perform.",
            },
            "args": {
                "type": "string",
                "description": (
                    "Additional arguments for the operation. Examples:\n"
                    "  status: (none needed)\n"
                    "  diff: '--staged' or a file path\n"
                    "  log: '-5' to show last 5 commits (default 10)\n"
                    "  branch: branch name to create, or empty to list\n"
                    "  commit: '-m \"commit message\"' (required)\n"
                    "  stash: 'push', 'pop', 'list', or 'push -m \"message\"'"
                ),
            },
        },
        "required": ["operation"],
    },
    requires_confirmation=False,  # Safety classifier handles per-operation checks
)

# Operations that only read repository state
_SAFE_OPERATIONS = {"status", "diff", "log", "branch"}

# Operations that mutate repository state
_CONFIRM_OPERATIONS = {"commit", "stash"}

# Flags blocked in git commit — use execute_shell for these (goes through safety classifier)
_BLOCKED_COMMIT_FLAGS = {"--amend", "--reset-author", "--allow-empty-message"}
_BLOCKED_COMMIT_PREFIXES = ("--author=", "--date=")

# Flags blocked in git branch — use execute_shell for destructive branch ops
_BLOCKED_BRANCH_FLAGS = {"-D", "-M", "--force", "--delete"}
_BLOCKED_BRANCH_PREFIXES = ("--force",)

# Subcommands blocked in git stash — use execute_shell for destructive stash ops
_BLOCKED_STASH_FLAGS = {"drop", "clear"}

# Read-only operations are classified SAFE and so run with no confirmation
# dialog.  That makes their flags load-bearing: `git diff --output=FILE`
# creates and truncates FILE even when the command then errors out, which is an
# arbitrary-file-write primitive reachable without any user interaction.  Flags
# are therefore allowlisted rather than denylisted.  Only arguments starting
# with "-" are checked — paths, revisions and pathspecs pass through untouched.
_ALLOWED_READ_FLAGS: dict[str, frozenset[str]] = {
    "status": frozenset(
        {
            "--porcelain", "--short", "-s", "--branch", "-b", "--long",
            "--untracked-files", "-u", "--ignored", "--no-renames",
            "--find-renames", "-z", "--column", "--no-column",
        }
    ),
    "diff": frozenset(
        {
            "--stat", "--numstat", "--shortstat", "--summary", "--cached",
            "--staged", "--name-only", "--name-status", "--patch", "-p", "-u",
            "--no-patch", "-s", "--unified", "-U", "--raw", "--word-diff",
            "--color", "--no-color", "--check", "--find-renames", "-M",
            "--find-copies", "-C", "--diff-filter", "--ignore-all-space", "-w",
            "--ignore-space-change", "-b", "--ignore-blank-lines", "-R",
            "--text", "-a", "--binary", "--full-index", "--abbrev", "-z",
            "--no-renames", "--relative", "--function-context", "-W",
        }
    ),
    "log": frozenset(
        {
            "--oneline", "--no-decorate", "--decorate", "--graph", "--stat",
            "--shortstat", "--numstat", "--name-only", "--name-status",
            "--max-count", "-n", "--skip", "--since", "--after", "--until",
            "--before", "--author", "--committer", "--grep", "--all",
            "--first-parent", "--merges", "--no-merges", "--reverse",
            "--format", "--pretty", "--abbrev-commit", "--date", "--follow",
            "--color", "--no-color", "-p", "--patch", "-z",
        }
    ),
    # `branch` also creates/renames/deletes, so the non-listing flags git
    # treats as safe are included here.  The force variants (-D, -M, -C,
    # --force, --delete) are deliberately absent and stay rejected.
    "branch": frozenset(
        {
            "--list", "-l", "-v", "-vv", "--verbose", "-a", "--all", "-r",
            "--remotes", "--merged", "--no-merged", "--contains",
            "--no-contains", "--sort", "--color", "--no-color", "--column",
            "--show-current", "-d", "-m", "-c", "--track", "--no-track",
            "--set-upstream-to", "--unset-upstream",
        }
    ),
}


def _reject_disallowed_flags(operation: str, extra_args: list[str]) -> str | None:
    """Return an error message if any flag is not allowlisted for ``operation``.

    Handles both ``--flag=value`` and bare ``--flag`` spellings, plus the
    attached-value short forms git accepts (``-U5``, ``-n10``, ``-M50%``).
    """
    allowed = _ALLOWED_READ_FLAGS.get(operation)
    if allowed is None:
        return None

    for arg in extra_args:
        # Everything after the `--` separator is a pathspec by definition
        if arg == "--":
            break
        if not arg.startswith("-") or arg == "-":
            continue  # a path, revision or pathspec — not our business

        # `-10` is git's shorthand for `--max-count=10`
        if operation == "log" and arg[1:].isdigit():
            continue

        name = arg.split("=", 1)[0]
        if name in allowed:
            continue
        # Short flags may carry their value attached: -U5, -n10, -M50%
        if len(name) > 2 and not name.startswith("--") and name[:2] in allowed:
            continue

        return (
            f"Flag {arg!r} is not allowed for the read-only git '{operation}' "
            "operation via git_tool. Use execute_shell if you really need it — "
            "that path goes through the safety classifier and will ask first."
        )
    return None


def _run_git(args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run a git command synchronously (to be called via asyncio.to_thread)."""
    return subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=cwd or os.getcwd(),
    )


def _is_git_repo(cwd: str | None = None) -> bool:
    """Check whether the current directory is inside a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd or os.getcwd(),
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _format_status(result: subprocess.CompletedProcess[str]) -> str:
    """Parse git status --porcelain=v1 into a structured summary."""
    if result.returncode != 0:
        return result.stderr.strip()

    lines = result.stdout.rstrip().splitlines()
    if not lines or all(not line.strip() for line in lines):
        return "Working tree clean — nothing to commit."

    staged: list[str] = []
    unstaged: list[str] = []
    untracked: list[str] = []

    for line in lines:
        if len(line) < 3:
            continue
        x, y = line[0], line[1]
        path = line[3:]

        if x == "?":
            untracked.append(path)
        else:
            if x not in (" ", "?"):
                staged.append(f"  {x} {path}")
            if y not in (" ", "?"):
                unstaged.append(f"  {y} {path}")

    parts: list[str] = []
    if staged:
        parts.append("Staged changes:\n" + "\n".join(staged))
    if unstaged:
        parts.append("Unstaged changes:\n" + "\n".join(unstaged))
    if untracked:
        parts.append("Untracked files:\n" + "\n".join(f"  {f}" for f in untracked))

    return "\n\n".join(parts) if parts else "Working tree clean — nothing to commit."


def _format_log(result: subprocess.CompletedProcess[str]) -> str:
    """Return formatted log output (already formatted by git --oneline)."""
    if result.returncode != 0:
        return result.stderr.strip()
    output = result.stdout.strip()
    return output if output else "No commits yet."


def _format_branch(result: subprocess.CompletedProcess[str]) -> str:
    """Return branch listing or creation result."""
    if result.returncode != 0:
        return result.stderr.strip()
    return result.stdout.strip() if result.stdout.strip() else result.stderr.strip()


def _format_diff(result: subprocess.CompletedProcess[str]) -> str:
    """Return unified diff output."""
    if result.returncode != 0 and result.returncode != 1:
        # git diff exits 1 when there are differences (with --exit-code),
        # but without that flag it exits 0 regardless
        return result.stderr.strip()
    output = result.stdout.strip()
    return output if output else "No differences."


def _format_stash(result: subprocess.CompletedProcess[str]) -> str:
    """Return stash operation result."""
    if result.returncode != 0:
        return result.stderr.strip()
    output = result.stdout.strip()
    return output if output else "Stash operation completed."


def _format_commit(result: subprocess.CompletedProcess[str]) -> str:
    """Return commit result."""
    if result.returncode != 0:
        return result.stderr.strip()
    # git commit outputs to stdout
    output = result.stdout.strip()
    return output if output else result.stderr.strip()


async def git_tool(operation: str, args: str = "") -> ToolResult:
    """Execute a git operation and return structured results."""
    if operation not in (_SAFE_OPERATIONS | _CONFIRM_OPERATIONS):
        return ToolResult(
            error=f"Unknown git operation: {operation}. "
            f"Supported: {', '.join(sorted(_SAFE_OPERATIONS | _CONFIRM_OPERATIONS))}",
            exit_code=1,
        )

    # Check we're in a git repo
    is_repo = await asyncio.to_thread(_is_git_repo)
    if not is_repo:
        return ToolResult(
            error="Not a git repository (or any parent up to mount point).",
            exit_code=1,
        )

    # Split user-provided args string into a list, respecting shell quoting
    import shlex

    try:
        extra_args = shlex.split(args) if args else []
    except ValueError as e:
        return ToolResult(error=f"Invalid arguments: {e}", exit_code=1)

    # Read-only operations run unconfirmed, so their flags are allowlisted.
    flag_error = _reject_disallowed_flags(operation, extra_args)
    if flag_error:
        return ToolResult(error=flag_error, exit_code=1)

    try:
        if operation == "status":
            result = await asyncio.to_thread(
                _run_git, ["status", "--porcelain=v1"] + extra_args
            )
            return ToolResult(output=_format_status(result), exit_code=result.returncode)

        elif operation == "diff":
            result = await asyncio.to_thread(_run_git, ["diff"] + extra_args)
            return ToolResult(output=_format_diff(result), exit_code=result.returncode)

        elif operation == "log":
            # Default to last 10 commits, one-line format
            log_args = ["log", "--oneline", "--no-decorate"]
            if not any(a.startswith("-") and a[1:].isdigit() for a in extra_args):
                log_args.append("-10")
            log_args += extra_args
            result = await asyncio.to_thread(_run_git, log_args)
            return ToolResult(output=_format_log(result), exit_code=result.returncode)

        elif operation == "branch":
            if extra_args:
                # Block destructive branch flags
                for arg in extra_args:
                    if arg in _BLOCKED_BRANCH_FLAGS or any(
                        arg.startswith(p) for p in _BLOCKED_BRANCH_PREFIXES
                    ):
                        return ToolResult(
                            error=f"Flag {arg!r} is not allowed via git_tool. "
                            "Use execute_shell for destructive branch operations.",
                            exit_code=1,
                        )
                # Create a new branch
                result = await asyncio.to_thread(
                    _run_git, ["branch"] + extra_args
                )
            else:
                # List branches
                result = await asyncio.to_thread(
                    _run_git, ["branch", "--list", "-v"]
                )
            return ToolResult(output=_format_branch(result), exit_code=result.returncode)

        elif operation == "commit":
            if not extra_args:
                return ToolResult(
                    error="commit requires arguments, e.g. -m \"your message\"",
                    exit_code=1,
                )
            # Block dangerous flags — use execute_shell for these
            for arg in extra_args:
                if arg in _BLOCKED_COMMIT_FLAGS or any(
                    arg.startswith(p) for p in _BLOCKED_COMMIT_PREFIXES
                ):
                    return ToolResult(
                        error=f"Flag {arg!r} is not allowed via git_tool. "
                        "Use execute_shell for advanced git commit options.",
                        exit_code=1,
                    )
            result = await asyncio.to_thread(_run_git, ["commit"] + extra_args)
            return ToolResult(output=_format_commit(result), exit_code=result.returncode)

        elif operation == "stash":
            # Block destructive stash subcommands
            if extra_args and extra_args[0] in _BLOCKED_STASH_FLAGS:
                return ToolResult(
                    error=f"Subcommand {extra_args[0]!r} is not allowed via git_tool. "
                    "Use execute_shell for destructive stash operations.",
                    exit_code=1,
                )
            stash_args = ["stash"] + extra_args if extra_args else ["stash", "list"]
            result = await asyncio.to_thread(_run_git, stash_args)
            return ToolResult(output=_format_stash(result), exit_code=result.returncode)

        else:
            return ToolResult(error=f"Unhandled operation: {operation}", exit_code=1)

    except subprocess.TimeoutExpired:
        return ToolResult(
            error=f"git {operation} timed out after 30 seconds.",
            exit_code=124,
        )
    except FileNotFoundError:
        return ToolResult(
            error="git not found. Is git installed?",
            exit_code=127,
        )
    except Exception as e:
        return ToolResult(
            error=f"git {operation} failed: {type(e).__name__}: {e}",
            exit_code=1,
        )
