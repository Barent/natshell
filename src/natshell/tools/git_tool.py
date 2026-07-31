"""Git integration tool — structured access to common git operations."""

from __future__ import annotations

import asyncio
import logging
import os
import re
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

# Flags permitted for the read-only operations.
#
# These four are classified SAFE and run with no confirmation dialog, so the
# question is not "which flags are known to be dangerous" but "which flags are
# known to be harmless".  `git diff --output=<path>` writes an arbitrary file;
# --ext-diff and --textconv run a configured external program.  A denylist has
# to be updated every time git grows another one.
#
# Only flags are checked.  A bare argument is a path, a revision or a branch
# name — none of which can write a file or start a process — so revision ranges
# and pathspecs continue to work untouched.
_ALLOWED_READ_FLAGS: dict[str, frozenset[str]] = {
    "status": frozenset(
        {
            "--short", "-s", "--branch", "-b", "--porcelain", "--long", "--verbose",
            "-v", "--ignored", "--untracked-files", "-u", "--no-renames",
            "--find-renames", "--column", "--no-column", "--ahead-behind",
        }
    ),
    "diff": frozenset(
        {
            "--staged", "--cached", "--stat", "--numstat", "--shortstat",
            "--dirstat", "--summary", "--name-only", "--name-status", "--patch",
            "-p", "-u", "--no-patch", "-s", "--raw", "--unified", "-U",
            "--ignore-all-space", "-w", "--ignore-space-change", "-b",
            "--ignore-space-at-eol", "--ignore-blank-lines", "--ignore-cr-at-eol",
            "--word-diff", "--word-diff-regex", "--color-words", "--find-renames",
            "-M", "--find-copies", "-C", "--no-renames", "--diff-filter",
            "--find-object", "-S", "-G", "--pickaxe-regex", "--pickaxe-all",
            "-R", "--no-prefix", "--src-prefix", "--dst-prefix", "--check",
            "--exit-code", "--quiet", "--minimal", "--patience", "--histogram",
            "--anchored", "--diff-algorithm", "--full-index", "--binary",
            "--abbrev", "--relative", "--text", "-a", "--function-context", "-W",
            "-l", "--merge-base", "--compact-summary",
        }
    ),
    "log": frozenset(
        {
            "--oneline", "--decorate", "--no-decorate", "--graph", "--stat",
            "--shortstat", "--numstat", "--name-only", "--name-status",
            "--abbrev-commit", "--no-abbrev-commit", "--author", "--committer",
            "--grep", "--all-match", "--invert-grep", "-i", "--regexp-ignore-case",
            "--since", "--after", "--until", "--before", "--max-count", "-n",
            "--skip", "--reverse", "--all", "--branches", "--tags", "--remotes",
            "--merges", "--no-merges", "--first-parent", "--follow", "--format",
            "--pretty", "--date", "--patch", "-p", "--no-patch", "--topo-order",
            "--date-order", "--relative-date", "--simplify-by-decoration",
        }
    ),
    "branch": frozenset(
        {
            "--list", "-l", "--all", "-a", "--remotes", "-r", "--verbose", "-v",
            "-vv", "--show-current", "--contains", "--no-contains", "--merged",
            "--no-merged", "--sort", "--format", "--points-at", "--column",
            "--no-column", "--track", "--no-track",
            # Safe (refuses when it would lose work) variants only.  Their force
            # counterparts -D, -M and -C are absent, so they are rejected here as
            # well as by _BLOCKED_BRANCH_FLAGS below.
            "-d", "-m", "-c",
        }
    ),
}

# `git log -5` and friends: a bare count is not a flag anyone needs listed.
_NUMERIC_SHORTHAND_RE = re.compile(r"^-\d+$")


def _reject_disallowed_flags(operation: str, extra_args: list[str]) -> str | None:
    """Return an error message if *extra_args* holds a flag this operation
    may not use, else None."""
    allowed = _ALLOWED_READ_FLAGS.get(operation)
    if allowed is None:
        return None
    for arg in extra_args:
        if arg == "--":
            break  # everything after this is a pathspec
        if not arg.startswith("-"):
            continue
        if _NUMERIC_SHORTHAND_RE.match(arg):
            continue
        # --unified=3 and -U3 both carry their value inline.
        name = arg.split("=", 1)[0]
        if name in allowed:
            continue
        if any(name.startswith(f) and f.startswith("-") and len(f) == 2 for f in allowed):
            continue  # short flag with an attached value, e.g. -U3, -M50
        return (
            f"Flag {arg!r} is not allowed for git {operation} via git_tool, which runs "
            "without confirmation. Use execute_shell if you need it — that goes through "
            "the safety classifier."
        )
    return None


# Flags blocked in git commit — use execute_shell for these (goes through safety classifier)
_BLOCKED_COMMIT_FLAGS = {"--amend", "--reset-author", "--allow-empty-message"}
# Compared against the flag name with any "=value" stripped, so that the space
# form (--author "X") is caught as well as --author=X.
_BLOCKED_COMMIT_NAMES = {"--author", "--date"}

# Flags blocked in git branch — use execute_shell for destructive branch ops
_BLOCKED_BRANCH_FLAGS = {"-D", "-M", "--force", "--delete"}
_BLOCKED_BRANCH_PREFIXES = ("--force",)

# Subcommands blocked in git stash — use execute_shell for destructive stash ops
_BLOCKED_STASH_FLAGS = {"drop", "clear"}


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

    # The read-only operations are classified SAFE and run with no dialog, so
    # their flags are allowlisted rather than denylisted.
    flag_error = _reject_disallowed_flags(operation, extra_args)
    if flag_error is not None:
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
                if arg in _BLOCKED_COMMIT_FLAGS or arg.split("=", 1)[0] in _BLOCKED_COMMIT_NAMES:
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
