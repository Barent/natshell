"""Working memory — persistent scratchpad in agents.md files."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Markers that indicate a project root directory
_PROJECT_MARKERS = (
    ".git",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "CMakeLists.txt",
)


@dataclass
class WorkingMemory:
    """Loaded working memory content."""

    content: str  # Raw markdown, truncated to max_chars
    source: Path  # File path it was loaded from
    is_project_local: bool  # True if from {project}/.natshell/agents.md


def find_project_root(start: Path) -> Path | None:
    """Walk up from *start* looking for a project root marker.

    Returns the first directory containing any of the standard markers
    (.git, pyproject.toml, package.json, Cargo.toml, go.mod, Makefile,
    CMakeLists.txt), or None if no marker is found before reaching /.
    """
    current = start.resolve()
    while True:
        for marker in _PROJECT_MARKERS:
            if (current / marker).exists():
                return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def find_memory_file(cwd: Path) -> Path | None:
    """Locate the agents.md file to load.

    Search order:
      1. {project_root}/.natshell/agents.md  (project-local)
      2. ~/.config/natshell/agents.md         (global fallback)

    Returns None if neither file exists.
    """
    root = find_project_root(cwd)
    if root is not None:
        local = root / ".natshell" / "agents.md"
        if local.is_file():
            return local

    global_path = Path.home() / ".config" / "natshell" / "agents.md"
    if global_path.is_file():
        return global_path

    return None


def load_working_memory(
    cwd: Path, max_chars: int = 4000
) -> WorkingMemory | None:
    """Load and truncate the working memory file.

    Returns None if no memory file is found.
    """
    path = find_memory_file(cwd)
    if path is None:
        return None

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to read working memory %s: %s", path, exc)
        return None

    if not content.strip():
        return None

    # Determine if project-local
    root = find_project_root(cwd)
    is_local = root is not None and path == root / ".natshell" / "agents.md"

    # Truncate to budget
    if len(content) > max_chars:
        content = content[:max_chars] + "\n... (truncated)"

    return WorkingMemory(content=content, source=path, is_project_local=is_local)


def memory_file_path(cwd: Path) -> Path:
    """Return the path the agent should write to.

    Prefers {project_root}/.natshell/agents.md when a project root is found,
    otherwise falls back to ~/.config/natshell/agents.md.
    """
    root = find_project_root(cwd)
    if root is not None:
        return root / ".natshell" / "agents.md"
    return Path.home() / ".config" / "natshell" / "agents.md"


def effective_memory_chars(n_ctx: int, base_chars: int = 4000) -> int:
    """Scale memory character budget with context window size.

    For large context windows, the memory scratchpad can be bigger.
    *base_chars* is used as-is below the 32K threshold, so the user's
    configured ``max_chars`` still acts as a floor for small models.
    """
    if n_ctx >= 524288:
        return 32000
    elif n_ctx >= 262144:
        return 24000
    elif n_ctx >= 131072:
        return 16000
    elif n_ctx >= 65536:
        return 12000
    elif n_ctx >= 32768:
        return 8000
    return base_chars


def should_inject_memory(n_ctx: int, min_ctx: int = 16384) -> bool:
    """Return True when the context window is large enough for memory injection."""
    return n_ctx >= min_ctx
