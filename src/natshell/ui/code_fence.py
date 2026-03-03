"""Parse markdown-style fenced code blocks into typed segments."""

from __future__ import annotations

import re
from dataclasses import dataclass

_FENCE_RE = re.compile(
    r"^```(\w*)\s*\n(.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)

_LANG_ALIASES: dict[str, str] = {
    "yml": "yaml",
    "shell": "bash",
    "sh": "bash",
    "console": "bash",
    "zsh": "bash",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "rb": "ruby",
    "rs": "rust",
    "cpp": "c++",
    "cxx": "c++",
    "": "text",
}


@dataclass(frozen=True, slots=True)
class TextSegment:
    """Plain prose text."""
    text: str


@dataclass(frozen=True, slots=True)
class CodeSegment:
    """Fenced code block."""
    code: str
    language: str


Segment = TextSegment | CodeSegment


def _normalize_language(lang: str) -> str:
    """Normalize a fence language tag to a Pygments lexer name."""
    lower = lang.lower()
    return _LANG_ALIASES.get(lower, lower)


def parse_code_fences(text: str) -> list[Segment]:
    """Split *text* into interleaved TextSegment / CodeSegment runs.

    Fast path: if no fences are found, returns ``[TextSegment(text)]``.
    Unclosed fences are treated as plain text (the regex simply won't match).
    """
    segments: list[Segment] = []
    last_end = 0

    for m in _FENCE_RE.finditer(text):
        # Text before this fence
        before = text[last_end:m.start()]
        if before:
            segments.append(TextSegment(before))

        lang = _normalize_language(m.group(1))
        code = m.group(2)
        # Strip one trailing newline from code (the fence close is on its own line)
        if code.endswith("\n"):
            code = code[:-1]
        segments.append(CodeSegment(code=code, language=lang))
        last_end = m.end()

    # Trailing text after the last fence (or the whole string if no fences)
    trailing = text[last_end:]
    if trailing:
        segments.append(TextSegment(trailing))

    # Fast path — no fences found at all
    if not segments:
        return [TextSegment(text)]

    return segments
