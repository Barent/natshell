"""Rich markup escaping for untrusted text."""

from __future__ import annotations


def escape_markup(text: str) -> str:
    """Escape Rich markup characters in untrusted text."""
    return text.replace("[", "\\[")
