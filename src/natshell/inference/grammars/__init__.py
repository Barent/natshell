"""Tool-call grammar registry.

Each family module (``qwen`` / ``mistral`` / ``gemma``) exposes a singleton
``GRAMMAR`` instance implementing the :class:`grammars.common.Grammar`
protocol — the family's native wire format.

:func:`get_grammar` maps a model family name (as returned by
``LocalEngine.model_family`` / ``_detect_model_family``) to its grammar.
Unknown families fall back to the Qwen wire format, which is what every
non-Qwen/Mistral/Gemma model was already served.

``ALL_FAMILY_STRIP`` is the union of every family's marker regexes (think
blocks, special tokens, native tool-call syntax).  The original
``local.py`` stripped this exact union from *all* model output — so a Qwen
response with stray Gemma tokens, or a Gemma response with a stray Qwen tag,
was cleaned regardless of which family owned the model.  Preserve that by
passing it as the default strip set.
"""
from __future__ import annotations

from natshell.inference.grammars import common, gemma, mistral, qwen
from natshell.inference.grammars.common import Grammar

__all__ = [
    "ALL_FAMILY_STRIP",
    "GRAMMARS",
    "Grammar",
    "get_grammar",
    "common",
    "gemma",
    "mistral",
    "qwen",
]

GRAMMARS: dict[str, Grammar] = {
    "qwen": qwen.GRAMMAR,
    "mistral": mistral.GRAMMAR,
    "gemma": gemma.GRAMMAR,
}

# Union of every family's marker patterns.  THINK / THINK_UNCLOSED are applied
# by strip_prose_markers() itself, so they are listed here only for reference
# and NOT included in the family strip set (to avoid a double-application).
ALL_FAMILY_STRIP: tuple = (
    gemma.GEMMA_THINK_RE,
    gemma.GEMMA_THINK_UNCLOSED_RE,
    gemma.GEMMA_SPECIAL_TOKEN_RE,
    qwen.TOOL_CALL_RE,
    gemma.GEMMA_TOOL_CALL_RE,
    mistral.MISTRAL_TOOL_CALLS_RE,
)


def get_grammar(family: str | None) -> Grammar:
    """Resolve a model family name to its Grammar (Qwen as fallback)."""
    if not family:
        return GRAMMARS["qwen"]
    return GRAMMARS.get(family, GRAMMARS["qwen"])
