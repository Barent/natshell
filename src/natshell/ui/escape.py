"""Rich markup escaping for untrusted text."""

from __future__ import annotations

from rich.markup import escape as _rich_escape


def escape_markup(text: str) -> str:
    """Escape Rich markup characters in untrusted text.

    Delegates to rich.markup.escape rather than replacing "[" with "\\[".  That
    replacement left an already-present backslash alone, so `\\[red]` became
    `\\\\[red]`, which Rich reads as one literal backslash followed by a live
    `[red]` span.  Model output could therefore style the confirmation dialog
    it was being approved in — including foreground on matching background,
    which hides the text the user is agreeing to.
    """
    return _rich_escape(text)
