"""Quote-aware splitting of a shell command into its sub-commands.

The safety classifier and ``execute_shell`` both need to know where one
sub-command ends and the next begins — the classifier to decide the risk of
each, ``execute_shell`` to find the ones that invoke ``sudo``.  When the two
disagree, a command can be classified as one thing and executed as another, so
both import the splitter from here rather than carrying their own regex.

Splitting is quote-aware: an operator inside ``'...'`` or ``"..."`` is data, not
a separator, so ``echo "a && b"`` is one sub-command.
"""

from __future__ import annotations

# Two-character operators must be tested before their single-character
# prefixes, otherwise "&&" tokenizes as two "&" delimiters.
_TWO_CHAR_OPERATORS = ("&&", "||")
_ONE_CHAR_OPERATORS = ";&|()\n\r"

DELIMITERS = frozenset(_TWO_CHAR_OPERATORS) | frozenset(_ONE_CHAR_OPERATORS)


def is_delimiter(part: str) -> bool:
    """Return True if *part* is a separator rather than command text."""
    return part in DELIMITERS


def split_with_delimiters(command: str) -> list[str]:
    """Split *command* on unquoted shell operators, keeping the separators.

    The returned parts concatenate back to *command* exactly, which lets a
    caller rewrite individual sub-commands and reassemble the whole without
    disturbing the user's spacing.
    """
    parts: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0
    n = len(command)

    while i < n:
        ch = command[i]

        if quote is not None:
            buf.append(ch)
            # Backslash escapes apply inside "..." but not inside '...'
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(command[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue

        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue

        if ch == "\\" and i + 1 < n:
            buf.append(ch)
            buf.append(command[i + 1])
            i += 2
            continue

        if command[i : i + 2] in _TWO_CHAR_OPERATORS:
            parts.append("".join(buf))
            buf = []
            parts.append(command[i : i + 2])
            i += 2
            continue

        if ch in _ONE_CHAR_OPERATORS:
            parts.append("".join(buf))
            buf = []
            parts.append(ch)
            i += 1
            continue

        buf.append(ch)
        i += 1

    parts.append("".join(buf))
    return parts


def split_commands(command: str) -> list[str]:
    """Return the non-empty sub-commands of *command*, separators removed."""
    return [
        stripped
        for part in split_with_delimiters(command)
        if not is_delimiter(part) and (stripped := part.strip())
    ]
