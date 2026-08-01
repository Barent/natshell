"""Shared command tokenizer for the safety classifier and execute_shell.

Both modules used to carry their own regex for finding sub-command boundaries,
and they disagreed about which characters were separators.  Two consequences:

1. A newline was not a separator in the old classifier regex, so only the first
   line of a multi-line command was ever classified while ``bash -c`` ran all
   of them.  A blocked command placed on the second line classified SAFE.

2. The left-paren was a separator in ``execute_shell`` but not for the
   classifier, so sudo wrapped in a subshell classified SAFE and then had
   cached root password piped into it.

Both now call ``split_commands()`` from this module.  The splitter is
quote-aware so an operator inside quotes stays data — which also stops a
semicolon inside a quoted argument from leaking the sudo password onto a
pipeline's stdin.
"""

from __future__ import annotations


def split_commands(command: str) -> list[str]:
    """Return non-empty sub-commands split on shell operators and newlines.

    Splits on ``&&``, ``||``, ``;``, ``&``, ``|``, ``(``, ``)``, and newline.
    Operators inside single or double quotes are ignored  (quote-aware).
    Consecutive delimiters produce at most one empty token between them, so
    the caller sees sub-commands rather than delimiter noise.

    Examples:
        >>> split_commands("echo hello && ls")
        ['echo hello', 'ls']
        >>> split_commands('grep "a;b" test')
        ["grep \\"a;b\\" test"]
    """
    result: list[str] = []
    current: list[str] = []
    i = 0
    length = len(command)

    while i < length:
        char = command[i]
        if char in ("'", '"'):
            # Consume the entire quoted string verbatim.
            quote = char
            tokens = [char]
            i += 1
            while i < length and command[i] != quote:
                if command[i] == "\\":
                    tokens.append(command[i])
                    i += 1
                    if i < length:
                        tokens.append(command[i])
                        i += 1
                else:
                    tokens.append(command[i])
                    i += 1
            if i < length:
                tokens.append(command[i])  # closing quote
                i += 1
            current.extend(tokens)
            continue

        # Shell operator or newline — start new sub-command
        if char in (
            "&",
            "|",
            ";",
            "(",
            ")",
            "\n",
        ) and not current and result and result[-1]:
            # Don't split on a bare delimiter before any content has been
            # collected for this token; swallow it.  This avoids producing
            # spurious empty tokens at the start of chains like "&& ls".
            i += 1
            continue

        if char in (
            "&",
            "|",
            ";",
            "(",
            ")",
        ) or char == "\n":
            # Handle two-char operators first (&&, ||)
            if char in (
                "&",
                "|",
            ) and i + 1 < length and command[i + 1] == char:
                i += 2
            else:
                i += 1

            if current or result:
                token = "".join(current).strip()
                if token:
                    result.append(token)
                current = []
            continue

        # Regular character — accumulate into current token
        current.append(char)
        i += 1

    # Flush remaining content
    token = "".join(current).strip()
    if token:
        result.append(token)

    return result


def split_with_delimiters(command: str) -> list[str]:
    """Return an alternating list of tokens and delimiters.

    Like ``split_commands()`` but keeps the delimiter strings between sub-commands,
    so the caller can reconstruct the original command after per-sub-command edits.

    Example:

        >>> split_with_delimiters("A && B || C")  # noqa: E501
        ['A', ' && ', 'B', ' || ', 'C']

    The even entries (0, 2, 4…) are sub-command tokens; the odd entries are
    delimiter fragments. Leading and trailing whitespace is included in each
    delimiter so that ``"".join(tokens)`` reconstructs the original command.
    """
    result: list[str] = []
    current: list[str] = []
    i = 0
    length = len(command)

    while i < length:
        char = command[i]
        if char in ("'", '"'):
            quote = char
            tokens = [char]
            i += 1
            while i < length and command[i] != quote:
                if command[i] == "\\":
                    tokens.append(command[i])
                    i += 1
                    if i < length:
                        tokens.append(command[i])
                        i += 1
                else:
                    tokens.append(command[i])
                    i += 1
            if i < length:
                tokens.append(command[i])
                i += 1
            current.extend(tokens)
            continue

        if char in (
            "&",
            "|",
            ";",
            "(",
            ")",
        ) or char == "\n":
            # Capture leading whitespace before operator into the delimiter.
            ws_start = i
            while ws_start > 0 and command[ws_start - 1] == " ":
                ws_start -= 1

            # Handle two-char operators
            if char in (
                "&",
                "|",
            ) and i + 1 < length and command[i + 1] == char:
                op_end = i + 2
            else:
                op_end = i + 1

            # Also capture trailing whitespace after operator so that joining
            # reconstructed tokens yields the original spacing.
            while op_end < length and command[op_end] == " ":
                op_end += 1

            delimiter = command[ws_start:op_end]

            token = "".join(current).strip()
            if token:
                result.append(token)
            result.append(delimiter)
            current = []
            i = op_end
            continue

        # Trailing whitespace at start (no prior content) — skip it.
        if char == " " and not current and (not result or result[-1] == ""):
            i += 1
            continue

        current.append(char)
        i += 1

    token = "".join(current).strip()
    if token:
        result.append(token)
    elif result and result[-1].endswith(" "):
        # Trailing whitespace only — drop it.
        pass

    return result if result else []
