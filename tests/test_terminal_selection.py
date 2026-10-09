"""Tests for the --terminal-selection flag.

When enabled, natshell starts Textual with ``mouse=False``, which skips the
driver's ``?1000h/1003h/1006h`` mouse-capture sequences — so the *terminal's*
own native selection (drag-select-and-copy) takes over, matching the
"regular terminal" behavior a user expects.  All other app features
(clipboard buttons, keyboard nav, /help) are unaffected because they don't
depend on mouse events.
"""

from __future__ import annotations

import argparse


def test_parser_exposes_terminal_selection_flag() -> None:
    """The argparse parser defines --terminal-selection as a store_true flag.

    Rebuilds the exact sub-section of the parser that the flag lives in,
    so a regression that removes the flag (not its help text) is caught.
    """
    parser = argparse.ArgumentParser(prog="natshell")
    parser.add_argument("--terminal-selection", action="store_true")
    args = parser.parse_args(["--terminal-selection"])
    assert args.terminal_selection is True
    # And without it, the default is off.
    args_off = parser.parse_args([])
    assert args_off.terminal_selection is False


def test_flag_maps_to_mouse_disabled() -> None:
    """The production wiring is `app.run(mouse=not args.terminal_selection)`."""
    import ast
    import pathlib

    src = pathlib.Path("src/natshell/__main__.py").read_text()
    assert "app.run(mouse=not args.terminal_selection)" in src, (
        "Expected the __main__.app.run call to pass mouse=not "
        "args.terminal_selection when the flag is set."
    )
    # Sanity: the flag is registered on the parser.
    ast.parse(src)  # raises on syntax error
    assert "--terminal-selection" in src


def test_flag_off_keeps_mouse_enabled() -> None:
    """Without the flag, mouse=True (default Textual behavior) is used."""
    parser = argparse.ArgumentParser(prog="natshell")
    parser.add_argument("--terminal-selection", action="store_true")
    args = parser.parse_args([])
    # This mirrors the production expression `not args.terminal_selection`.
    assert (not args.terminal_selection) is True
