"""Untrusted text must not survive escaping as live Rich markup.

escape_markup was `text.replace("[", "\\[")`, which does not double a backslash
that is already there.  So the attacker writes `\\[style]`; the escaper turns
the `[` into `\\[` giving `\\\\[style]`; Rich reads `\\\\` as one literal
backslash followed by an unescaped `[style]` — a live style span.

That matters because it renders inside the confirmation dialog the user is
being asked to approve: a write_file preview can paint its own text in
background-on-background and hide what is really being written.
"""

from __future__ import annotations

import pytest
from rich.console import Console
from rich.text import Text

from natshell.ui.escape import escape_markup


def _spans(markup: str) -> list:
    """The style spans Rich produces when it renders *markup*."""
    return Text.from_markup(markup).spans


def _plain(markup: str) -> str:
    return Text.from_markup(markup).plain


class TestNoLiveSpansSurvive:
    @pytest.mark.parametrize(
        "hostile",
        [
            r"\[#0d1b2a on #0d1b2a]invisible\[/]",
            r"\[red]red\[/red]",
            r"\[blink]",
            r"\\[bold]still bold\\[/bold]",
            "[bold]plain[/bold]",
            r"\[link=https://evil.example]click\[/link]",
        ],
    )
    def test_escaped_text_produces_no_spans(self, hostile):
        assert _spans(escape_markup(hostile)) == []

    @pytest.mark.parametrize(
        "hostile",
        [
            r"\[red]x\[/]",
            "[red]x[/]",
            r"\\[red]x\\[/]",
            r"\\\[red]x",
        ],
    )
    def test_text_renders_verbatim(self, hostile):
        """What the user sees is what the model actually produced."""
        assert _plain(escape_markup(hostile)) == hostile

    def test_confirmation_preview_cannot_hide_itself(self):
        """The specific case: same foreground and background inside a dialog."""
        payload = r"\[#0d1b2a on #0d1b2a]rm -rf /\[/]"
        rendered = Text.from_markup(escape_markup(payload))
        assert rendered.spans == []
        assert "rm -rf /" in rendered.plain


class TestOrdinaryTextIsUnchanged:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "hello world",
            "a normal sentence with no brackets",
            "an array: items[0] and items[1]",
            "a windows path C:\\Users\\me",
            "json: {\"key\": [1, 2]}",
            "python: print(f'{x}')",
        ],
    )
    def test_round_trips(self, text):
        assert _plain(escape_markup(text)) == text

    def test_console_output_matches(self):
        """End to end through a Console, since that is how it is used."""
        console = Console(file=None, width=200, no_color=True, record=True)
        console.print(escape_markup(r"\[red]danger\[/]"))
        assert r"\[red]danger\[/]" in console.export_text()
