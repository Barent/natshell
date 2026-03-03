"""Tests for the syntax render module."""

from __future__ import annotations

from rich.console import Group
from rich.syntax import Syntax
from rich.text import Text

from natshell.ui.code_fence import CodeSegment, TextSegment
from natshell.ui.syntax_render import render_segments


class TestRenderSegments:
    def test_plain_text_returns_text_object(self):
        segments = [TextSegment("Hello world")]
        result = render_segments(segments)
        assert isinstance(result, Text)

    def test_plain_text_with_prefix(self):
        segments = [TextSegment("Hello")]
        result = render_segments(segments, prefix_markup="[bold]Prefix:[/] ")
        assert isinstance(result, Text)
        assert "Prefix" in result.plain

    def test_plain_text_with_suffix(self):
        segments = [TextSegment("Hello")]
        result = render_segments(segments, suffix_markup="\n[dim]extra[/]")
        assert isinstance(result, Text)

    def test_mixed_segments_returns_group(self):
        segments = [
            TextSegment("Before\n"),
            CodeSegment(code="x = 1", language="python"),
            TextSegment("\nAfter"),
        ]
        result = render_segments(segments)
        assert isinstance(result, Group)

    def test_group_contains_syntax(self):
        segments = [
            TextSegment("Before\n"),
            CodeSegment(code="x = 1", language="python"),
        ]
        result = render_segments(segments)
        assert isinstance(result, Group)
        renderables = list(result.renderables)
        assert any(isinstance(r, Syntax) for r in renderables)

    def test_text_style_applied(self):
        segments = [TextSegment("thinking...")]
        result = render_segments(segments, text_style="dim italic")
        assert isinstance(result, Text)
        assert result.style == "dim italic"

    def test_empty_language_uses_text_lexer(self):
        segments = [CodeSegment(code="raw stuff", language="text")]
        result = render_segments(segments)
        assert isinstance(result, Group)
        renderables = list(result.renderables)
        syntax_parts = [r for r in renderables if isinstance(r, Syntax)]
        assert len(syntax_parts) >= 1

    def test_prefix_on_code_only(self):
        """When the first segment is code, prefix is added as a separate Text."""
        segments = [CodeSegment(code="x = 1", language="python")]
        result = render_segments(segments, prefix_markup="[bold]Label:[/] ")
        assert isinstance(result, Group)
        renderables = list(result.renderables)
        # First element should be the prefix Text
        assert isinstance(renderables[0], Text)
        assert "Label" in renderables[0].plain

    def test_suffix_on_code_only(self):
        """When the last segment is code, suffix is added as a separate Text."""
        segments = [CodeSegment(code="x = 1", language="python")]
        result = render_segments(segments, suffix_markup="\n[dim]stats[/]")
        assert isinstance(result, Group)
        renderables = list(result.renderables)
        assert isinstance(renderables[-1], Text)

    def test_escapes_markup_in_text(self):
        """Brackets in text are escaped so they don't break Rich markup."""
        segments = [TextSegment("list[int]")]
        result = render_segments(segments)
        assert isinstance(result, Text)
        # The plain text should contain the original content
        assert "list" in result.plain
