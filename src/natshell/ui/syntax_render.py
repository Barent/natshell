"""Convert parsed code-fence segments into Rich renderables."""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.syntax import Syntax
from rich.text import Text

from natshell.ui.code_fence import CodeSegment, Segment, TextSegment
from natshell.ui.escape import escape_markup


def render_segments(
    segments: list[Segment],
    prefix_markup: str = "",
    suffix_markup: str = "",
    text_style: str = "",
) -> RenderableType:
    """Turn a list of segments into a single Rich renderable.

    Parameters
    ----------
    segments:
        Output of :func:`~natshell.ui.code_fence.parse_code_fences`.
    prefix_markup:
        Rich markup prepended to the first text segment (e.g. ``"[bold green]NatShell:[/] "``).
    suffix_markup:
        Rich markup appended after the last text segment (e.g. a metrics line).
    text_style:
        Base Rich style applied to every ``TextSegment`` (e.g. ``"dim italic"``).

    Returns
    -------
    A ``Text`` object (fast path for pure prose) or a ``Group`` of mixed renderables.
    """
    # Fast path: single text segment → same behaviour as before (plain Text)
    if len(segments) == 1 and isinstance(segments[0], TextSegment):
        markup = prefix_markup + escape_markup(segments[0].text) + suffix_markup
        return Text.from_markup(markup, style=text_style)

    parts: list[RenderableType] = []
    for i, seg in enumerate(segments):
        if isinstance(seg, TextSegment):
            escaped = escape_markup(seg.text)
            if i == 0:
                escaped = prefix_markup + escaped
            if i == len(segments) - 1:
                escaped += suffix_markup
            parts.append(Text.from_markup(escaped, style=text_style))
        else:
            assert isinstance(seg, CodeSegment)
            parts.append(
                Syntax(
                    seg.code,
                    seg.language,
                    theme="monokai",
                    background_color="#0d1b2a",
                    word_wrap=True,
                )
            )

    # If prefix/suffix need attaching but all segments were code, add empty Text wrappers
    if parts and isinstance(segments[0], CodeSegment) and prefix_markup:
        parts.insert(0, Text.from_markup(prefix_markup, style=text_style))
    if parts and isinstance(segments[-1], CodeSegment) and suffix_markup:
        parts.append(Text.from_markup(suffix_markup, style=text_style))

    return Group(*parts)
