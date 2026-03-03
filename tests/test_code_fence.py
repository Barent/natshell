"""Tests for the code fence parser."""

from __future__ import annotations

from natshell.ui.code_fence import (
    CodeSegment,
    TextSegment,
    _normalize_language,
    parse_code_fences,
)


class TestNormalizeLanguage:
    def test_known_alias_yml(self):
        assert _normalize_language("yml") == "yaml"

    def test_known_alias_shell(self):
        assert _normalize_language("shell") == "bash"

    def test_known_alias_sh(self):
        assert _normalize_language("sh") == "bash"

    def test_known_alias_console(self):
        assert _normalize_language("console") == "bash"

    def test_known_alias_js(self):
        assert _normalize_language("js") == "javascript"

    def test_known_alias_py(self):
        assert _normalize_language("py") == "python"

    def test_empty_string(self):
        assert _normalize_language("") == "text"

    def test_unknown_passthrough(self):
        assert _normalize_language("rust") == "rust"

    def test_case_insensitive(self):
        assert _normalize_language("YML") == "yaml"
        assert _normalize_language("Python") == "python"


class TestParseCodeFences:
    def test_no_fences_returns_single_text_segment(self):
        result = parse_code_fences("Hello world")
        assert result == [TextSegment("Hello world")]

    def test_single_fence_with_language(self):
        text = 'Here is code:\n```python\nprint("hi")\n```\nDone.'
        result = parse_code_fences(text)
        assert len(result) == 3
        assert isinstance(result[0], TextSegment)
        assert result[0].text == "Here is code:\n"
        assert isinstance(result[1], CodeSegment)
        assert result[1].language == "python"
        assert result[1].code == 'print("hi")'
        assert isinstance(result[2], TextSegment)
        assert result[2].text == "\nDone."

    def test_single_fence_without_language(self):
        text = "```\nhello\n```"
        result = parse_code_fences(text)
        assert len(result) == 1
        assert isinstance(result[0], CodeSegment)
        assert result[0].language == "text"
        assert result[0].code == "hello"

    def test_multiple_fences(self):
        text = (
            "First:\n```python\na = 1\n```\n"
            "Second:\n```bash\necho hi\n```\n"
        )
        result = parse_code_fences(text)
        assert len(result) == 4
        assert isinstance(result[0], TextSegment)
        assert isinstance(result[1], CodeSegment)
        assert result[1].language == "python"
        assert isinstance(result[2], TextSegment)
        assert isinstance(result[3], CodeSegment)
        assert result[3].language == "bash"

    def test_adjacent_fences(self):
        text = "```python\na = 1\n```\n```bash\necho hi\n```\n"
        result = parse_code_fences(text)
        code_segments = [s for s in result if isinstance(s, CodeSegment)]
        assert len(code_segments) == 2

    def test_fence_at_start(self):
        text = "```python\ncode\n```\nTrailing text."
        result = parse_code_fences(text)
        assert isinstance(result[0], CodeSegment)
        assert result[0].code == "code"

    def test_fence_at_end(self):
        text = "Leading text.\n```python\ncode\n```"
        result = parse_code_fences(text)
        assert isinstance(result[-1], CodeSegment)

    def test_language_aliases(self):
        for alias, expected in [("yml", "yaml"), ("shell", "bash"), ("console", "bash")]:
            text = f"```{alias}\nstuff\n```\n"
            result = parse_code_fences(text)
            code = [s for s in result if isinstance(s, CodeSegment)]
            assert len(code) == 1
            assert code[0].language == expected

    def test_unclosed_fence_treated_as_plain_text(self):
        text = "Hello\n```python\nprint('hi')\nno closing"
        result = parse_code_fences(text)
        assert len(result) == 1
        assert isinstance(result[0], TextSegment)
        assert result[0].text == text

    def test_empty_code_block(self):
        text = "```python\n\n```\n"
        result = parse_code_fences(text)
        code = [s for s in result if isinstance(s, CodeSegment)]
        assert len(code) == 1
        assert code[0].code == ""

    def test_inline_backticks_not_confused(self):
        text = "Use `code` and ``double`` backticks."
        result = parse_code_fences(text)
        assert len(result) == 1
        assert isinstance(result[0], TextSegment)

    def test_empty_string(self):
        result = parse_code_fences("")
        assert result == [TextSegment("")]

    def test_multiline_code(self):
        text = "```python\ndef foo():\n    return 42\n```\n"
        result = parse_code_fences(text)
        code = [s for s in result if isinstance(s, CodeSegment)]
        assert len(code) == 1
        assert "def foo():" in code[0].code
        assert "return 42" in code[0].code
