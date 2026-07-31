"""Tests for the shared quote-aware command splitter."""

from __future__ import annotations

import pytest

from natshell.safety.command_split import (
    is_delimiter,
    split_commands,
    split_with_delimiters,
)


class TestSplitCommands:
    @pytest.mark.parametrize(
        "command,expected",
        [
            ("ls", ["ls"]),
            ("ls -la /tmp", ["ls -la /tmp"]),
            ("a && b", ["a", "b"]),
            ("a || b", ["a", "b"]),
            ("a ; b", ["a", "b"]),
            ("a | b", ["a", "b"]),
            ("a & b", ["a", "b"]),
            ("a && b || c ; d | e", ["a", "b", "c", "d", "e"]),
        ],
    )
    def test_operators(self, command, expected):
        assert split_commands(command) == expected

    @pytest.mark.parametrize(
        "command,expected",
        [
            ("cd /tmp\nrm -rf /", ["cd /tmp", "rm -rf /"]),
            ("echo hi\r\nrm -rf ~", ["echo hi", "rm -rf ~"]),
            ("a\nb\nc", ["a", "b", "c"]),
        ],
    )
    def test_newline_is_a_separator(self, command, expected):
        """A newline separates commands exactly as ';' does — bash -c runs both."""
        assert split_commands(command) == expected

    @pytest.mark.parametrize(
        "command,expected",
        [
            ("(sudo rm -rf ~)", ["sudo rm -rf ~"]),
            ("(a; b)", ["a", "b"]),
            ("nohup rm -rf ~ &", ["nohup rm -rf ~"]),
        ],
    )
    def test_subshell_parens_separate(self, command, expected):
        assert split_commands(command) == expected

    @pytest.mark.parametrize(
        "command",
        [
            'echo "a && b"',
            "echo 'a && b'",
            'echo "a; b"',
            'echo "a | b"',
            'echo "use sudo"',
        ],
    )
    def test_operators_inside_quotes_are_data(self, command):
        assert split_commands(command) == [command]

    def test_escaped_operator_outside_quotes(self):
        assert split_commands(r"echo a\;b") == [r"echo a\;b"]

    def test_backslash_is_literal_inside_single_quotes(self):
        """In sh, '\\' inside '...' is a literal backslash and does not escape."""
        assert split_commands(r"echo 'a\' ; rm -rf ~") == [r"echo 'a\'", "rm -rf ~"]

    def test_empty_segments_dropped(self):
        assert split_commands("a ;; b") == ["a", "b"]
        assert split_commands("") == []
        assert split_commands("   ") == []

    def test_windows_paths_survive(self):
        """Backslash path separators must not be mangled or split."""
        assert split_commands(r"dir C:\Users") == [r"dir C:\Users"]
        assert split_commands(r"Remove-Item -Recurse -Force C:\\") == [
            r"Remove-Item -Recurse -Force C:\\"
        ]


class TestSplitWithDelimiters:
    @pytest.mark.parametrize(
        "command",
        [
            "a && b",
            "a; b | c",
            "(sudo rm -rf ~)",
            'echo "a && b" ; ls',
            "cd /tmp\nrm -rf /",
            r"dir C:\Users",
            "",
        ],
    )
    def test_parts_rejoin_to_the_original(self, command):
        """execute_shell rewrites parts in place, so this must be lossless."""
        assert "".join(split_with_delimiters(command)) == command

    def test_delimiters_are_reported_as_such(self):
        parts = split_with_delimiters("a && b")
        assert [p for p in parts if is_delimiter(p)] == ["&&"]

    def test_quoted_operator_is_not_a_delimiter(self):
        parts = split_with_delimiters('echo "a && b"')
        assert not any(is_delimiter(p) for p in parts)
