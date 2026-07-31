"""Writes must not travel through a symlink, and agents.md is not special.

Two halves of one chain.  write_file resolved its path, which follows symlinks,
while BackupManager.backup() refused symlinks and returned None — a value the
caller discarded.  And the classifier returned SAFE for any path ending in
".natshell/agents.md", checked before the sensitive-path loop and before the
mode checks, on the string the model supplied.

So creating a link at .natshell/agents.md and writing to it overwrote the link's
target with no confirmation and no backup.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from natshell.config import SafetyConfig
from natshell.safety.classifier import Risk, SafetyClassifier
from natshell.tools.edit_file import edit_file
from natshell.tools.write_file import write_file

requires_symlink = pytest.mark.skipif(
    os.name == "nt" and not os.environ.get("NATSHELL_TEST_SYMLINKS"),
    reason="creating symlinks on Windows needs privilege; set NATSHELL_TEST_SYMLINKS to run",
)


class TestAgentsMdIsNotAutoApproved:
    """agents.md is loaded from cwd every run and spliced into the system
    prompt, so an unconfirmed write to it is a persistent instruction change."""

    @pytest.mark.parametrize(
        "path",
        [
            ".natshell/agents.md",
            "/home/user/project/.natshell/agents.md",
            "~/.config/natshell/agents.md",
            "/tmp/anywhere-i-like/.natshell/agents.md",
        ],
    )
    def test_write_confirms(self, path):
        c = SafetyClassifier(SafetyConfig(mode="confirm"))
        assert c.classify_tool_call("write_file", {"path": path, "content": "x"}) == Risk.CONFIRM

    @pytest.mark.parametrize("path", [".natshell/agents.md", "~/.config/natshell/agents.md"])
    def test_edit_confirms(self, path):
        c = SafetyClassifier(SafetyConfig(mode="confirm"))
        risk = c.classify_tool_call(
            "edit_file", {"path": path, "old_text": "a", "new_text": "b"}
        )
        assert risk == Risk.CONFIRM


class TestSymlinkWrites:
    @requires_symlink
    async def test_write_file_refuses_a_symlink(self, tmp_path: Path):
        real = tmp_path / "bashrc"
        real.write_text("original contents\n")
        link = tmp_path / "agents.md"
        link.symlink_to(real)

        result = await write_file(str(link), "PWNED")

        assert result.exit_code == 1
        assert "symlink" in result.error.lower()
        assert real.read_text() == "original contents\n"

    @requires_symlink
    async def test_append_refuses_a_symlink(self, tmp_path: Path):
        real = tmp_path / "bashrc"
        real.write_text("original\n")
        link = tmp_path / "agents.md"
        link.symlink_to(real)

        result = await write_file(str(link), "PWNED", mode="append")

        assert result.exit_code == 1
        assert real.read_text() == "original\n"

    @requires_symlink
    async def test_edit_file_refuses_a_symlink(self, tmp_path: Path):
        real = tmp_path / "bashrc"
        real.write_text("original\n")
        link = tmp_path / "agents.md"
        link.symlink_to(real)

        result = await edit_file(str(link), "original", "PWNED")

        assert result.exit_code == 1
        assert real.read_text() == "original\n"

    async def test_ordinary_write_still_works(self, tmp_path: Path):
        target = tmp_path / "notes.txt"
        result = await write_file(str(target), "hello")
        assert result.exit_code == 0
        assert target.read_text() == "hello"

    async def test_overwrite_still_works(self, tmp_path: Path):
        target = tmp_path / "notes.txt"
        target.write_text("before")
        result = await write_file(str(target), "after")
        assert result.exit_code == 0
        assert target.read_text() == "after"

    async def test_refuses_a_directory(self, tmp_path: Path):
        d = tmp_path / "adir"
        d.mkdir()
        result = await write_file(str(d), "x")
        assert result.exit_code == 1
        assert "not a file" in result.error.lower()


class TestResolveWriteTarget:
    def test_expands_user(self, tmp_path: Path, monkeypatch):
        from natshell.tools.safe_path import resolve_write_target

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        target, error = resolve_write_target("~/notes.txt")
        assert error == ""
        assert target == (tmp_path / "notes.txt").resolve()

    def test_new_file_is_allowed(self, tmp_path: Path):
        from natshell.tools.safe_path import resolve_write_target

        target, error = resolve_write_target(str(tmp_path / "does-not-exist.txt"))
        assert error == ""
        assert target is not None
