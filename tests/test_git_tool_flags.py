"""git_tool's read-only operations must actually be read-only.

status, diff, log and branch are classified SAFE, so they run with no dialog,
and shlex.split(args) was appended straight to the git command line.
`git diff --output=<path>` is therefore an unconfirmed arbitrary file write
whose contents the model influences.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from natshell.tools.git_tool import git_tool


@pytest.fixture
def repo(tmp_path: Path, monkeypatch):
    """A real git repository with one commit, as the cwd."""
    import subprocess

    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.st"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("one\n")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("two\n")
    return tmp_path


class TestReadOnlyOperationsCannotWrite:
    @pytest.mark.parametrize(
        "operation,args",
        [
            ("diff", "--output=PWNED"),
            ("diff", "--output PWNED"),
            ("log", "--output=PWNED"),
            ("status", "--output=PWNED"),
        ],
    )
    async def test_output_flag_is_refused(self, repo: Path, operation, args):
        result = await git_tool(operation, args)

        assert result.exit_code == 1
        assert not (repo / "PWNED").exists(), "git wrote a file from a SAFE operation"

    @pytest.mark.parametrize(
        "args",
        ["--ext-diff", "-Oorderfile", "--textconv"],
    )
    async def test_other_escape_hatches_refused(self, repo: Path, args):
        result = await git_tool("diff", args)
        assert result.exit_code == 1
        assert "not allowed" in result.error.lower()

    async def test_error_names_the_flag_and_the_alternative(self, repo: Path):
        result = await git_tool("diff", "--output=PWNED")
        assert "--output" in result.error
        assert "execute_shell" in result.error


class TestOrdinaryUsageStillWorks:
    @pytest.mark.parametrize(
        "operation,args",
        [
            ("status", ""),
            ("status", "--short"),
            ("status", "-s -b"),
            ("diff", ""),
            ("diff", "--stat"),
            ("diff", "--staged"),
            ("diff", "--name-only"),
            ("diff", "-w"),
            ("diff", "HEAD"),
            ("diff", "a.txt"),
            ("diff", "-- a.txt"),
            ("log", ""),
            ("log", "-5"),
            ("log", "--oneline"),
            ("log", "--stat"),
            ("log", "--author=Test"),
            ("log", "--pretty=oneline"),
            ("branch", ""),
            ("branch", "--list"),
            ("branch", "-a"),
        ],
    )
    async def test_allowed(self, repo: Path, operation, args):
        result = await git_tool(operation, args)
        assert result.exit_code == 0, f"{operation} {args!r}: {result.error}"

    async def test_paths_and_revisions_are_not_flags(self, repo: Path):
        """Only flags are allowlisted; a path or revision cannot write a file."""
        result = await git_tool("diff", "HEAD -- a.txt")
        assert result.exit_code == 0

    async def test_branch_creation_still_works(self, repo: Path):
        result = await git_tool("branch", "feature-x")
        assert result.exit_code == 0

    async def test_destructive_branch_flag_still_blocked(self, repo: Path):
        result = await git_tool("branch", "-D main")
        assert result.exit_code == 1


class TestCommitAuthorSpoofing:
    async def test_author_equals_form_blocked(self, repo: Path):
        result = await git_tool("commit", '--author="Evil <e@x>" -m msg')
        assert result.exit_code == 1

    async def test_author_space_form_blocked(self, repo: Path):
        """--author was checked as a prefix, so the space form split into two
        tokens and passed."""
        result = await git_tool("commit", '--author "Evil <e@x>" -m msg')
        assert result.exit_code == 1
        assert "author" in result.error.lower()

    async def test_date_space_form_blocked(self, repo: Path):
        result = await git_tool("commit", '--date "2001-01-01" -m msg')
        assert result.exit_code == 1

    async def test_ordinary_commit_still_works(self, repo: Path):
        result = await git_tool("commit", '-a -m "a message"')
        assert result.exit_code == 0, result.error
