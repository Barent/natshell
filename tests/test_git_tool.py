"""Tests for the git_tool integration tool."""

from __future__ import annotations

import subprocess

import pytest

from natshell.config import SafetyConfig
from natshell.safety.classifier import Risk, SafetyClassifier
from natshell.tools.git_tool import (
    _CONFIRM_OPERATIONS,
    _SAFE_OPERATIONS,
    git_tool,
)


def _make_classifier(mode: str = "confirm") -> SafetyClassifier:
    """Create a classifier with minimal config for git_tool tests."""
    config = SafetyConfig(
        mode=mode,
        always_confirm=[],
        blocked=[],
    )
    return SafetyClassifier(config)


@pytest.fixture()
def git_repo(tmp_path, monkeypatch):
    """Create a temporary git repo and chdir into it."""
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    return tmp_path


@pytest.fixture()
def git_repo_with_commit(git_repo):
    """Create a temp git repo with one initial commit."""
    readme = git_repo / "README.md"
    readme.write_text("# Test\n")
    subprocess.run(
        ["git", "add", "README.md"], cwd=git_repo, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=git_repo,
        capture_output=True,
        check=True,
    )
    return git_repo


# ─── Operation sets ─────────────────────────────────────────────────────────


class TestOperationSets:
    def test_safe_operations(self):
        assert _SAFE_OPERATIONS == {"status", "diff", "log", "branch"}

    def test_confirm_operations(self):
        assert _CONFIRM_OPERATIONS == {"commit", "stash"}


# ─── Status ─────────────────────────────────────────────────────────────────


class TestGitStatus:
    async def test_clean_working_tree(self, git_repo_with_commit):
        result = await git_tool("status")
        assert result.exit_code == 0
        assert "clean" in result.output.lower()

    async def test_untracked_files(self, git_repo_with_commit):
        (git_repo_with_commit / "new_file.txt").write_text("hello\n")
        result = await git_tool("status")
        assert result.exit_code == 0
        assert "Untracked files" in result.output
        assert "new_file.txt" in result.output

    async def test_staged_changes(self, git_repo_with_commit):
        (git_repo_with_commit / "staged.txt").write_text("staged\n")
        subprocess.run(
            ["git", "add", "staged.txt"],
            cwd=git_repo_with_commit,
            capture_output=True,
            check=True,
        )
        result = await git_tool("status")
        assert result.exit_code == 0
        assert "Staged changes" in result.output
        assert "staged.txt" in result.output

    async def test_unstaged_changes(self, git_repo_with_commit):
        readme = git_repo_with_commit / "README.md"
        readme.write_text("# Modified\n")
        result = await git_tool("status")
        assert result.exit_code == 0
        assert "Unstaged changes" in result.output


# ─── Diff ───────────────────────────────────────────────────────────────────


class TestGitDiff:
    async def test_no_diff(self, git_repo_with_commit):
        result = await git_tool("diff")
        assert result.exit_code == 0
        assert "No differences" in result.output

    async def test_unstaged_diff(self, git_repo_with_commit):
        readme = git_repo_with_commit / "README.md"
        readme.write_text("# Changed\n")
        result = await git_tool("diff")
        assert result.exit_code == 0
        assert "Changed" in result.output

    async def test_staged_diff(self, git_repo_with_commit):
        (git_repo_with_commit / "new.txt").write_text("new content\n")
        subprocess.run(
            ["git", "add", "new.txt"],
            cwd=git_repo_with_commit,
            capture_output=True,
            check=True,
        )
        result = await git_tool("diff", args="--staged")
        assert result.exit_code == 0
        assert "new content" in result.output


# ─── Log ────────────────────────────────────────────────────────────────────


class TestGitLog:
    async def test_log_shows_commits(self, git_repo_with_commit):
        result = await git_tool("log")
        assert result.exit_code == 0
        assert "Initial commit" in result.output

    async def test_log_limit(self, git_repo_with_commit):
        # Make a second commit
        (git_repo_with_commit / "second.txt").write_text("second\n")
        subprocess.run(
            ["git", "add", "second.txt"],
            cwd=git_repo_with_commit,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Second commit"],
            cwd=git_repo_with_commit,
            capture_output=True,
            check=True,
        )
        result = await git_tool("log", args="-1")
        assert result.exit_code == 0
        assert "Second commit" in result.output
        assert "Initial commit" not in result.output

    async def test_log_empty_repo(self, git_repo):
        """Log on a repo with no commits should handle gracefully."""
        result = await git_tool("log")
        # git log on empty repo returns exit code 128
        assert result.exit_code != 0 or "No commits" in result.output


# ─── Branch ─────────────────────────────────────────────────────────────────


class TestGitBranch:
    async def test_list_branches(self, git_repo_with_commit):
        result = await git_tool("branch")
        assert result.exit_code == 0
        # Should show the default branch (main or master)
        assert result.output.strip()

    async def test_create_branch(self, git_repo_with_commit):
        result = await git_tool("branch", args="feature-test")
        assert result.exit_code == 0
        # Verify branch was created
        list_result = await git_tool("branch")
        assert "feature-test" in list_result.output


# ─── Commit ─────────────────────────────────────────────────────────────────


class TestGitCommit:
    async def test_commit_with_message(self, git_repo_with_commit):
        (git_repo_with_commit / "new.txt").write_text("content\n")
        subprocess.run(
            ["git", "add", "new.txt"],
            cwd=git_repo_with_commit,
            capture_output=True,
            check=True,
        )
        result = await git_tool("commit", args='-m "Test commit"')
        assert result.exit_code == 0
        assert "Test commit" in result.output

    async def test_commit_requires_args(self, git_repo_with_commit):
        result = await git_tool("commit")
        assert result.exit_code == 1
        assert "requires arguments" in result.error

    async def test_commit_nothing_staged(self, git_repo_with_commit):
        result = await git_tool("commit", args='-m "Empty"')
        # git commit with nothing staged returns non-zero
        assert result.exit_code != 0


# ─── Stash ──────────────────────────────────────────────────────────────────


class TestGitStash:
    async def test_stash_list_empty(self, git_repo_with_commit):
        result = await git_tool("stash")
        assert result.exit_code == 0

    async def test_stash_push_and_pop(self, git_repo_with_commit):
        readme = git_repo_with_commit / "README.md"
        readme.write_text("# Stashed change\n")
        push_result = await git_tool("stash", args="push")
        assert push_result.exit_code == 0
        # Verify file reverted
        assert readme.read_text() == "# Test\n"
        # Pop the stash
        pop_result = await git_tool("stash", args="pop")
        assert pop_result.exit_code == 0
        assert readme.read_text() == "# Stashed change\n"

    async def test_stash_list_with_entries(self, git_repo_with_commit):
        readme = git_repo_with_commit / "README.md"
        readme.write_text("# Stashed\n")
        await git_tool("stash", args="push")
        result = await git_tool("stash", args="list")
        assert result.exit_code == 0
        assert "stash@{0}" in result.output


# ─── Error handling ─────────────────────────────────────────────────────────


class TestGitErrors:
    async def test_not_a_git_repo(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = await git_tool("status")
        assert result.exit_code == 1
        assert "not a git repository" in result.error.lower()

    async def test_unknown_operation(self, git_repo_with_commit):
        result = await git_tool("rebase")
        assert result.exit_code == 1
        assert "Unknown git operation" in result.error

    async def test_invalid_args_quoting(self, git_repo_with_commit):
        result = await git_tool("log", args="'unterminated")
        assert result.exit_code == 1
        assert "Invalid arguments" in result.error


# ─── Safety classification ──────────────────────────────────────────────────


class TestGitToolClassification:
    def test_status_is_safe(self):
        c = _make_classifier()
        assert c.classify_tool_call("git_tool", {"operation": "status"}) == Risk.SAFE

    def test_diff_is_safe(self):
        c = _make_classifier()
        assert c.classify_tool_call("git_tool", {"operation": "diff"}) == Risk.SAFE

    def test_log_is_safe(self):
        c = _make_classifier()
        assert c.classify_tool_call("git_tool", {"operation": "log"}) == Risk.SAFE

    def test_branch_is_safe(self):
        c = _make_classifier()
        assert c.classify_tool_call("git_tool", {"operation": "branch"}) == Risk.SAFE

    def test_commit_requires_confirm(self):
        c = _make_classifier()
        assert c.classify_tool_call("git_tool", {"operation": "commit"}) == Risk.CONFIRM

    def test_stash_requires_confirm(self):
        c = _make_classifier()
        assert c.classify_tool_call("git_tool", {"operation": "stash"}) == Risk.CONFIRM

    def test_yolo_downgrades_commit(self):
        c = _make_classifier(mode="yolo")
        assert c.classify_tool_call("git_tool", {"operation": "commit"}) == Risk.SAFE

    def test_yolo_downgrades_stash(self):
        c = _make_classifier(mode="yolo")
        assert c.classify_tool_call("git_tool", {"operation": "stash"}) == Risk.SAFE

    def test_yolo_safe_ops_unchanged(self):
        c = _make_classifier(mode="yolo")
        assert c.classify_tool_call("git_tool", {"operation": "status"}) == Risk.SAFE
        assert c.classify_tool_call("git_tool", {"operation": "diff"}) == Risk.SAFE

    def test_unknown_operation_confirms(self):
        c = _make_classifier()
        assert c.classify_tool_call("git_tool", {"operation": "push"}) == Risk.CONFIRM
