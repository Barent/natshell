"""Regression table for classifier bypasses.

Every case here is a command that reached the user's shell with a lower risk
than it deserved.  These assert the *security property* — "this must not be
SAFE" — not the mechanism that delivers it, so a future rewrite of the
classifier internals should keep them passing.

Unlike ``test_safety.py``, which hand-builds a ``SafetyConfig``, these run
against the patterns NatShell actually ships in ``config.default.toml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import natshell
from natshell.config import NatShellConfig, SafetyConfig, _merge_toml
from natshell.safety.classifier import Risk, SafetyClassifier

_DEFAULT_CONFIG = Path(natshell.__file__).parent / "config.default.toml"


def _shipped_safety_config() -> SafetyConfig:
    """The safety config as shipped, not a test-local approximation."""
    config = NatShellConfig()
    _merge_toml(config, _DEFAULT_CONFIG)
    return config.safety


def _shipped_classifier(mode: str = "confirm") -> SafetyClassifier:
    safety = _shipped_safety_config()
    safety.mode = mode
    return SafetyClassifier(safety)


def test_shipped_config_actually_has_patterns():
    """Guards against the whole suite passing vacuously on an empty config."""
    safety = _shipped_safety_config()
    assert len(safety.blocked) > 0
    assert len(safety.always_confirm) > 0


# ─── C5: a newline is a command separator ───────────────────────────────────


class TestNewlineSeparator:
    """``bash -c`` runs every line; the classifier only ever saw the first."""

    @pytest.mark.parametrize(
        "command",
        [
            "cd /tmp\nrm -rf /",
            "echo hi\nrm -rf /",
            "ls\n\nrm -rf /",
            "cd /tmp\r\nrm -rf /",
        ],
    )
    def test_blocked_after_a_newline_is_still_blocked(self, command):
        assert _shipped_classifier().classify_command(command) == Risk.BLOCKED

    @pytest.mark.parametrize(
        "command",
        [
            "echo hi\nrm -rf /home/user",
            "ls -la\nsudo apt install nginx",
            "pwd\nchown -R root /srv",
            "echo ok\ncrontab -e",
        ],
    )
    def test_risky_after_a_newline_is_not_safe(self, command):
        assert _shipped_classifier().classify_command(command) != Risk.SAFE

    def test_newline_does_not_make_benign_commands_risky(self):
        assert _shipped_classifier().classify_command("ls -la\npwd\nwhoami") == Risk.SAFE


# ─── C8: the subshell paren the two tokenizers disagreed about ──────────────


class TestSubshellParens:
    """``execute_shell`` split on '(' to find sudo; the classifier did not, so
    ``(sudo ...)`` classified SAFE *and* got the cached root password piped in.
    """

    @pytest.mark.parametrize(
        "command",
        [
            "(sudo rm -rf ~)",
            "(sudo apt install nginx)",
            "( sudo systemctl stop sshd )",
            "true && (sudo chown -R root /)",
        ],
    )
    def test_sudo_in_a_subshell_is_not_safe(self, command):
        assert _shipped_classifier().classify_command(command) != Risk.SAFE

    def test_blocked_in_a_subshell_is_still_blocked(self):
        assert _shipped_classifier().classify_command("(rm -rf /)") == Risk.BLOCKED

    def test_quoted_paren_is_not_a_separator(self):
        assert _shipped_classifier().classify_command('echo "(hi)"') == Risk.SAFE


# ─── H1: a confirm match must not mask a blocked sub-command ────────────────


class TestBlockedIsNeverDowngraded:
    """The whole-command confirm pass ran before sub-commands were examined, so
    the first confirm match returned and the blocked sub-command after it was
    never reached.  BLOCKED is the one tier that survives danger mode, so
    losing it there loses it everywhere.
    """

    @pytest.mark.parametrize(
        "command",
        [
            "sudo apt update && rm -rf /",
            "rm notes.txt; rm -rf /",
            "echo x | tee /tmp/out && rm -rf /",
            "chown user file && rm -rf /",
            "kill -9 123 || rm -rf /",
        ],
    )
    def test_confirm_match_first_still_reports_blocked(self, command):
        assert _shipped_classifier().classify_command(command) == Risk.BLOCKED

    def test_subshell_expansion_does_not_mask_blocked(self):
        """`...` and $(...) returned CONFIRM before sub-commands were checked."""
        assert _shipped_classifier().classify_command("echo `date` && rm -rf /") == Risk.BLOCKED
        assert _shipped_classifier().classify_command("echo $(date) && rm -rf /") == Risk.BLOCKED

    def test_blocked_survives_danger_mode_behind_a_confirm_match(self):
        c = _shipped_classifier(mode="danger")
        risk = c.classify_tool_call("execute_shell", {"command": "sudo apt update && rm -rf /"})
        assert risk == Risk.BLOCKED


# ─── C6a: the patterns anchor on the first token, so anything that shifts ────
#          the binary off the front of the string evades every one of them


class TestInvocationNormalization:
    @pytest.mark.parametrize(
        "command",
        [
            "/bin/rm -rf /home/user/Documents",  # absolute path
            "/usr/bin/rm -rf /home/user",
            "LC_ALL=C rm -rf /home/user/Documents",  # env assignment prefix
            "FOO=bar BAZ=qux rm -rf /home/user",
            "command rm -rf /home/user",  # shell builtin wrapper
            "env rm -rf /home/user",
            "nohup rm -rf /home/user",
            "timeout 5 rm -rf /home/user",
            "nice -n 10 rm -rf /home/user",
            "xargs -I{} rm -rf /home/user",
            "/usr/bin/sudo apt install nginx",
            "setsid chown -R root /srv",
        ],
    )
    def test_wrapped_invocation_is_not_safe(self, command):
        assert _shipped_classifier().classify_command(command) != Risk.SAFE

    @pytest.mark.parametrize(
        "command",
        [
            "/bin/rm -rf /",
            "LC_ALL=C rm -rf /",
            "command rm -rf /",
            "sudo apt update && /bin/rm -rf /",
            "echo hi\n/bin/rm -rf /",
        ],
    )
    def test_wrapped_blocked_command_is_still_blocked(self, command):
        assert _shipped_classifier().classify_command(command) == Risk.BLOCKED

    @pytest.mark.parametrize(
        "command",
        [
            "ls -la",
            "/usr/bin/ls -la",
            "apt list --installed",
            "env python3 script.py",
            "docker ps",
            "systemctl status nginx",
            "grep -r TODO .",
            "timeout 5 curl https://example.com",
        ],
    )
    def test_benign_commands_stay_safe(self, command):
        """Normalization must not manufacture confirmations for ordinary work."""
        assert _shipped_classifier().classify_command(command) == Risk.SAFE
