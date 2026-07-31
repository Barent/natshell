"""PowerShell parameter names are abbreviatable and cmdlets have aliases.

The shipped Windows patterns are written as if PowerShell were bash:
`Remove-Item\\s+.*-Recurse` requires the parameter spelled in full, but
PowerShell accepts any unambiguous prefix — `-Recurse`, `-Recurs`, `-Rec`,
`-R` all bind to the same parameter. The aliases (`ri`, `del`, `rm`, `rd`,
`erase`) are not mentioned at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import natshell
from natshell.config import NatShellConfig, _merge_toml
from natshell.safety.classifier import Risk, SafetyClassifier


def _windows_classifier(mode: str = "confirm") -> SafetyClassifier:
    config = NatShellConfig()
    _merge_toml(config, Path(natshell.__file__).parent / "config.default.toml")
    config.safety.mode = mode
    return SafetyClassifier(config.safety)


class TestRemoveItemAbbreviations:
    @pytest.mark.parametrize(
        "command",
        [
            r"Remove-Item ./temp -Recurse",
            r"Remove-Item ./temp -Recurs",
            r"Remove-Item ./temp -Recur",
            r"Remove-Item ./temp -Rec",
            r"Remove-Item ./temp -Re",
            r"Remove-Item ./temp -R",
            r"Remove-Item -Rec -For ./temp",
            r"remove-item ./temp -rec",  # PowerShell is case-insensitive
            r"REMOVE-ITEM ./temp -RECURSE",
        ],
    )
    def test_recursive_delete_is_not_safe(self, command):
        assert _windows_classifier().classify_command(command) != Risk.SAFE

    @pytest.mark.parametrize(
        "command",
        [
            r"ri ./temp -Recurse",
            r"rd ./temp -Rec",
            r"erase ./temp -Recurse",
        ],
    )
    def test_aliases_are_not_safe(self, command):
        assert _windows_classifier().classify_command(command) != Risk.SAFE

    @pytest.mark.parametrize(
        "command",
        [
            r"Remove-Item -Recurse -Force C:\ ",
            r"Remove-Item -Rec -For C:\ ",
            r"Remove-Item -R -F C:\ ",
            r"remove-item -recurse -force c:\ ",
        ],
    )
    def test_wiping_the_system_drive_is_blocked(self, command):
        assert _windows_classifier().classify_command(command.strip()) == Risk.BLOCKED


class TestOtherWindowsCmdlets:
    @pytest.mark.parametrize(
        "command",
        [
            r"Stop-Service wuauserv",
            r"Set-ExecutionPolicy Unrestricted",
        ],
    )
    def test_still_confirms(self, command):
        assert _windows_classifier().classify_command(command) != Risk.SAFE

    @pytest.mark.parametrize(
        "command",
        [
            r"dir C:\Users",
            r"Get-Process",
            r"ipconfig /all",
            r"systeminfo",
            r"Get-ChildItem -Recurse",  # listing recursively is not deleting
            r"Get-Content notes.txt",
            r"Remove-Item notes.txt",  # a single non-recursive delete
        ],
    )
    def test_benign_stays_safe(self, command):
        assert _windows_classifier().classify_command(command) == Risk.SAFE
