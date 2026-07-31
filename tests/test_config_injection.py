"""Config values must be written as data, not as TOML source.

save_config_value rendered a string as f'"{value}"' with no escaping, so a
value containing a quote and a newline closed the string and continued the
file as attacker-chosen TOML.  update_config's key allowlist was then beside
the point: the injected text could reach any section, including [safety].

The same shape is a denial of service.  Any value with an unbalanced quote
produced a file that tomllib could not parse, and load_config -> _merge_toml
-> tomllib.load raised uncaught, so NatShell refused to start until someone
hand-edited the file.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from natshell.config import load_config, save_config_value

# Closes the string, then opens a section the allowlist was built to protect.
INJECTION = 'x"\nblocked = []\nalways_confirm = []\n[safety]\nmode = "danger"'


@pytest.fixture
def config_dir(tmp_path: Path):
    d = tmp_path / ".config" / "natshell"
    with patch("natshell.config._get_config_dir", return_value=d):
        yield d


class TestStringEscaping:
    def test_injection_does_not_create_new_keys(self, config_dir: Path):
        save_config_value("prompt", "persona", INJECTION)

        data = tomllib.loads((config_dir / "config.toml").read_text(encoding="utf-8"))
        assert data["prompt"]["persona"] == INJECTION
        assert "safety" not in data

    def test_injection_does_not_reach_the_live_config(self, config_dir: Path):
        save_config_value("prompt", "persona", INJECTION)

        config = load_config(config_dir / "config.toml")
        assert config.safety.mode != "danger"
        assert len(config.safety.always_confirm) > 0

    @pytest.mark.parametrize(
        "value",
        [
            'has "quotes"',
            "has\nnewline",
            "has\r\ncrlf",
            "trailing backslash\\",
            r"C:\Users\stump\models",  # a Windows path is not an escape sequence
            "tab\there",
            'both " and \\',
            "\x00\x01",
        ],
    )
    def test_values_round_trip_exactly(self, config_dir: Path, value: str):
        save_config_value("prompt", "persona", value)

        data = tomllib.loads((config_dir / "config.toml").read_text(encoding="utf-8"))
        assert data["prompt"]["persona"] == value

    def test_file_always_parses(self, config_dir: Path):
        for value in [INJECTION, 'unbalanced "', "\\", "]\n[safety]"]:
            save_config_value("prompt", "persona", value)
            tomllib.loads((config_dir / "config.toml").read_text(encoding="utf-8"))

    def test_existing_sections_survive(self, config_dir: Path):
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text(
            '[agent]\ntemperature = 0.5\n\n[ui]\ntheme = "light"\n', encoding="utf-8"
        )
        save_config_value("prompt", "persona", INJECTION)

        data = tomllib.loads((config_dir / "config.toml").read_text(encoding="utf-8"))
        assert data["agent"]["temperature"] == 0.5
        assert data["ui"]["theme"] == "light"


class TestCorruptConfigDoesNotBlockStartup:
    def test_unparseable_user_config_falls_back_to_defaults(self, tmp_path: Path):
        bad = tmp_path / "config.toml"
        bad.write_text('[safety]\nmode = "confirm\n', encoding="utf-8")  # unclosed string

        config = load_config(bad)

        assert config.safety.mode == "confirm"
        assert len(config.safety.always_confirm) > 0

    def test_partially_valid_config_still_starts(self, tmp_path: Path):
        bad = tmp_path / "config.toml"
        bad.write_text("[agent]\ntemperature = = 0.5\n", encoding="utf-8")

        config = load_config(bad)
        assert config.agent.temperature > 0


class TestListValues:
    """skills.disabled is declared as a list, but _coerce_value had no list
    branch, so the value stayed a string and set() shredded it into letters."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('["a", "b"]', ["a", "b"]),
            ("a, b", ["a", "b"]),
            ("solo", ["solo"]),
            ("[]", []),
            ("", []),
        ],
    )
    def test_coerce_list(self, raw, expected):
        from natshell.tools.update_config import _coerce_value

        assert _coerce_value(raw, "list") == expected

    async def test_disabled_skills_round_trip(self, config_dir: Path):
        from natshell.tools.update_config import update_config

        result = await update_config("skills", "disabled", '["web-research"]')
        assert result.exit_code == 0

        data = tomllib.loads((config_dir / "config.toml").read_text(encoding="utf-8"))
        assert data["skills"]["disabled"] == ["web-research"]

    async def test_disabled_skill_name_is_not_shredded(self, config_dir: Path):
        from natshell.tools.update_config import update_config

        await update_config("skills", "disabled", "web-research")

        config = load_config(config_dir / "config.toml")
        assert config.skills.disabled == ["web-research"]
