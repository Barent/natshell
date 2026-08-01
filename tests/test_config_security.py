"""Security tests for config writing: TOML escaping, atomic writes, and fault tolerance."""
from __future__ import annotations

import tomllib
from unittest.mock import patch

import pytest

from natshell.config import (
    NatShellConfig,
    _merge_toml,
    _toml_escape,
    _write_config_atomically,
    load_config,
    save_config_value,
    save_skills_disabled,
)


class TestTomlEscape:
    """_toml_escape must handle every adversarial string."""

    def test_plain(self):
        assert _toml_escape("hello") == '"hello"'

    def test_quote(self):
        assert _toml_escape('say "hi"') == '"say \\"hi\\""'

    def test_backslash(self):
        escaped = _toml_escape(r"C:\Users\me")
        assert escaped == r'"C:\\Users\\me"'

    def test_newline(self):
        assert _toml_escape("a\nb") == '"a\\nb"'

    def test_tab(self):
        assert _toml_escape("x\ty") == '"x\\ty"'

    def test_control_char(self):
        assert _toml_escape("a\x00b") == '"a\\u0000b"'

    def test_del(self):
        assert _toml_escape("foo\x7Fbar") == '"foo\\u007fbar"'

    def test_roundtrips(self):
        """Escaped string must parse back to the original via tomllib."""
        samples = [
            "hello",
            "it's fine",
            'say "bye"',
            r"back\slash",
            "line1\nline2",
            "\tindented",
            "ctrl\x01char",
        ]
        for raw in samples:
            escaped = _toml_escape(raw)
            parsed = tomllib.loads(f"key = {escaped}")
            assert parsed["key"] == raw, f"Roundtrip failed for {raw!r}"

    def test_injection_attempt(self):
        """An injected section header must not render as a new TOML table."""
        evil = '[safety]\nmode = "danger"\n'
        escaped = _toml_escape(evil)
        document = f"[model]\npath = {escaped}"
        parsed = tomllib.loads(document)
        assert "safety" not in parsed
        assert parsed["model"]["path"] == evil


class TestWriteConfigAtomically:
    """Atomic writes must reject invalid TOML and leave no temp files."""

    def test_valid_toml_succeeds(self, tmp_path):
        dest = tmp_path / "ok.toml"
        _write_config_atomically(dest, '[model]\npath = "foo"\n')
        assert dest.exists()
        data = tomllib.loads(dest.read_text())
        assert data["model"]["path"] == "foo"

    def test_invalid_toml_rejects(self, tmp_path):
        dest = tmp_path / "bad.toml"
        with pytest.raises(RuntimeError, match="invalid TOML"):
            _write_config_atomically(dest, "[model\n")
        assert not dest.exists()

    def test_preserves_existing_on_failure(self, tmp_path):
        dest = tmp_path / "keep.toml"
        dest.write_text('[model]\npath = "original"\n')
        with pytest.raises(RuntimeError):
            _write_config_atomically(dest, "[broken\n")
        assert dest.exists()
        data = tomllib.loads(dest.read_text())
        assert data["model"]["path"] == "original"

    def test_no_temp_file_left(self, tmp_path):
        dest = tmp_path / "clean.toml"
        _write_config_atomically(dest, "[x]\ny = 1\n")
        tomls = list(tmp_path.glob("*.toml"))
        assert len(tomls) == 1
        assert tomls[0].name == "clean.toml"


class TestSaveConfigValueSecurity:
    """save_config_value must escape strings — no raw interpolation."""

    def _mock_dir(self, tmp_path):
        return patch("natshell.config._get_config_dir", return_value=tmp_path)

    def test_string_with_quotes_is_safe(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text('[model]\npath = "original"\n')
        with self._mock_dir(tmp_path):
            save_config_value("model", "path", 'value with "quotes"')
        data = tomllib.loads(cfg.read_text())
        assert data["model"]["path"] == 'value with "quotes"'

    def test_backslash_windows_path(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text('[model]\npath = ""\n')
        with self._mock_dir(tmp_path):
            save_config_value("model", "path", r"C:\Users\test")
        data = tomllib.loads(cfg.read_text())
        assert data["model"]["path"] == r"C:\Users\test"

    def test_injection_blocked(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text('[model]\npath = ""\n[safety]\nmode = "warn"\n')
        evil = 'x"\n[safety]\nmode = "danger"\n'
        with self._mock_dir(tmp_path):
            save_config_value("model", "path", evil)
        data = tomllib.loads(cfg.read_text())
        assert data["safety"]["mode"] == "warn"

    def test_newstring_in_value_does_not_crash(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text("[agent]\ntemperature = 0.3\n")
        with self._mock_dir(tmp_path):
            save_config_value("agent", "temperature", 17)
        data = tomllib.loads(cfg.read_text())
        assert data["agent"]["temperature"] == 17


class TestSaveSkillsDisabledSecurity:
    """save_skills_disabled must escape skill names."""

    def _mock_dir(self, tmp_path):
        return patch("natshell.config._get_config_dir", return_value=tmp_path)

    def test_normal_list(self, tmp_path):
        cfg = tmp_path / "config.toml"
        with self._mock_dir(tmp_path):
            save_skills_disabled(["kiwix", "spotify"])
        data = tomllib.loads(cfg.read_text())
        assert data["skills"]["disabled"] == ["kiwix", "spotify"]

    def test_name_with_quote_escaped(self, tmp_path):
        cfg = tmp_path / "config.toml"
        with self._mock_dir(tmp_path):
            save_skills_disabled(['malicious"name'])
        data = tomllib.loads(cfg.read_text())
        assert data["skills"]["disabled"] == ['malicious"name']


class TestLoadConfigFaultTolerance:
    """_merge_toml must not crash on corrupted TOML."""

    def test_bad_toml_skipped(self, tmp_path):
        bad_cfg = tmp_path / "config.toml"
        bad_cfg.write_text("[broken\n")
        cfg = NatShellConfig()
        _merge_toml(cfg, bad_cfg)
        # Default values preserved — no crash
        assert cfg.model.path == "auto"

    def test_good_toml_merged(self, tmp_path):
        good_cfg = tmp_path / "config.toml"
        good_cfg.write_text('[agent]\nmax_steps = 42\n')
        cfg = NatShellConfig()
        _merge_toml(cfg, good_cfg)
        assert cfg.agent.max_steps == 42

    def test_load_config_with_bad_file(self, tmp_path):
        bad_cfg = tmp_path / "config.toml"
        bad_cfg.write_text("[unterminated\n")
        cfg = load_config(bad_cfg)
        # Gracefully returns defaults instead of crashing
        assert isinstance(cfg, NatShellConfig)
