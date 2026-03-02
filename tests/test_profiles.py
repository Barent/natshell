"""Tests for configuration profiles."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from natshell.config import (
    NatShellConfig,
    ProfileConfig,
    apply_profile,
    list_profiles,
    load_config,
)

# ─── ProfileConfig defaults ─────────────────────────────────────────────


class TestProfileConfigDefaults:
    def test_default_values(self):
        p = ProfileConfig()
        assert p.ollama_model == ""
        assert p.ollama_url == ""
        assert p.remote_url == ""
        assert p.remote_model == ""
        assert p.api_key == ""
        assert p.n_ctx == 0
        assert p.temperature == 0.0
        assert p.engine == ""
        assert p.n_gpu_layers == -2

    def test_natshell_config_has_profiles(self):
        cfg = NatShellConfig()
        assert isinstance(cfg.profiles, dict)
        assert len(cfg.profiles) == 0


# ─── Parsing [profiles.*] from TOML ─────────────────────────────────────


class TestProfileParsing:
    def test_loads_single_profile(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [profiles.coder]
            ollama_model = "qwen3-coder:30b"
            n_ctx = 131072
            temperature = 0.3
        """)
        )

        cfg = load_config(str(config_file))
        assert "coder" in cfg.profiles
        p = cfg.profiles["coder"]
        assert p.ollama_model == "qwen3-coder:30b"
        assert p.n_ctx == 131072
        assert p.temperature == 0.3

    def test_loads_multiple_profiles(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [profiles.lite]
            ollama_model = "qwen3:4b"
            n_ctx = 8192

            [profiles.coder]
            ollama_model = "qwen3-coder:30b"
            n_ctx = 131072
        """)
        )

        cfg = load_config(str(config_file))
        assert len(cfg.profiles) == 2
        assert "lite" in cfg.profiles
        assert "coder" in cfg.profiles
        assert cfg.profiles["lite"].ollama_model == "qwen3:4b"
        assert cfg.profiles["coder"].n_ctx == 131072

    def test_loads_remote_api_profile(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [profiles.remote-api]
            remote_url = "https://api.example.com/v1"
            remote_model = "gpt-4o"
            api_key = "sk-test"
            n_ctx = 128000
        """)
        )

        cfg = load_config(str(config_file))
        p = cfg.profiles["remote-api"]
        assert p.remote_url == "https://api.example.com/v1"
        assert p.remote_model == "gpt-4o"
        assert p.api_key == "sk-test"
        assert p.n_ctx == 128000

    def test_loads_local_cpu_profile(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [profiles.local-cpu]
            engine = "local"
            n_gpu_layers = 0
            n_ctx = 4096
        """)
        )

        cfg = load_config(str(config_file))
        p = cfg.profiles["local-cpu"]
        assert p.engine == "local"
        assert p.n_gpu_layers == 0
        assert p.n_ctx == 4096

    def test_ignores_unknown_keys(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [profiles.test]
            ollama_model = "qwen3:4b"
            unknown_key = "should be ignored"
        """)
        )

        cfg = load_config(str(config_file))
        assert "test" in cfg.profiles
        assert cfg.profiles["test"].ollama_model == "qwen3:4b"
        assert not hasattr(cfg.profiles["test"], "unknown_key")

    def test_no_profiles_section(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text('[ui]\ntheme = "dark"\n')

        cfg = load_config(str(config_file))
        assert len(cfg.profiles) == 0

    def test_profiles_alongside_other_sections(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [agent]
            temperature = 0.5

            [profiles.test]
            ollama_model = "qwen3:4b"
            n_ctx = 8192

            [ui]
            theme = "light"
        """)
        )

        cfg = load_config(str(config_file))
        assert cfg.agent.temperature == 0.5
        assert cfg.ui.theme == "light"
        assert "test" in cfg.profiles
        assert cfg.profiles["test"].ollama_model == "qwen3:4b"


# ─── list_profiles ───────────────────────────────────────────────────────


class TestListProfiles:
    def test_empty_config(self):
        cfg = NatShellConfig()
        assert list_profiles(cfg) == []

    def test_returns_sorted_names(self):
        cfg = NatShellConfig()
        cfg.profiles["coder"] = ProfileConfig(ollama_model="qwen3-coder:30b")
        cfg.profiles["lite"] = ProfileConfig(ollama_model="qwen3:4b")
        cfg.profiles["balanced"] = ProfileConfig(ollama_model="qwen3:14b")
        assert list_profiles(cfg) == ["balanced", "coder", "lite"]


# ─── apply_profile ───────────────────────────────────────────────────────


class TestApplyProfile:
    def test_unknown_profile_raises(self):
        cfg = NatShellConfig()
        with pytest.raises(KeyError, match="Unknown profile: nonexistent"):
            apply_profile(cfg, "nonexistent")

    def test_applies_ollama_model(self):
        cfg = NatShellConfig()
        cfg.profiles["test"] = ProfileConfig(ollama_model="qwen3:14b")
        apply_profile(cfg, "test")
        assert cfg.ollama.default_model == "qwen3:14b"

    def test_applies_ollama_url(self):
        cfg = NatShellConfig()
        cfg.profiles["test"] = ProfileConfig(ollama_url="http://myserver:11434")
        apply_profile(cfg, "test")
        assert cfg.ollama.url == "http://myserver:11434"

    def test_applies_remote_settings(self):
        cfg = NatShellConfig()
        cfg.profiles["test"] = ProfileConfig(
            remote_url="https://api.example.com/v1",
            remote_model="gpt-4o",
            api_key="sk-test",
        )
        apply_profile(cfg, "test")
        assert cfg.remote.url == "https://api.example.com/v1"
        assert cfg.remote.model == "gpt-4o"
        assert cfg.remote.api_key == "sk-test"

    def test_applies_n_ctx_to_both(self):
        cfg = NatShellConfig()
        cfg.profiles["test"] = ProfileConfig(n_ctx=65536)
        apply_profile(cfg, "test")
        assert cfg.ollama.n_ctx == 65536
        assert cfg.remote.n_ctx == 65536

    def test_applies_temperature(self):
        cfg = NatShellConfig()
        cfg.profiles["test"] = ProfileConfig(temperature=0.7)
        apply_profile(cfg, "test")
        assert cfg.agent.temperature == 0.7

    def test_applies_engine(self):
        cfg = NatShellConfig()
        cfg.profiles["test"] = ProfileConfig(engine="local")
        apply_profile(cfg, "test")
        assert cfg.engine.preferred == "local"

    def test_applies_n_gpu_layers(self):
        cfg = NatShellConfig()
        cfg.profiles["test"] = ProfileConfig(n_gpu_layers=0)
        apply_profile(cfg, "test")
        assert cfg.model.n_gpu_layers == 0

    def test_n_gpu_layers_negative_two_is_not_applied(self):
        """n_gpu_layers=-2 is the sentinel 'don't override' value."""
        cfg = NatShellConfig()
        cfg.model.n_gpu_layers = 42
        cfg.profiles["test"] = ProfileConfig(n_gpu_layers=-2)
        apply_profile(cfg, "test")
        assert cfg.model.n_gpu_layers == 42  # unchanged

    def test_only_overrides_nondefault_values(self):
        cfg = NatShellConfig()
        cfg.ollama.default_model = "original-model"
        cfg.agent.temperature = 0.5
        cfg.profiles["test"] = ProfileConfig(n_ctx=32768)
        apply_profile(cfg, "test")
        # Only n_ctx should change
        assert cfg.ollama.n_ctx == 32768
        # These should be untouched
        assert cfg.ollama.default_model == "original-model"
        assert cfg.agent.temperature == 0.5

    def test_full_profile_overrides(self):
        cfg = NatShellConfig()
        cfg.profiles["full"] = ProfileConfig(
            ollama_model="qwen3-coder:30b",
            ollama_url="http://server:11434",
            n_ctx=131072,
            temperature=0.3,
            engine="remote",
        )
        apply_profile(cfg, "full")
        assert cfg.ollama.default_model == "qwen3-coder:30b"
        assert cfg.ollama.url == "http://server:11434"
        assert cfg.ollama.n_ctx == 131072
        assert cfg.remote.n_ctx == 131072
        assert cfg.agent.temperature == 0.3
        assert cfg.engine.preferred == "remote"
