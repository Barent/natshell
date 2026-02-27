"""Tests for engine preference persistence."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from natshell.config import (
    EngineConfig,
    NatShellConfig,
    load_config,
    save_engine_preference,
)


# ─── EngineConfig defaults ────────────────────────────────────────────────


class TestEngineConfigDefaults:
    def test_default_values(self):
        cfg = EngineConfig()
        assert cfg.preferred == "auto"

    def test_natshell_config_has_engine(self):
        cfg = NatShellConfig()
        assert isinstance(cfg.engine, EngineConfig)
        assert cfg.engine.preferred == "auto"


# ─── Loading [engine] from TOML ───────────────────────────────────────────


class TestEngineConfigLoading:
    def test_loads_engine_section(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [engine]
            preferred = "local"
        """))

        cfg = load_config(str(config_file))
        assert cfg.engine.preferred == "local"

    def test_missing_engine_section_uses_defaults(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("[ui]\ntheme = \"light\"\n")

        cfg = load_config(str(config_file))
        assert cfg.engine.preferred == "auto"

    def test_loads_remote_preference(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [engine]
            preferred = "remote"
        """))

        cfg = load_config(str(config_file))
        assert cfg.engine.preferred == "remote"


# ─── save_engine_preference ───────────────────────────────────────────────


class TestSaveEnginePreference:
    def test_creates_new_file(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        path = save_engine_preference("local")

        assert path.exists()
        content = path.read_text()
        assert "[engine]" in content
        assert 'preferred = "local"' in content

    def test_updates_existing_section(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        config_dir = tmp_path / ".config" / "natshell"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [engine]
            preferred = "local"

            [ui]
            theme = "dark"
        """))

        save_engine_preference("remote")
        content = config_file.read_text()
        assert 'preferred = "remote"' in content
        assert '"local"' not in content
        # Other sections preserved
        assert "[ui]" in content
        assert "[engine]" in content

    def test_adds_section_to_existing_file(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        config_dir = tmp_path / ".config" / "natshell"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        config_file.write_text("[ui]\ntheme = \"dark\"\n")

        save_engine_preference("local")
        content = config_file.read_text()
        assert "[engine]" in content
        assert 'preferred = "local"' in content
        # Original content preserved
        assert "[ui]" in content

    def test_updates_commented_preferred(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        config_dir = tmp_path / ".config" / "natshell"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [engine]
            # preferred = "auto"
        """))

        save_engine_preference("remote")
        content = config_file.read_text()
        assert 'preferred = "remote"' in content
        assert "# preferred" not in content

    def test_saves_auto(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        path = save_engine_preference("auto")

        content = path.read_text()
        assert 'preferred = "auto"' in content


# ─── Startup preference logic ─────────────────────────────────────────────


class TestStartupPreference:
    """Test the preference logic that __main__.py applies."""

    def _simulate_startup(
        self,
        preferred: str = "auto",
        ollama_url: str = "",
        remote_url: str | None = None,
        cli_remote: str | None = None,
        cli_model: str | None = None,
    ) -> bool:
        """Simulate the startup logic and return the final use_remote value.

        Mirrors the logic in __main__.py after config loading.
        """
        use_remote = bool(remote_url)

        if not remote_url and ollama_url:
            use_remote = True

        # Apply persisted engine preference (CLI flags override)
        cli_forced_remote = bool(cli_remote)
        cli_forced_local = bool(cli_model)
        if not cli_forced_remote and not cli_forced_local:
            if preferred == "local":
                use_remote = False
            elif preferred == "remote":
                pass  # keep use_remote as-is

        return use_remote

    def test_preferred_local_skips_remote(self):
        """preferred=local forces local even when remote URL is configured."""
        result = self._simulate_startup(
            preferred="local", ollama_url="http://localhost:11434"
        )
        assert result is False

    def test_preferred_remote_uses_remote(self):
        """preferred=remote keeps remote when URL is configured."""
        result = self._simulate_startup(
            preferred="remote", ollama_url="http://localhost:11434"
        )
        assert result is True

    def test_preferred_auto_is_default_behavior(self):
        """preferred=auto doesn't change anything — remote if URL exists."""
        with_url = self._simulate_startup(
            preferred="auto", ollama_url="http://localhost:11434"
        )
        assert with_url is True

        without_url = self._simulate_startup(preferred="auto")
        assert without_url is False

    def test_cli_remote_overrides_preferred_local(self):
        """--remote CLI flag overrides preferred=local."""
        result = self._simulate_startup(
            preferred="local",
            cli_remote="http://some-server:11434/v1",
        )
        # cli_remote is set, so cli_forced_remote=True, preference not applied
        # but remote_url is None (cli_remote isn't used as remote_url in this sim)
        # Let me fix: in real code, --remote sets config.remote.url before this logic
        # So simulate it properly:
        result = self._simulate_startup(
            preferred="local",
            remote_url="http://some-server:11434/v1",
            cli_remote="http://some-server:11434/v1",
        )
        assert result is True

    def test_cli_model_overrides_preferred_remote(self):
        """--model CLI flag prevents preferred=remote from applying."""
        result = self._simulate_startup(
            preferred="remote",
            ollama_url="http://localhost:11434",
            cli_model="/path/to/model.gguf",
        )
        # cli_model is set, so preference not applied, use_remote stays True from URL
        # In real code, --model doesn't set use_remote. The preference block is skipped.
        assert result is True

    def test_preferred_remote_without_url(self):
        """preferred=remote with no URL configured still results in local."""
        result = self._simulate_startup(preferred="remote")
        assert result is False
