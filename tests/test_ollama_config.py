"""Tests for OllamaConfig and save_ollama_default."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from natshell.config import (
    NatShellConfig,
    OllamaConfig,
    load_config,
    save_ollama_default,
)


# ─── OllamaConfig defaults ─────────────────────────────────────────────────


class TestOllamaConfigDefaults:
    def test_default_values(self):
        cfg = OllamaConfig()
        assert cfg.url == ""
        assert cfg.default_model == ""

    def test_natshell_config_has_ollama(self):
        cfg = NatShellConfig()
        assert isinstance(cfg.ollama, OllamaConfig)


# ─── Loading [ollama] from TOML ────────────────────────────────────────────


class TestOllamaConfigLoading:
    def test_loads_ollama_section(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [ollama]
            url = "http://myhost:11434"
            default_model = "llama3:8b"
        """))

        cfg = load_config(str(config_file))
        assert cfg.ollama.url == "http://myhost:11434"
        assert cfg.ollama.default_model == "llama3:8b"

    def test_missing_ollama_section_uses_defaults(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("[ui]\ntheme = \"light\"\n")

        cfg = load_config(str(config_file))
        assert cfg.ollama.url == ""
        assert cfg.ollama.default_model == ""

    def test_partial_ollama_section(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [ollama]
            url = "http://gpu-box:11434"
        """))

        cfg = load_config(str(config_file))
        assert cfg.ollama.url == "http://gpu-box:11434"
        assert cfg.ollama.default_model == ""


# ─── save_ollama_default ────────────────────────────────────────────────────


class TestSaveOllamaDefault:
    def test_creates_new_file(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        path = save_ollama_default("qwen3:4b")

        assert path.exists()
        content = path.read_text()
        assert "[ollama]" in content
        assert 'default_model = "qwen3:4b"' in content

    def test_updates_existing_section(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        config_dir = tmp_path / ".config" / "natshell"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [ollama]
            url = "http://localhost:11434"
            default_model = "old-model"

            [ui]
            theme = "dark"
        """))

        save_ollama_default("new-model")
        content = config_file.read_text()
        assert 'default_model = "new-model"' in content
        assert "old-model" not in content
        # Other sections should be preserved
        assert "[ui]" in content
        assert '[ollama]' in content

    def test_adds_section_to_existing_file(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        config_dir = tmp_path / ".config" / "natshell"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        config_file.write_text("[ui]\ntheme = \"dark\"\n")

        save_ollama_default("llama3:8b")
        content = config_file.read_text()
        assert "[ollama]" in content
        assert 'default_model = "llama3:8b"' in content
        # Original content should be preserved
        assert "[ui]" in content

    def test_updates_commented_default_model(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        config_dir = tmp_path / ".config" / "natshell"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [ollama]
            url = "http://localhost:11434"
            # default_model = "qwen3:4b"
        """))

        save_ollama_default("mistral:7b")
        content = config_file.read_text()
        assert 'default_model = "mistral:7b"' in content
        assert "# default_model" not in content
