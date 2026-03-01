"""Tests for extracted slash command handlers and model manager."""

from __future__ import annotations

from natshell.config import NatShellConfig, OllamaConfig, RemoteConfig
from natshell.inference.engine import EngineInfo
from natshell.model_manager import (
    find_local_model,
    format_model_info,
    format_model_list,
    get_remote_base_url,
    resolve_local_model_path,
)
from natshell.tools.limits import ToolLimits

# ─── ToolLimits ──────────────────────────────────────────────────────────────


class TestToolLimits:
    def test_defaults(self):
        lim = ToolLimits()
        assert lim.max_output_chars == 4000
        assert lim.read_file_lines == 200

    def test_custom_values(self):
        lim = ToolLimits(max_output_chars=16000, read_file_lines=1000)
        assert lim.max_output_chars == 16000
        assert lim.read_file_lines == 1000


# ─── get_remote_base_url ─────────────────────────────────────────────────────


class TestGetRemoteBaseUrl:
    def test_ollama_url_preferred(self):
        config = NatShellConfig(
            ollama=OllamaConfig(url="http://ollama:11434"),
            remote=RemoteConfig(url="http://remote:8080"),
        )
        info = EngineInfo(engine_type="remote", base_url="http://engine:9999")
        url = get_remote_base_url(config, info)
        assert "ollama" in url

    def test_remote_url_fallback(self):
        config = NatShellConfig(remote=RemoteConfig(url="http://remote:8080"))
        info = EngineInfo(engine_type="remote")
        url = get_remote_base_url(config, info)
        assert "remote" in url

    def test_none_when_empty(self):
        config = NatShellConfig()
        info = EngineInfo(engine_type="local")
        assert get_remote_base_url(config, info) is None


# ─── format_model_info ───────────────────────────────────────────────────────


class TestFormatModelInfo:
    def test_remote_engine_info(self):
        info = EngineInfo(
            engine_type="remote",
            model_name="qwen3:4b",
            base_url="http://localhost:11434/v1",
            n_ctx=8192,
        )
        config = NatShellConfig()
        result = format_model_info(info, config)
        assert "remote" in result
        assert "qwen3:4b" in result
        assert "8192" in result


# ─── format_model_list ───────────────────────────────────────────────────────


class TestFormatModelList:
    def test_marks_active_model(self):
        from natshell.inference.ollama import OllamaModel

        models = [
            OllamaModel(name="model-a", size_gb=2.5, parameter_size="4B"),
            OllamaModel(name="model-b", size_gb=7.0, parameter_size="8B"),
        ]
        result = format_model_list(models, "model-a")
        assert "active" in result
        assert "model-a" in result
        assert "model-b" in result


# ─── find_local_model ────────────────────────────────────────────────────────


class TestFindLocalModel:
    def test_find_by_name(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        assert find_local_model("model.gguf", [model]) == model

    def test_find_by_stem(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        assert find_local_model("model", [model]) == model

    def test_not_found(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        assert find_local_model("other", [model]) is None


# ─── resolve_local_model_path ────────────────────────────────────────────────


class TestResolveLocalModelPath:
    def test_auto_expands(self):
        config = NatShellConfig()
        path = resolve_local_model_path(config)
        assert "natshell" in path
        assert "models" in path

    def test_explicit_path(self):
        config = NatShellConfig()
        config.model.path = "/custom/model.gguf"
        assert resolve_local_model_path(config) == "/custom/model.gguf"
