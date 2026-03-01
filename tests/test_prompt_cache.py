"""Tests for prompt caching in local inference."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from natshell.config import ModelConfig


class TestPromptCacheConfig:
    def test_default_config_has_prompt_cache_enabled(self):
        """ModelConfig defaults should have prompt_cache=True and prompt_cache_mb=256."""
        mc = ModelConfig()
        assert mc.prompt_cache is True
        assert mc.prompt_cache_mb == 256


class TestLocalEnginePromptCache:
    def _build_engine(self, prompt_cache: bool = True, prompt_cache_mb: int = 128):
        """Build a LocalEngine with a fully mocked llama_cpp module."""
        mock_llm = MagicMock()
        mock_llama_cls = MagicMock(return_value=mock_llm)
        mock_cache = MagicMock()
        mock_cache_cls = MagicMock(return_value=mock_cache)

        fake_llama_cpp = MagicMock(
            Llama=mock_llama_cls,
            LlamaRAMCache=mock_cache_cls,
        )

        with (
            patch.dict(sys.modules, {"llama_cpp": fake_llama_cpp}),
            patch("natshell.gpu.gpu_backend_available", return_value=False),
            patch("natshell.gpu.best_gpu_index", return_value=0),
        ):
            from natshell.inference.local import LocalEngine

            engine = LocalEngine(
                model_path="/tmp/fake-model-4B.gguf",
                n_ctx=4096,
                n_gpu_layers=0,
                prompt_cache=prompt_cache,
                prompt_cache_mb=prompt_cache_mb,
            )

        return engine, mock_llm, mock_cache_cls

    def test_accepts_prompt_cache_param(self):
        """LocalEngine should accept prompt_cache parameter without error."""
        engine, mock_llm, mock_cache_cls = self._build_engine(
            prompt_cache=True, prompt_cache_mb=128,
        )

        assert engine.n_ctx == 4096
        # set_cache should have been called on the llm instance
        mock_llm.set_cache.assert_called_once()
        # LlamaRAMCache should have been constructed with correct capacity
        mock_cache_cls.assert_called_once_with(capacity_bytes=128 * 1024 * 1024)

    def test_cache_not_set_when_disabled(self):
        """When prompt_cache=False, set_cache should not be called."""
        engine, mock_llm, mock_cache_cls = self._build_engine(
            prompt_cache=False,
        )

        assert engine.n_ctx == 4096
        mock_llm.set_cache.assert_not_called()
        mock_cache_cls.assert_not_called()
