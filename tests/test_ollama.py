"""Tests for the Ollama/remote API client module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from natshell.inference.ollama import (
    OllamaModel,
    list_models,
    normalize_base_url,
    ping_server,
)


# ─── normalize_base_url ─────────────────────────────────────────────────────


class TestNormalizeBaseUrl:
    def test_strips_v1_suffix(self):
        assert normalize_base_url("http://localhost:11434/v1") == "http://localhost:11434"

    def test_strips_v1_with_trailing_slash(self):
        assert normalize_base_url("http://localhost:11434/v1/") == "http://localhost:11434"

    def test_no_v1_unchanged(self):
        assert normalize_base_url("http://localhost:11434") == "http://localhost:11434"

    def test_strips_trailing_slash(self):
        assert normalize_base_url("http://localhost:11434/") == "http://localhost:11434"

    def test_custom_host_and_port(self):
        assert normalize_base_url("http://192.168.1.5:8080/v1") == "http://192.168.1.5:8080"

    def test_bare_url(self):
        assert normalize_base_url("http://myhost") == "http://myhost"


# ─── ping_server ────────────────────────────────────────────────────────────


class TestPingServer:
    async def test_success_ollama_running(self):
        mock_response = httpx.Response(200, text="Ollama is running")
        with patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_response)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await ping_server("http://localhost:11434")
            assert result is True

    async def test_success_non_ollama_200(self):
        mock_response = httpx.Response(200, text="OK")
        with patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_response)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await ping_server("http://localhost:8080")
            assert result is True

    async def test_connection_failure(self):
        with patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await ping_server("http://badhost:11434")
            assert result is False

    async def test_strips_v1_before_ping(self):
        mock_response = httpx.Response(200, text="Ollama is running")
        with patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_response)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await ping_server("http://localhost:11434/v1")
            assert result is True
            # Should have pinged the root, not /v1/
            instance.get.assert_called_with("http://localhost:11434/")


# ─── list_models ────────────────────────────────────────────────────────────


class TestListModels:
    async def test_ollama_api_tags(self):
        """list_models parses Ollama /api/tags response."""
        ollama_response = httpx.Response(200, json={
            "models": [
                {
                    "name": "qwen3:4b",
                    "size": 2684354560,
                    "details": {"parameter_size": "4B", "family": "qwen3"},
                },
                {
                    "name": "llama3:8b",
                    "size": 5368709120,
                    "details": {"parameter_size": "8B", "family": "llama"},
                },
            ]
        })

        with patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=ollama_response)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            models = await list_models("http://localhost:11434")

        assert len(models) == 2
        assert models[0].name == "qwen3:4b"
        assert models[0].size_gb == 2.5
        assert models[0].parameter_size == "4B"
        assert models[1].name == "llama3:8b"

    async def test_openai_v1_models_fallback(self):
        """list_models falls back to /v1/models when /api/tags fails."""
        tags_404 = httpx.Response(404, text="Not Found")
        openai_response = httpx.Response(200, json={
            "data": [
                {"id": "gpt-4"},
                {"id": "gpt-3.5-turbo"},
            ]
        })

        with patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(side_effect=[tags_404, openai_response])
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            models = await list_models("http://localhost:8080")

        assert len(models) == 2
        assert models[0].name == "gpt-4"
        assert models[1].name == "gpt-3.5-turbo"

    async def test_connection_failure_returns_empty(self):
        with patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("refused"))
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            models = await list_models("http://badhost:11434")
            assert models == []
