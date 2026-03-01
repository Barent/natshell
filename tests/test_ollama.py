"""Tests for the Ollama/remote API client module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx

from natshell.inference.ollama import (
    _get_running_context,
    _model_matches,
    get_model_context_length,
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
        ollama_response = httpx.Response(
            200,
            json={
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
            },
        )

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
        openai_response = httpx.Response(
            200,
            json={
                "data": [
                    {"id": "gpt-4"},
                    {"id": "gpt-3.5-turbo"},
                ]
            },
        )

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


# ─── get_model_context_length ────────────────────────────────────────────────


# ─── _model_matches ─────────────────────────────────────────────────────────


class TestModelMatches:
    def test_exact_match_name(self):
        assert _model_matches("qwen3:4b", "qwen3:4b", "qwen3:4b") is True

    def test_exact_match_model_field(self):
        assert _model_matches("qwen3:4b", "other", "qwen3:4b") is True

    def test_no_match(self):
        assert _model_matches("qwen3:4b", "llama3:8b", "llama3:8b") is False

    def test_implicit_latest_tag(self):
        """Query without tag matches entry with :latest."""
        assert _model_matches("qwen3", "qwen3:latest", "qwen3:latest") is True

    def test_implicit_latest_no_false_positive(self):
        """Query without tag should not match a different tag."""
        assert _model_matches("qwen3", "qwen3:4b", "qwen3:4b") is False


# ─── _get_running_context ───────────────────────────────────────────────────


class TestGetRunningContext:
    async def test_model_loaded_returns_context_length(self):
        """Returns context_length when model is loaded in /api/ps."""
        ps_response = httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "qwen3:4b",
                        "model": "qwen3:4b",
                        "context_length": 32768,
                        "size_vram": 2684354560,
                    }
                ]
            },
        )
        with patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=ps_response)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await _get_running_context("http://localhost:11434", "qwen3:4b")
            assert result == 32768

    async def test_model_not_loaded_returns_none(self):
        """Returns None when model is not in /api/ps results."""
        ps_response = httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "llama3:8b",
                        "model": "llama3:8b",
                        "context_length": 8192,
                    }
                ]
            },
        )
        with patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=ps_response)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await _get_running_context("http://localhost:11434", "qwen3:4b")
            assert result is None

    async def test_non_ollama_server_returns_none(self):
        """Non-Ollama server (404 on /api/ps) returns None."""
        ps_response = httpx.Response(404, text="Not Found")
        with patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=ps_response)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await _get_running_context("http://localhost:8080", "gpt-4")
            assert result is None

    async def test_connection_failure_returns_none(self):
        """Connection failure returns None."""
        with patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("refused"))
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await _get_running_context("http://badhost:11434", "qwen3:4b")
            assert result is None

    async def test_empty_models_returns_none(self):
        """Empty models list returns None."""
        ps_response = httpx.Response(200, json={"models": []})
        with patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=ps_response)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await _get_running_context("http://localhost:11434", "qwen3:4b")
            assert result is None

    async def test_implicit_latest_tag_match(self):
        """Query "qwen3" matches entry with name "qwen3:latest"."""
        ps_response = httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "qwen3:latest",
                        "model": "qwen3:latest",
                        "context_length": 4096,
                    }
                ]
            },
        )
        with patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=ps_response)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await _get_running_context("http://localhost:11434", "qwen3")
            assert result == 4096


# ─── get_model_context_length ────────────────────────────────────────────────


class TestGetModelContextLength:
    async def test_prefers_api_ps_over_api_show(self):
        """When model is running, uses /api/ps context instead of /api/show metadata."""
        with patch(
            "natshell.inference.ollama._get_running_context",
            new_callable=AsyncMock,
            return_value=32768,
        ):
            result = await get_model_context_length("http://localhost:11434", "qwen3:4b")
            assert result == 32768

    async def test_falls_back_to_api_show_when_not_running(self):
        """When model is not in /api/ps, falls back to /api/show metadata."""
        show_response = httpx.Response(
            200,
            json={"model_info": {"qwen2.context_length": 262144}},
        )
        with (
            patch(
                "natshell.inference.ollama._get_running_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient,
        ):
            instance = AsyncMock()
            instance.post = AsyncMock(return_value=show_response)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await get_model_context_length("http://localhost:11434", "qwen3:4b")
            assert result == 262144

    async def test_returns_context_length_from_model_info(self):
        """Successful /api/show query returns context length from architecture-prefixed key."""
        show_response = httpx.Response(
            200,
            json={
                "model_info": {
                    "general.architecture": "qwen2",
                    "qwen2.context_length": 32768,
                    "qwen2.embedding_length": 3584,
                }
            },
        )
        with (
            patch(
                "natshell.inference.ollama._get_running_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient,
        ):
            instance = AsyncMock()
            instance.post = AsyncMock(return_value=show_response)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await get_model_context_length("http://localhost:11434", "qwen3:32b")
            assert result == 32768

    async def test_returns_llama_context_length(self):
        """Works with llama architecture prefix."""
        show_response = httpx.Response(
            200,
            json={
                "model_info": {
                    "general.architecture": "llama",
                    "llama.context_length": 8192,
                }
            },
        )
        with (
            patch(
                "natshell.inference.ollama._get_running_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient,
        ):
            instance = AsyncMock()
            instance.post = AsyncMock(return_value=show_response)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await get_model_context_length("http://localhost:11434", "llama3:8b")
            assert result == 8192

    async def test_non_ollama_server_returns_zero(self):
        """Non-Ollama server (404 on both endpoints) returns 0."""
        show_response = httpx.Response(404, text="Not Found")
        with (
            patch(
                "natshell.inference.ollama._get_running_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient,
        ):
            instance = AsyncMock()
            instance.post = AsyncMock(return_value=show_response)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await get_model_context_length("http://localhost:8080", "gpt-4")
            assert result == 0

    async def test_connection_failure_returns_zero(self):
        """Connection failure returns 0."""
        with (
            patch(
                "natshell.inference.ollama._get_running_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient,
        ):
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("refused"))
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await get_model_context_length("http://badhost:11434", "qwen3:4b")
            assert result == 0

    async def test_missing_model_info_returns_zero(self):
        """Response without model_info key returns 0."""
        show_response = httpx.Response(
            200,
            json={
                "license": "apache-2.0",
                "modelfile": "FROM qwen3:4b",
            },
        )
        with (
            patch(
                "natshell.inference.ollama._get_running_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient,
        ):
            instance = AsyncMock()
            instance.post = AsyncMock(return_value=show_response)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await get_model_context_length("http://localhost:11434", "qwen3:4b")
            assert result == 0

    async def test_strips_v1_from_url(self):
        """URL with /v1 suffix is normalized before querying."""
        show_response = httpx.Response(200, json={"model_info": {"qwen2.context_length": 4096}})
        with (
            patch(
                "natshell.inference.ollama._get_running_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("natshell.inference.ollama.httpx.AsyncClient") as MockClient,
        ):
            instance = AsyncMock()
            instance.post = AsyncMock(return_value=show_response)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await get_model_context_length("http://localhost:11434/v1", "qwen3:4b")
            assert result == 4096
            # Should have posted to the normalized URL
            instance.post.assert_called_with(
                "http://localhost:11434/api/show",
                json={"model": "qwen3:4b"},
            )


# ─── ContextOverflowError detection in RemoteEngine ──────────────────────────


class TestContextOverflowDetection:
    async def test_400_context_length_raises_overflow(self):
        """HTTP 400 with context-length body raises ContextOverflowError."""
        from natshell.inference.remote import ContextOverflowError, RemoteEngine

        engine = RemoteEngine(base_url="http://localhost:11434", model="test")
        mock_response = httpx.Response(
            status_code=400,
            text="model requires more context length",
            request=httpx.Request("POST", "http://localhost:11434/chat/completions"),
        )
        engine.client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "bad request", request=mock_response.request, response=mock_response
            )
        )
        try:
            await engine.chat_completion(messages=[{"role": "user", "content": "hi"}])
            assert False, "Should have raised"
        except ContextOverflowError as e:
            assert "context" in str(e).lower()
        finally:
            await engine.close()

    async def test_413_raises_overflow(self):
        """HTTP 413 with known pattern raises ContextOverflowError."""
        from natshell.inference.remote import ContextOverflowError, RemoteEngine

        engine = RemoteEngine(base_url="http://localhost:11434", model="test")
        mock_response = httpx.Response(
            status_code=413,
            text="request too large: prompt exceeds token limit",
            request=httpx.Request("POST", "http://localhost:11434/chat/completions"),
        )
        engine.client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "too large", request=mock_response.request, response=mock_response
            )
        )
        try:
            await engine.chat_completion(messages=[{"role": "user", "content": "hi"}])
            assert False, "Should have raised"
        except ContextOverflowError as e:
            assert "token limit" in str(e).lower() or "request too large" in str(e).lower()
        finally:
            await engine.close()

    async def test_400_non_context_raises_connection_error(self):
        """HTTP 400 with unrelated body raises regular ConnectionError."""
        from natshell.inference.remote import ContextOverflowError, RemoteEngine

        engine = RemoteEngine(base_url="http://localhost:11434", model="test")
        mock_response = httpx.Response(
            status_code=400,
            text="invalid model format",
            request=httpx.Request("POST", "http://localhost:11434/chat/completions"),
        )
        engine.client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "bad request", request=mock_response.request, response=mock_response
            )
        )
        try:
            await engine.chat_completion(messages=[{"role": "user", "content": "hi"}])
            assert False, "Should have raised"
        except ContextOverflowError:
            assert False, "Should not be ContextOverflowError"
        except ConnectionError as e:
            assert "400" in str(e)
        finally:
            await engine.close()
