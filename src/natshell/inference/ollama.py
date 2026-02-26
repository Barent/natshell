"""Remote API client for Ollama and OpenAI-compatible servers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class OllamaModel:
    """A model available on a remote server."""

    name: str
    size_gb: float = 0.0
    parameter_size: str = ""
    family: str = ""


def normalize_base_url(url: str) -> str:
    """Strip /v1 suffix and trailing slash to get the server root URL.

    >>> normalize_base_url("http://localhost:11434/v1")
    'http://localhost:11434'
    >>> normalize_base_url("http://localhost:11434/v1/")
    'http://localhost:11434'
    >>> normalize_base_url("http://localhost:11434")
    'http://localhost:11434'
    """
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


async def ping_server(base_url: str) -> bool:
    """Check if a server is reachable. Returns True if we get any 200 response."""
    base_url = normalize_base_url(base_url)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/")
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, OSError):
        return False


async def list_models(base_url: str) -> list[OllamaModel]:
    """List available models from a remote server.

    Tries Ollama native /api/tags first, then falls back to OpenAI /v1/models.
    Returns [] on failure.
    """
    base_url = normalize_base_url(base_url)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try Ollama native endpoint first
            try:
                resp = await client.get(f"{base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    return _parse_ollama_models(data)
            except (httpx.HTTPError, ValueError, KeyError):
                pass

            # Fall back to OpenAI-compatible endpoint
            try:
                resp = await client.get(f"{base_url}/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    return _parse_openai_models(data)
            except (httpx.HTTPError, ValueError, KeyError):
                pass

    except (httpx.ConnectError, httpx.ConnectTimeout, OSError):
        pass

    return []


def _parse_ollama_models(data: dict) -> list[OllamaModel]:
    """Parse Ollama /api/tags response."""
    models = []
    for m in data.get("models", []):
        details = m.get("details", {})
        size_bytes = m.get("size", 0)
        models.append(OllamaModel(
            name=m.get("name", ""),
            size_gb=round(size_bytes / (1024**3), 1) if size_bytes else 0.0,
            parameter_size=details.get("parameter_size", ""),
            family=details.get("family", ""),
        ))
    return models


def _parse_openai_models(data: dict) -> list[OllamaModel]:
    """Parse OpenAI /v1/models response."""
    models = []
    for m in data.get("data", []):
        models.append(OllamaModel(
            name=m.get("id", ""),
        ))
    return models
