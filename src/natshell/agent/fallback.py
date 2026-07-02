"""Remote-engine failure classification and local-fallback engine loading."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def describe_remote_error(error: Exception) -> str:
    """Build a short user-facing label for a remote inference failure.

    Returns "Remote server timed out" when the underlying cause was a
    timeout (so the existing phrasing is preserved for the common case),
    or "Remote server error: {message}" otherwise — surfacing the actual
    exception text (e.g. "Remote API error 500: model runner has
    unexpectedly stopped…") instead of falsely claiming a timeout.
    """
    import httpx

    timeout_types = (
        httpx.ReadTimeout,
        httpx.PoolTimeout,
        httpx.ConnectTimeout,
    )
    if isinstance(error, timeout_types):
        return "Remote server timed out"
    cause = getattr(error, "__cause__", None)
    if isinstance(cause, timeout_types):
        return "Remote server timed out"
    message = str(error) or error.__class__.__name__
    return f"Remote server error: {message}"


def can_fallback(error: Exception, engine: Any, fallback_config: Any) -> bool:
    """Check if we should attempt fallback to local model."""
    import httpx

    from natshell.inference.remote import (
        AuthenticationError,
        ContextOverflowError,
        RemoteEngine,
    )

    # Context overflow is not a connectivity issue — don't swap engines
    if isinstance(error, ContextOverflowError):
        return False
    # Auth errors mean the server is reachable but the key is wrong — don't swap
    if isinstance(error, AuthenticationError):
        return False
    if not isinstance(engine, RemoteEngine):
        return False
    if fallback_config is None:
        return False
    return isinstance(
        error,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.PoolTimeout,
            httpx.RemoteProtocolError,
            ConnectionError,
            OSError,
        ),
    )


async def load_fallback_engine(fallback_config: Any):
    """Load a local engine for fallback. Returns the engine or None on failure."""
    if fallback_config is None:
        return None

    # Resolve model path
    model_path = fallback_config.path
    if model_path == "auto":
        from natshell.platform import data_dir

        model_dir = data_dir() / "models"
        model_path = str(model_dir / fallback_config.hf_file)

    if not Path(model_path).exists():
        logger.warning("Local model not found at %s — cannot fall back", model_path)
        return None

    try:
        from natshell.inference.local import LocalEngine

        engine = await asyncio.to_thread(
            LocalEngine,
            model_path=model_path,
            n_ctx=fallback_config.n_ctx,
            n_threads=fallback_config.n_threads,
            n_gpu_layers=fallback_config.n_gpu_layers,
            main_gpu=fallback_config.main_gpu,
        )

        # Warn if GPU offload was requested but unavailable
        if fallback_config.n_gpu_layers != 0:
            try:
                from llama_cpp import llama_supports_gpu_offload

                if not llama_supports_gpu_offload():
                    logger.warning(
                        "Fallback model running on CPU — llama-cpp-python"
                        " was built without GPU support. Reinstall with"
                        ' CMAKE_ARGS="-DGGML_VULKAN=on" for GPU acceleration.'
                    )
            except ImportError:
                pass

        return engine
    except Exception:
        logger.exception("Failed to load local model for fallback")
        return None
