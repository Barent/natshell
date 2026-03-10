"""Model management helpers — extracted from app.py.

Pure logic for formatting model info, listing models, validating
server connectivity, resolving local model paths, and downloading
bundled model tiers.  Functions accept the data they need (config,
engine info, etc.) so they stay decoupled from the Textual app.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable

from natshell.config import NatShellConfig, save_ollama_default
from natshell.inference.engine import EngineInfo
from natshell.inference.ollama import (
    OllamaModel,
    get_model_context_length,
    list_models,
    normalize_base_url,
    ping_server,
)

logger = logging.getLogger(__name__)

# ─── Bundled model tiers ─────────────────────────────────────────────

BUNDLED_TIERS: dict[str, dict[str, str]] = {
    "light": {
        "name": "Light",
        "description": "Qwen3-4B (~2.5 GB, low RAM)",
        "hf_repo": "Qwen/Qwen3-4B-GGUF",
        "hf_file": "Qwen3-4B-Q4_K_M.gguf",
    },
    "standard": {
        "name": "Standard",
        "description": "Qwen3-8B (~5 GB, general purpose)",
        "hf_repo": "Qwen/Qwen3-8B-GGUF",
        "hf_file": "Qwen3-8B-Q4_K_M.gguf",
    },
    "enhanced": {
        "name": "Enhanced",
        "description": "Mistral Nemo 12B (~7.5 GB, 128K context)",
        "hf_repo": "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        "hf_file": "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
    },
}

MODELS_DIR = Path.home() / ".local" / "share" / "natshell" / "models"


def format_download_menu(models_dir: Path | None = None) -> str:
    """List bundled tiers with checkmarks for already-downloaded ones."""
    if models_dir is None:
        models_dir = MODELS_DIR
    lines = ["[bold]Bundled Model Tiers[/]\n"]
    for key, tier in BUNDLED_TIERS.items():
        downloaded = (models_dir / tier["hf_file"]).exists()
        mark = " [green]✓ downloaded[/]" if downloaded else ""
        lines.append(f"  [bold]{key:<10}[/] {tier['description']}{mark}")
    lines.append("\n[dim]Use /model download <tier> to download (light, standard, enhanced)[/]")
    return "\n".join(lines)


async def download_bundled_model(
    tier_key: str,
    models_dir: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Download a bundled model tier via huggingface_hub.

    Args:
        tier_key: One of 'light', 'standard', 'enhanced'.
        models_dir: Directory to save to (default: ~/.local/share/natshell/models/).
        progress_callback: Optional callable for status messages.

    Returns:
        Path to the downloaded .gguf file.

    Raises:
        ValueError: If tier_key is not valid.
        RuntimeError: If the download fails.
    """
    if tier_key not in BUNDLED_TIERS:
        raise ValueError(
            f"Unknown tier '{tier_key}'. Valid tiers: {', '.join(BUNDLED_TIERS)}"
        )

    if models_dir is None:
        models_dir = MODELS_DIR

    tier = BUNDLED_TIERS[tier_key]
    models_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(f"Downloading {tier['name']} ({tier['description']})...")

    def _do_download() -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=tier["hf_repo"],
            filename=tier["hf_file"],
            local_dir=str(models_dir),
        )

    try:
        path_str = await asyncio.to_thread(_do_download)
    except Exception as e:
        logger.error("Model download failed: %s", e)
        raise RuntimeError(f"Download failed: {e}") from e

    return Path(path_str)

# ─── URL resolution ─────────────────────────────────────────────────

def get_remote_base_url(config: NatShellConfig, engine_info: EngineInfo) -> str | None:
    """Find the remote base URL from config or current engine.

    Checks ollama.url, then remote.url, then the engine's base_url.
    Returns a normalised URL or None if nothing is configured.
    """
    if config.ollama.url:
        return normalize_base_url(config.ollama.url)
    if config.remote.url:
        return normalize_base_url(config.remote.url)
    if engine_info.base_url:
        return normalize_base_url(engine_info.base_url)
    return None


# ─── /model (info) ──────────────────────────────────────────────────

def format_model_info(engine_info: EngineInfo, config: NatShellConfig) -> str:
    """Build the model-info display text shown by ``/model``."""
    parts = ["[bold]Model Info[/]"]
    parts.append(f"  Engine: {engine_info.engine_type}")
    if engine_info.model_name:
        parts.append(f"  Model: {engine_info.model_name}")
    if engine_info.base_url:
        parts.append(f"  URL: {engine_info.base_url}")
    if engine_info.n_ctx:
        parts.append(f"  Context: {engine_info.n_ctx} tokens")
    if engine_info.n_gpu_layers:
        parts.append(f"  GPU layers: {engine_info.n_gpu_layers}")
    if engine_info.engine_type == "local":
        try:
            from llama_cpp import llama_supports_gpu_offload

            gpu_ok = llama_supports_gpu_offload()
        except ImportError:
            gpu_ok = False
        status = "[green]active[/]" if gpu_ok else "[red]unavailable (CPU-only build)[/]"
        parts.append(f"  GPU backend: {status}")

        # Show detected GPU hardware
        from natshell.gpu import detect_gpus

        gpus = detect_gpus()
        if gpus:
            resolved = engine_info.resolved_main_gpu if engine_info.resolved_main_gpu is not None else engine_info.main_gpu
            if resolved is not None and resolved >= 0:
                # Find the selected GPU by device index
                selected = next(
                    (g for g in gpus if g.device_index == resolved), gpus[0]
                )
            else:
                selected = gpus[0]
            vram = f", {selected.vram_mb} MB VRAM" if selected.vram_mb else ""
            parts.append(f"  GPU: {selected.name}{vram}")
            if len(gpus) > 1:
                parts.append(f"  Device index: {selected.device_index} (of {len(gpus)} GPUs)")

    remote_url = get_remote_base_url(config, engine_info)
    if remote_url:
        parts.append("\n[dim]Tip: /model list — see available remote models[/]")
    return "\n".join(parts)


# ─── /model list ─────────────────────────────────────────────────────

async def fetch_model_list(
    base_url: str,
) -> tuple[bool, list[OllamaModel] | None, str | None]:
    """Ping the remote server and fetch its model list.

    Returns ``(reachable, models, error_message)``.
    *models* is ``None`` when the server is unreachable or returns nothing.
    """
    reachable = await ping_server(base_url)
    if not reachable:
        return False, None, f"[red]Cannot reach server at {base_url}[/]"

    models = await list_models(base_url)
    if not models:
        return True, None, "Server is running but returned no models."

    return True, models, None


def format_model_list(models: list[OllamaModel], current_model_name: str) -> str:
    """Format a list of remote models for display."""
    lines = ["[bold]Available Models[/]"]
    for m in models:
        marker = " [green]◀ active[/]" if m.name == current_model_name else ""
        detail = ""
        if m.size_gb:
            detail += f" ({m.size_gb} GB)"
        if m.parameter_size:
            detail += f" [{m.parameter_size}]"
        lines.append(f"  {m.name}{detail}{marker}")
    lines.append("\n[dim]Use /model use <name> to switch[/]")
    return "\n".join(lines)


# ─── /model use ──────────────────────────────────────────────────────

async def prepare_remote_engine_params(
    model_name: str,
    base_url: str,
    config: NatShellConfig,
) -> tuple[str, int]:
    """Resolve the API URL and context length for a remote model switch.

    Returns ``(api_url, n_ctx)``.
    """
    api_url = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
    if config.remote.n_ctx > 0:
        n_ctx = config.remote.n_ctx
    elif config.ollama.n_ctx > 0:
        n_ctx = config.ollama.n_ctx
    else:
        n_ctx = await get_model_context_length(base_url, model_name)
    return api_url, n_ctx


# ─── /model switch ───────────────────────────────────────────────────

def format_local_models(available: list[Path], current_model_name: str) -> str:
    """Format the list of local .gguf models for display."""
    lines = ["[bold]Local Models[/]"]
    for gguf in available:
        marker = " [green]◀ active[/]" if gguf.name == current_model_name else ""
        lines.append(f"  {gguf.name}{marker}")
    lines.append("\n[dim]Use /model switch <filename> to switch[/]")
    return "\n".join(lines)


def find_local_model(name: str, available: list[Path]) -> Path | None:
    """Find a local .gguf model by filename or stem.

    Returns the matching ``Path`` or ``None``.
    """
    for gguf in available:
        if gguf.name == name or gguf.stem == name:
            return gguf
    return None


# ─── /model local ────────────────────────────────────────────────────

def resolve_local_model_path(config: NatShellConfig) -> str:
    """Resolve the local model path from config, expanding 'auto'."""
    mc = config.model
    model_path = mc.path
    if model_path == "auto":
        model_dir = Path.home() / ".local" / "share" / "natshell" / "models"
        model_path = str(model_dir / mc.hf_file)
    return model_path


# ─── /model default ──────────────────────────────────────────────────

def set_default_model(
    model_name: str, config: NatShellConfig, engine_info: EngineInfo
) -> str:
    """Persist the default model to user config and return display text."""
    remote_url = get_remote_base_url(config, engine_info)
    config_path = save_ollama_default(model_name, url=remote_url)
    parts = [f"Default model set to [bold]{model_name}[/]"]
    if remote_url:
        parts.append(f"Server: {remote_url}")
    parts.append(f"Saved to {config_path}")
    return "\n".join(parts)
