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
from natshell.platform import data_dir as _data_dir

logger = logging.getLogger(__name__)

# ─── Bundled model tiers ─────────────────────────────────────────────

BUNDLED_TIERS: dict[str, dict[str, str]] = {
    # ``hf_revision`` pins the exact HuggingFace commit the file is served
    # from (40-char SHA), so a repo that later points ``main`` elsewhere — or
    # a compromised org re-uploading under the same name — can't serve
    # different bytes. ``sha256`` is the file's digest (verified locally
    # before the model is first loaded). Together they pin the bytes.
    "light": {
        "name": "Light",
        "description": "Qwen3.5-4B (~2.7 GB, low RAM, 262K context)",
        "hf_repo": "unsloth/Qwen3.5-4B-GGUF",
        "hf_file": "Qwen3.5-4B-Q4_K_M.gguf",
        "hf_revision": "e87f176479d0855a907a41277aca2f8ee7a09523",
        "sha256": "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
    },
    "standard": {
        "name": "Standard",
        "description": "Qwen3.5-9B (~5.7 GB, general purpose, 262K context)",
        "hf_repo": "unsloth/Qwen3.5-9B-GGUF",
        "hf_file": "Qwen3.5-9B-Q4_K_M.gguf",
        "hf_revision": "3885219b6810b007914f3a7950a8d1b469d598a5",
        "sha256": "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
    },
    "enhanced": {
        "name": "Enhanced",
        "description": "Mistral Nemo 12B (~7.5 GB, 128K context)",
        "hf_repo": "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        "hf_file": "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        "hf_revision": "a2dd64a0a76ea1bdb2bb6ab6fa5496b003c7c908",
        "sha256": "7c1a10d202d8788dbe5628dc962254d10654c853cae6aaeca0618f05490d4a46",
    },
    "coder": {
        "name": "Coder (MoE)",
        "description": "Qwen3.5-35B-A3B (~22 GB, 3B-active MoE, needs 24+ GB RAM)",
        "hf_repo": "unsloth/Qwen3.5-35B-A3B-GGUF",
        "hf_file": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
        "hf_revision": "bc014a17be43adabd7066b7a86075ff935c6a4e2",
        "sha256": "3b46d1066bc91cc2d613e3bc22ce691dd77e6f0d33c9060690d24ce6de494375",
    },
    "gemma-light": {
        "name": "Gemma Light",
        "description": "Gemma 4 E2B (~1.5 GB, 128K context)",
        "hf_repo": "unsloth/gemma-4-E2B-it-GGUF",
        "hf_file": "gemma-4-E2B-it-Q4_K_M.gguf",
        "hf_revision": "0314792d7f1f7e229411f620751375812bb9faf2",
        "sha256": "740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8",
    },
    "gemma-standard": {
        "name": "Gemma Standard",
        "description": "Gemma 4 E4B (~5 GB, 128K context)",
        "hf_repo": "unsloth/gemma-4-E4B-it-GGUF",
        "hf_file": "gemma-4-E4B-it-Q4_K_M.gguf",
        "hf_revision": "bfc15c382204943c3a8fff0c750b94ae2364d7a3",
        "sha256": "85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87",
    },
    "gemma-enhanced": {
        "name": "Gemma Enhanced",
        "description": "Gemma 4 12B (~7.1 GB, 128K context)",
        "hf_repo": "unsloth/gemma-4-12b-it-GGUF",
        "hf_file": "gemma-4-12b-it-Q4_K_M.gguf",
        "hf_revision": "fc034cfff751157913579611efad8462ac1be606",
        "sha256": "0a270ec9fe6b34f4a0d33992b6135117b484ebc4766ab76b51d4ae8c457e4c42",
    },
}

MODELS_DIR = _data_dir() / "models"


def format_download_menu(models_dir: Path | None = None) -> str:
    """List bundled tiers with checkmarks for already-downloaded ones."""
    if models_dir is None:
        models_dir = MODELS_DIR
    lines = ["[bold]Bundled Model Tiers[/]\n"]
    for key, tier in BUNDLED_TIERS.items():
        downloaded = (models_dir / tier["hf_file"]).exists()
        mark = " [green]✓ downloaded[/]" if downloaded else ""
        lines.append(f"  [bold]{key:<10}[/] {tier['description']}{mark}")
    lines.append(
        "\n[dim]Use /model download <tier> to download"
        " (light, standard, enhanced, coder,"
        " gemma-light, gemma-standard, gemma-enhanced)[/]"
    )
    return "\n".join(lines)


def _sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the lowercase sha256 hex digest of a (possibly multi-GB) file.

    Reads in 1 MiB chunks so memory stays flat regardless of model size.
    """
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_file(path: Path, filename: str, expected_sha256: str) -> str:
    """Verify a downloaded model file's sha256 against the pinned digest.

    Returns the verified digest on success. Raises RuntimeError on
    mismatch — callers must not load a model whose bytes do not match
    the digest it was pinned with (supply-chain guard).
    """
    actual = _sha256_of(path)
    if actual != expected_sha256.lower():
        raise RuntimeError(
            f"Integrity check failed: sha256 mismatch for {filename} "
            f"(expected {expected_sha256}, got {actual}). "
            f"The download may be corrupted or tampered with."
        )
    return actual


async def download_bundled_model(
    tier_key: str,
    models_dir: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Download a bundled model tier via huggingface_hub.

    Args:
        tier_key: One of 'light', 'standard', 'enhanced', 'coder',
            'gemma-light', 'gemma-standard', 'gemma-enhanced'.
        models_dir: Directory to save to (default: ~/.local/share/natshell/models/).
        progress_callback: Optional callable for status messages.

    Returns:
        Path to the downloaded .gguf file.

    Raises:
        ValueError: If tier_key is not valid.
        RuntimeError: If the download fails, or the sha256 integrity check
            (when the tier defines one) does not match.
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
        import os

        from huggingface_hub import hf_hub_download

        # Disable hf_xet (Rust-based Xet Storage downloader) — its native
        # subprocess spawning triggers "bad value(s) in fds_to_keep" on
        # Python 3.14+ where the default multiprocessing start method is
        # forkserver.  Regular HTTP download works fine for GGUF files.
        prev_xet = os.environ.get("HF_HUB_DISABLE_XET")
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        try:
            path = hf_hub_download(
                repo_id=tier["hf_repo"],
                filename=tier["hf_file"],
                # Pin the exact commit if the tier has one; fall back to
                # the repo's current HEAD otherwise (still the documented
                # default, but weaker — prefer adding hf_revision).
                revision=tier.get("hf_revision") or None,
                local_dir=str(models_dir),
            )

            expected = tier.get("sha256")
            if expected:
                actual = verify_model_file(
                    Path(path), tier["hf_file"], expected
                )
                if progress_callback:
                    progress_callback(
                        f"  ↳ integrity: sha256 verified ({actual[:12]}…)"
                    )

            return path
        finally:
            if prev_xet is None:
                os.environ.pop("HF_HUB_DISABLE_XET", None)
            else:
                os.environ["HF_HUB_DISABLE_XET"] = prev_xet

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
            # resolved_main_gpu is the actual device index passed to llama;
            # main_gpu is the configured value (-1 = auto).
            resolved_idx = engine_info.resolved_main_gpu
            configured = engine_info.main_gpu
            auto_selected = configured is None or configured < 0

            if resolved_idx is not None and resolved_idx >= 0:
                selected = next((g for g in gpus if g.device_index == resolved_idx), gpus[0])
            else:
                selected = gpus[0]

            vram = f", {selected.vram_mb} MB VRAM" if selected.vram_mb else ""
            gpu_type = "discrete" if selected.is_discrete else "integrated"
            auto_tag = " (auto-selected)" if auto_selected else ""
            parts.append(f"  GPU: {selected.name} [{gpu_type}]{vram}{auto_tag}")
            if len(gpus) > 1:
                parts.append(f"  Device index: {selected.device_index} (of {len(gpus)} GPUs)")
                # Show all detected GPUs so misconfiguration is visible
                for g in gpus:
                    marker = " ◄ active" if g.device_index == selected.device_index else ""
                    g_vram = f", {g.vram_mb} MB" if g.vram_mb else ""
                    g_type = "discrete" if g.is_discrete else "integrated"
                    parts.append(
                        f"    [{g.device_index}] {g.name} [{g_type}]{g_vram}{marker}"
                    )

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
        model_dir = _data_dir() / "models"
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
