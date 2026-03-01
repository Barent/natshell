"""Model management helpers — extracted from app.py.

Pure logic for formatting model info, listing models, validating
server connectivity, and resolving local model paths.  Functions
accept the data they need (config, engine info, etc.) so they stay
decoupled from the Textual app.
"""

from __future__ import annotations

from pathlib import Path

from natshell.config import NatShellConfig, save_ollama_default
from natshell.inference.engine import EngineInfo
from natshell.inference.ollama import (
    OllamaModel,
    get_model_context_length,
    list_models,
    normalize_base_url,
    ping_server,
)

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
            if engine_info.main_gpu is not None and engine_info.main_gpu >= 0:
                # Find the selected GPU by device index
                selected = next(
                    (g for g in gpus if g.device_index == engine_info.main_gpu), gpus[0]
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
