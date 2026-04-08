"""First-run setup wizard for NatShell.

Guides users through model tier selection and writes initial config.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, TextIO

from natshell.model_manager import BUNDLED_TIERS

# Wizard choices: numbered keys map to BUNDLED_TIERS + non-download options.
MODEL_TIERS: dict[str, dict[str, str]] = {
    "1": BUNDLED_TIERS["light"],
    "2": BUNDLED_TIERS["standard"],
    "3": BUNDLED_TIERS["enhanced"],
    "4": BUNDLED_TIERS["gemma"],
    "5": {
        "name": "Remote only",
        "description": "Use an Ollama/remote server (no local download)",
        "hf_repo": "",
        "hf_file": "",
    },
    "6": {
        "name": "Skip",
        "description": "Configure later",
        "hf_repo": "",
        "hf_file": "",
    },
}


def _check_llama_available() -> bool:
    """Return True if llama-cpp-python can be imported."""
    try:
        import llama_cpp  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _write_ollama_config(config_path: Path) -> None:
    """Write a minimal config that points to a local Ollama server."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "[engine]",
        'preferred = "remote"',
        "",
        "[ollama]",
        'url = "http://localhost:11434"',
        'default_model = "qwen3:4b"',
        "",
    ]
    config_path.write_text("\n".join(lines) + "\n")
    try:
        os.chmod(config_path, 0o600)
    except OSError:
        pass


def _detect_gpu_info() -> tuple[str, bool]:
    """Detect GPU info for display in the wizard.

    Returns (description_string, gpu_backend_available).
    """
    try:
        from natshell.gpu import detect_gpus, gpu_backend_available

        gpus = detect_gpus()
        backend = gpu_backend_available()
        if gpus:
            gpu = gpus[0]
            vram = f" ({gpu.vram_mb} MB VRAM)" if gpu.vram_mb else ""
            return f"{gpu.name}{vram}", backend
        return "", backend
    except Exception:
        return "", False


def _write_initial_config(
    config_path: Path,
    hf_repo: str,
    hf_file: str,
    *,
    n_gpu_layers: int = -1,
) -> None:
    """Write a minimal initial config.toml with chmod 0o600."""
    config_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    if hf_repo and hf_file:
        lines.append("[model]")
        lines.append(f'hf_repo = "{hf_repo}"')
        lines.append(f'hf_file = "{hf_file}"')
        lines.append(f"n_gpu_layers = {n_gpu_layers}")
        lines.append("")

    config_path.write_text("\n".join(lines) + "\n" if lines else "")

    # Secure permissions — config may later contain API keys
    try:
        os.chmod(config_path, 0o600)
    except OSError:
        pass


def run_setup_wizard(
    output: TextIO = sys.stdout,
    input_fn: Callable[[str], str] = input,
) -> Path | None:
    """Run the interactive first-run setup wizard.

    Args:
        output: Where to print prompts (default: stdout).
        input_fn: Callable for reading user input (default: builtin input).

    Returns:
        Path to the written config file, or None if skipped.
    """
    from natshell.platform import config_dir as _config_dir

    cfg_dir = _config_dir()
    config_path = cfg_dir / "config.toml"

    output.write("\n")
    output.write("  ─── NatShell First-Run Setup ───\n")
    output.write("\n")

    # Check if llama-cpp-python is available
    llama_available = _check_llama_available()

    # GPU detection
    gpu_desc, gpu_backend = _detect_gpu_info()
    if gpu_desc:
        output.write(f"  GPU detected: {gpu_desc}\n")
        if llama_available and not gpu_backend:
            output.write(
                "  ⚠ llama-cpp-python was built without GPU support.\n"
                "    Reinstall with GPU flags for hardware"
                " acceleration.\n"
            )
        output.write("\n")

    # On Windows, if llama-cpp-python isn't installed, steer toward Ollama
    from natshell.platform import is_windows

    if is_windows() and not llama_available:
        output.write(
            "  ⚠ llama-cpp-python is not installed — local models"
            " will not work.\n"
            "    Ollama is recommended for Windows.\n"
            "\n"
        )
        default_choice = "5"
    else:
        default_choice = "2"

    output.write("  Select a model tier:\n")
    output.write("\n")
    local_tag = " (requires llama-cpp-python)" if not llama_available else ""
    output.write(
        f"    1) Light       — Qwen3-4B        (~2.5 GB, low RAM)"
        f"{local_tag}\n"
    )
    rec = " ★ Recommended" if llama_available else local_tag
    output.write(
        f"    2) Standard    — Qwen3-8B        (~5 GB, general purpose)"
        f"{rec}\n"
    )
    output.write(
        f"    3) Enhanced    — Mistral Nemo 12B (~7.5 GB, 128K context)"
        f"{local_tag}\n"
    )
    output.write(
        f"    4) Gemma 4     — Gemma 4 E4B     (~5 GB, 128K context)"
        f"{local_tag}\n"
    )
    ollama_rec = " ★ Recommended" if not llama_available else ""
    output.write(
        f"    5) Remote only — use an Ollama server"
        f" (no local download){ollama_rec}\n"
    )
    output.write("    6) Skip        — configure later\n")
    output.write("\n")

    try:
        choice = input_fn(f"  Choice [{default_choice}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        output.write("\n")
        return None

    if not choice:
        choice = default_choice

    if choice not in MODEL_TIERS:
        output.write(
            f"  Invalid choice '{choice}',"
            f" defaulting to option {default_choice}\n"
        )
        choice = default_choice

    tier = MODEL_TIERS[choice]
    output.write(f"  Selected: {tier['name']} — {tier['description']}\n")

    if choice in ("5", "6"):
        # Remote-only or skip — don't write model config
        if choice == "5":
            output.write(
                "\n"
                "  To get started with Ollama:\n"
                "    1. Install from https://ollama.com\n"
                "    2. ollama pull qwen3:4b\n"
                "    3. Run natshell again — it will auto-detect"
                " the Ollama server.\n"
                "\n"
                "  Or configure manually in"
                f" {config_path}\n"
            )
            # Write Ollama config so natshell auto-connects
            _write_ollama_config(config_path)
            output.write(f"  Config written to {config_path}\n")
            return config_path
        return None

    # Warn before downloading if llama-cpp-python is missing
    if not llama_available:
        output.write(
            "\n"
            "  ⚠ Warning: llama-cpp-python is not installed.\n"
            "    The model will download, but natshell cannot"
            " run it yet.\n"
            "    You will need to install llama-cpp-python or"
            " switch to Ollama.\n"
            "\n"
        )
        try:
            proceed = input_fn("  Continue with download? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            output.write("\n")
            return None
        if proceed != "y":
            output.write("  Skipping download.\n")
            return None

    # Write config
    _write_initial_config(
        config_path,
        tier["hf_repo"],
        tier["hf_file"],
    )

    output.write(f"  Config written to {config_path}\n")
    return config_path


def should_run_wizard(
    config_path: Path | None = None,
    *,
    no_setup: bool = False,
    headless: bool = False,
    mcp: bool = False,
    download: bool = False,
) -> bool:
    """Determine whether the setup wizard should run.

    Returns True when:
    - No user config exists
    - Not suppressed by --no-setup, --headless, --mcp, or --download
    """
    if no_setup or headless or mcp or download:
        return False

    if config_path is None:
        from natshell.platform import config_dir
        config_path = config_dir() / "config.toml"

    return not config_path.exists()
