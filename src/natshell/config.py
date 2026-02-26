"""Configuration loading and management for NatShell."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    path: str = "auto"
    hf_repo: str = "Qwen/Qwen3-4B-GGUF"
    hf_file: str = "Qwen3-4B-Q4_K_M.gguf"
    n_ctx: int = 0  # 0 = auto (inferred from model size)
    n_threads: int = 0
    n_gpu_layers: int = -1


@dataclass
class RemoteConfig:
    url: str | None = None
    model: str = ""
    api_key: str = ""


@dataclass
class OllamaConfig:
    url: str = ""
    default_model: str = ""


@dataclass
class AgentConfig:
    max_steps: int = 15
    temperature: float = 0.3
    max_tokens: int = 2048


@dataclass
class SafetyConfig:
    mode: str = "confirm"
    always_confirm: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)


@dataclass
class UIConfig:
    theme: str = "dark"


@dataclass
class NatShellConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    ui: UIConfig = field(default_factory=UIConfig)


def load_config(config_path: str | Path | None = None) -> NatShellConfig:
    """Load configuration from TOML file, falling back to defaults.

    Search order:
    1. Explicit config_path argument
    2. ~/.config/natshell/config.toml
    3. Built-in defaults
    """
    config = NatShellConfig()

    # Load defaults from bundled config
    default_path = Path(__file__).parent.parent.parent / "config.default.toml"
    if default_path.exists():
        _merge_toml(config, default_path)

    # Load user config
    if config_path:
        user_path = Path(config_path)
    else:
        user_path = Path.home() / ".config" / "natshell" / "config.toml"

    if user_path.exists():
        _merge_toml(config, user_path)

    return config


def _merge_toml(config: NatShellConfig, path: Path) -> None:
    """Merge a TOML file into the config, overwriting only specified fields."""
    with open(path, "rb") as f:
        data = tomllib.load(f)

    if "model" in data:
        for key, value in data["model"].items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)

    if "remote" in data:
        for key, value in data["remote"].items():
            if hasattr(config.remote, key):
                setattr(config.remote, key, value)

    if "ollama" in data:
        for key, value in data["ollama"].items():
            if hasattr(config.ollama, key):
                setattr(config.ollama, key, value)

    if "agent" in data:
        for key, value in data["agent"].items():
            if hasattr(config.agent, key):
                setattr(config.agent, key, value)

    if "safety" in data:
        for key, value in data["safety"].items():
            if hasattr(config.safety, key):
                setattr(config.safety, key, value)

    if "ui" in data:
        for key, value in data["ui"].items():
            if hasattr(config.ui, key):
                setattr(config.ui, key, value)


def save_ollama_default(model_name: str, url: str | None = None) -> Path:
    """Persist the default Ollama model (and optionally URL) to user config.

    Uses simple line-based TOML editing to avoid a tomli_w dependency.
    Returns the path to the config file.
    """
    config_dir = Path.home() / ".config" / "natshell"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"

    if config_path.exists():
        lines = config_path.read_text().splitlines(keepends=True)
    else:
        lines = []

    # Find [ollama] section and locate existing keys
    ollama_idx = None
    next_section_idx = None
    default_model_idx = None
    url_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[ollama]":
            ollama_idx = i
        elif ollama_idx is not None and next_section_idx is None:
            if re.match(r"^\[.+\]", stripped):
                next_section_idx = i
            elif stripped.startswith("default_model") or stripped.startswith("# default_model"):
                default_model_idx = i
            elif stripped.startswith("url") or stripped.startswith("# url"):
                url_idx = i

    model_line = f'default_model = "{model_name}"\n'
    url_line = f'url = "{url}"\n' if url else None

    if ollama_idx is not None:
        # Section exists — update or insert keys
        # Insert point: end of section or before next section
        insert_at = next_section_idx if next_section_idx is not None else len(lines)

        if default_model_idx is not None:
            lines[default_model_idx] = model_line
        else:
            lines.insert(insert_at, model_line)
            # Adjust indices after insertion
            if url_idx is not None and url_idx >= insert_at:
                url_idx += 1
            insert_at += 1

        if url_line:
            if url_idx is not None:
                lines[url_idx] = url_line
            else:
                # Insert URL before default_model for readability
                target = default_model_idx if default_model_idx is not None else insert_at
                lines.insert(target, url_line)
    else:
        # No [ollama] section — add one at the end
        if lines and not lines[-1].endswith("\n"):
            lines.append("\n")
        lines.append("\n[ollama]\n")
        if url_line:
            lines.append(url_line)
        lines.append(model_line)

    config_path.write_text("".join(lines))
    return config_path


def save_model_config(hf_repo: str, hf_file: str) -> Path:
    """Persist the default local model (hf_repo / hf_file) to user config.

    Uses simple line-based TOML editing to avoid a tomli_w dependency.
    Returns the path to the config file.
    """
    config_dir = Path.home() / ".config" / "natshell"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"

    if config_path.exists():
        lines = config_path.read_text().splitlines(keepends=True)
    else:
        lines = []

    # Find [model] section and locate existing keys
    model_idx = None
    next_section_idx = None
    hf_repo_idx = None
    hf_file_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[model]":
            model_idx = i
        elif model_idx is not None and next_section_idx is None:
            if re.match(r"^\[.+\]", stripped):
                next_section_idx = i
            elif stripped.startswith("hf_repo") or stripped.startswith("# hf_repo"):
                hf_repo_idx = i
            elif stripped.startswith("hf_file") or stripped.startswith("# hf_file"):
                hf_file_idx = i

    repo_line = f'hf_repo = "{hf_repo}"\n'
    file_line = f'hf_file = "{hf_file}"\n'

    if model_idx is not None:
        # Section exists — update or insert keys
        insert_at = next_section_idx if next_section_idx is not None else len(lines)

        if hf_repo_idx is not None:
            lines[hf_repo_idx] = repo_line
        else:
            lines.insert(insert_at, repo_line)
            if hf_file_idx is not None and hf_file_idx >= insert_at:
                hf_file_idx += 1
            insert_at += 1

        if hf_file_idx is not None:
            lines[hf_file_idx] = file_line
        else:
            lines.insert(insert_at, file_line)
    else:
        # No [model] section — add one at the end
        if lines and not lines[-1].endswith("\n"):
            lines.append("\n")
        lines.append("\n[model]\n")
        lines.append(repo_line)
        lines.append(file_line)

    config_path.write_text("".join(lines))
    return config_path
