"""Configuration loading and management for NatShell."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    path: str = "auto"
    hf_repo: str = "Qwen/Qwen3-4B-GGUF"
    hf_file: str = "Qwen3-4B-Q4_K_M.gguf"
    n_ctx: int = 8192
    n_threads: int = 0
    n_gpu_layers: int = 0


@dataclass
class RemoteConfig:
    url: str | None = None
    model: str = ""
    api_key: str = ""


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
