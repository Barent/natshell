"""Configuration loading and management for NatShell."""

from __future__ import annotations

import logging
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    path: str = "auto"
    hf_repo: str = "Qwen/Qwen3-4B-GGUF"
    hf_file: str = "Qwen3-4B-Q4_K_M.gguf"
    n_ctx: int = 0  # 0 = auto (inferred from model size)
    n_threads: int = 0
    n_gpu_layers: int = -1
    main_gpu: int = -1  # -1 = auto-detect best GPU
    prompt_cache: bool = True  # Enable llama-cpp-python RAM prompt cache
    prompt_cache_mb: int = 256  # Cache capacity in megabytes


@dataclass
class RemoteConfig:
    url: str | None = None
    model: str = ""
    api_key: str = ""
    n_ctx: int = 0  # 0 = auto (query server), >0 = override


@dataclass
class OllamaConfig:
    url: str = ""
    default_model: str = ""
    n_ctx: int = 0  # 0 = auto (query server), >0 = override


@dataclass
class AgentConfig:
    max_steps: int = 15
    plan_max_steps: int = 35  # Higher limit for plan execution steps
    temperature: float = 0.3
    max_tokens: int = 8192
    context_reserve: int = 0  # Extra tokens to reserve (0 = auto ~800 tokens)


@dataclass
class SafetyConfig:
    mode: str = "confirm"
    always_confirm: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)


@dataclass
class UIConfig:
    theme: str = "dark"


@dataclass
class BackupConfig:
    enabled: bool = True
    max_per_file: int = 10


@dataclass
class EngineConfig:
    preferred: str = "auto"  # "auto", "local", or "remote"


@dataclass
class McpConfig:
    safety_mode: str = "strict"  # "strict" (confirm->error) or "permissive" (confirm->auto-approve)


@dataclass
class KiwixConfig:
    url: str = "http://localhost:8080"


@dataclass
class MemoryConfig:
    enabled: bool = True
    max_chars: int = 4000    # ~1000 tokens
    min_ctx: int = 16384     # Skip memory injection below this n_ctx


@dataclass
class PromptConfig:
    extra_instructions: str = ""
    persona: str = ""


@dataclass
class ProfileConfig:
    """A named configuration profile that can override settings across sections."""
    # Ollama/remote
    ollama_model: str = ""      # → ollama.default_model
    ollama_url: str = ""        # → ollama.url
    remote_url: str = ""        # → remote.url
    remote_model: str = ""      # → remote.model
    api_key: str = ""           # → remote.api_key
    # Context and inference
    n_ctx: int = 0              # → ollama.n_ctx or remote.n_ctx
    temperature: float = 0.0    # → agent.temperature (0.0 = don't override)
    # Engine
    engine: str = ""            # → engine.preferred ("local"/"remote")
    # Local model
    n_gpu_layers: int = -2      # → model.n_gpu_layers (-2 = don't override)


@dataclass
class NatShellConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    kiwix: KiwixConfig = field(default_factory=KiwixConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    profiles: dict[str, ProfileConfig] = field(default_factory=dict)


# ── Valid config keys (section → {key: type_string}) ─────────────────────

VALID_CONFIG_KEYS: dict[str, dict[str, str]] = {
    "model": {
        "path": "str",
        "hf_repo": "str",
        "hf_file": "str",
        "n_ctx": "int",
        "n_threads": "int",
        "n_gpu_layers": "int",
        "main_gpu": "int",
        "prompt_cache": "bool",
        "prompt_cache_mb": "int",
    },
    "remote": {
        "url": "str",
        "model": "str",
        "api_key": "str",
        "n_ctx": "int",
    },
    "ollama": {
        "url": "str",
        "default_model": "str",
        "n_ctx": "int",
    },
    "agent": {
        "max_steps": "int",
        "plan_max_steps": "int",
        "temperature": "float",
        "max_tokens": "int",
        "context_reserve": "int",
    },
    "safety": {
        "mode": "str",
    },
    "ui": {
        "theme": "str",
    },
    "backup": {
        "enabled": "bool",
        "max_per_file": "int",
    },
    "engine": {
        "preferred": "str",
    },
    "mcp": {
        "safety_mode": "str",
    },
    "kiwix": {
        "url": "str",
    },
    "prompt": {
        "extra_instructions": "str",
        "persona": "str",
    },
    "memory": {
        "enabled": "bool",
        "max_chars": "int",
        "min_ctx": "int",
    },
}

CONFIG_ENUMS: dict[str, dict[str, list[str]]] = {
    "safety": {
        "mode": ["confirm", "warn", "danger"],
    },
    "engine": {
        "preferred": ["auto", "local", "remote"],
    },
    "mcp": {
        "safety_mode": ["strict", "permissive"],
    },
    "ui": {
        "theme": ["dark", "light"],
    },
}


def save_config_value(section: str, key: str, value: str | int | float | bool) -> Path:
    """Persist a single config value to the user config file.

    Uses simple line-based TOML editing (same pattern as save_engine_preference).
    Returns the path to the config file.
    """
    config_dir = Path.home() / ".config" / "natshell"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"

    if config_path.exists():
        lines = config_path.read_text().splitlines(keepends=True)
    else:
        lines = []

    # Format value as TOML
    if isinstance(value, bool):
        val_str = "true" if value else "false"
    elif isinstance(value, str):
        val_str = f'"{value}"'
    else:
        val_str = str(value)

    section_header = f"[{section}]"
    section_idx = None
    next_section_idx = None
    key_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == section_header:
            section_idx = i
        elif section_idx is not None and next_section_idx is None:
            if re.match(r"^\[.+\]", stripped):
                next_section_idx = i
            elif stripped.startswith(key) or stripped.startswith(f"# {key}"):
                # Verify this is the actual key, not a prefix match
                if re.match(rf"^#?\s*{re.escape(key)}\s*=", stripped):
                    key_idx = i

    new_line = f"{key} = {val_str}\n"

    if section_idx is not None:
        insert_at = next_section_idx if next_section_idx is not None else len(lines)
        if key_idx is not None:
            lines[key_idx] = new_line
        else:
            lines.insert(insert_at, new_line)
    else:
        if lines and not lines[-1].endswith("\n"):
            lines.append("\n")
        lines.append(f"\n{section_header}\n")
        lines.append(new_line)

    config_path.write_text("".join(lines))
    return config_path


def load_config(config_path: str | Path | None = None) -> NatShellConfig:
    """Load configuration from TOML file, falling back to defaults.

    Search order:
    1. Explicit config_path argument
    2. ~/.config/natshell/config.toml
    3. Built-in defaults
    """
    config = NatShellConfig()

    # Load defaults from bundled config
    default_path = Path(__file__).parent / "config.default.toml"
    if default_path.exists():
        _merge_toml(config, default_path)

    # Load user config
    if config_path:
        user_path = Path(config_path)
    else:
        user_path = Path.home() / ".config" / "natshell" / "config.toml"

    if user_path.exists():
        _merge_toml(config, user_path)

    # Support NATSHELL_API_KEY environment variable as alternative to config file
    env_api_key = os.environ.get("NATSHELL_API_KEY")
    if env_api_key:
        config.remote.api_key = env_api_key

    # Warn if config file contains an API key and has permissive permissions
    if config.remote.api_key and user_path.exists():
        try:
            perms = user_path.stat().st_mode & 0o777
            if perms & 0o077:
                logger.warning(
                    "Config file %s has permissive permissions (%04o) and contains an API key. "
                    "Run: chmod 600 %s",
                    user_path,
                    perms,
                    user_path,
                )
        except OSError:
            pass

    return config


_SECTIONS = (
    "model", "remote", "ollama", "agent", "safety",
    "ui", "backup", "engine", "mcp", "kiwix", "prompt", "memory",
)


def _merge_toml(config: NatShellConfig, path: Path) -> None:
    """Merge a TOML file into the config, overwriting only specified fields."""
    with open(path, "rb") as f:
        data = tomllib.load(f)

    for section_name in _SECTIONS:
        if section_name in data:
            section_obj = getattr(config, section_name, None)
            if section_obj is None:
                continue
            for key, value in data[section_name].items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)

    if "profiles" in data:
        for name, profile_data in data["profiles"].items():
            if isinstance(profile_data, dict):
                profile = ProfileConfig()
                for key, value in profile_data.items():
                    if hasattr(profile, key):
                        setattr(profile, key, value)
                config.profiles[name] = profile


def save_config_values(
    section: str, values: dict[str, str | int | float | bool]
) -> Path:
    """Persist multiple config values in one section. Returns the config path."""
    path = None
    for key, value in values.items():
        path = save_config_value(section, key, value)
    return path  # type: ignore[return-value]


def save_ollama_default(model_name: str, url: str | None = None) -> Path:
    """Persist the default Ollama model (and optionally URL) to user config."""
    values: dict[str, str | int | float | bool] = {"default_model": model_name}
    if url:
        values["url"] = url
    return save_config_values("ollama", values)


def save_model_config(hf_repo: str, hf_file: str) -> Path:
    """Persist the default local model (hf_repo / hf_file) to user config."""
    return save_config_values("model", {"hf_repo": hf_repo, "hf_file": hf_file})


def save_engine_preference(preferred: str) -> Path:
    """Persist the engine preference ("local", "remote", or "auto") to user config."""
    return save_config_value("engine", "preferred", preferred)


def list_profiles(config: NatShellConfig) -> list[str]:
    """Return available profile names."""
    return sorted(config.profiles.keys())


def apply_profile(config: NatShellConfig, name: str) -> None:
    """Apply a named profile to the config, overriding only non-default values.

    Raises KeyError if the profile name is not found.
    """
    if name not in config.profiles:
        raise KeyError(f"Unknown profile: {name}")

    profile = config.profiles[name]

    if profile.ollama_model:
        config.ollama.default_model = profile.ollama_model
    if profile.ollama_url:
        config.ollama.url = profile.ollama_url
    if profile.remote_url:
        config.remote.url = profile.remote_url
    if profile.remote_model:
        config.remote.model = profile.remote_model
    if profile.api_key:
        config.remote.api_key = profile.api_key
    if profile.n_ctx:
        config.ollama.n_ctx = profile.n_ctx
        config.remote.n_ctx = profile.n_ctx
    if profile.temperature:
        config.agent.temperature = profile.temperature
    if profile.engine:
        config.engine.preferred = profile.engine
    if profile.n_gpu_layers != -2:
        config.model.n_gpu_layers = profile.n_gpu_layers
