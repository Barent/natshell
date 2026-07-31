"""Configuration loading and management for NatShell."""

from __future__ import annotations

import logging
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from natshell.platform import config_dir as _platform_config_dir

logger = logging.getLogger(__name__)


def _get_config_dir() -> Path:
    """Return the config directory, delegating to the platform helper."""
    return _platform_config_dir()


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
    danger_fast: bool = False


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
class SkillsConfig:
    enabled: bool = True
    disabled: list[str] = field(default_factory=list)
    inject_in_compact: bool = False


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
    skills: SkillsConfig = field(default_factory=SkillsConfig)
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
        "danger_fast": "bool",
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
    "skills": {
        "enabled": "bool",
        "disabled": "list",
        "inject_in_compact": "bool",
    },
}

# The subset of VALID_CONFIG_KEYS that the model may write through the
# update_config tool.  VALID_CONFIG_KEYS remains the full schema of what the
# config file accepts; this is the smaller question of what a tool call is
# allowed to change on the user's behalf.
#
# What is missing, and why:
#
#   [safety]  mode, danger_fast    turn off confirmation permanently
#   [mcp]     safety_mode          turn off the MCP transport's only gate
#   [remote]  url, api_key         point inference at another endpoint, which
#                                  sends the whole conversation there
#   [prompt]  persona,             spliced verbatim into every future system
#             extra_instructions   prompt, so it persists across sessions
#   [model]   path, hf_repo,       choose which weights get downloaded and run
#             hf_file
#   [backup]  enabled              disable undo ahead of a destructive edit
#   [skills]  enabled              re-enable skill loading after a user
#                                  switched it off
#   [engine]  preferred            switch inference to a different backend
#   [ollama]  url                  as [remote] url
#
# None of these are things a user asks for mid-conversation often enough to be
# worth the tool call; all of them are things injected text would ask for.
# They remain editable in ~/.config/natshell/config.toml, by the user, on
# purpose.
LLM_WRITABLE_KEYS: dict[str, frozenset[str]] = {
    "agent": frozenset(
        {"max_steps", "plan_max_steps", "temperature", "max_tokens", "context_reserve"}
    ),
    "model": frozenset(
        {"n_ctx", "n_threads", "n_gpu_layers", "main_gpu", "prompt_cache", "prompt_cache_mb"}
    ),
    "ollama": frozenset({"default_model", "n_ctx"}),
    "ui": frozenset({"theme"}),
    "backup": frozenset({"max_per_file"}),
    "kiwix": frozenset({"url"}),
    "memory": frozenset({"enabled", "max_chars", "min_ctx"}),
    "skills": frozenset({"disabled", "inject_in_compact"}),
}


def is_llm_writable(section: str, key: str) -> bool:
    """True if the update_config tool may change [section].key."""
    return key in LLM_WRITABLE_KEYS.get(section, frozenset())


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


# TOML basic-string escapes, per the spec's list of what must not appear raw.
_TOML_STRING_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def toml_string(value: str) -> str:
    """Render *value* as an escaped TOML basic string, quotes included.

    Values used to be interpolated as f'"{value}"' with no escaping at all, so
    a value containing a quote and a newline ended the string and continued the
    file as TOML source — reaching sections the caller's key allowlist was
    written to protect.  Values that merely contained a stray quote produced a
    file that would not parse, which stopped NatShell from starting.

    Escaping rather than rejecting also fixes an ordinary bug: a Windows path
    like C:\\Users\\me is not a valid TOML string until its backslashes are
    doubled.
    """
    out = []
    for char in value:
        escape = _TOML_STRING_ESCAPES.get(char)
        if escape is not None:
            out.append(escape)
        elif char < "\x20" or char == "\x7f":
            out.append(f"\\u{ord(char):04X}")
        else:
            out.append(char)
    return '"' + "".join(out) + '"'


def _format_toml_value(value: str | int | float | bool | list) -> str:
    """Render a Python value as TOML source."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return toml_string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_string(str(item)) for item in value) + "]"
    return str(value)


def _write_config_atomically(config_path: Path, text: str) -> None:
    """Write *text* to *config_path*, but only if it parses as TOML.

    Writing an unparseable config is a self-inflicted denial of service:
    load_config raised on the next start and NatShell would not launch until
    the file was edited by hand.  Validating first means a bad write fails
    where it happens, with the original file still in place.
    """
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"refusing to write invalid TOML to {config_path}: {e}") from e

    tmp_path = config_path.with_name(config_path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    if config_path.exists():
        # os.replace would otherwise hand the file default permissions, and
        # this file can hold an API key.
        try:
            os.chmod(tmp_path, config_path.stat().st_mode & 0o777)
        except OSError:
            pass
    os.replace(tmp_path, config_path)


def save_config_value(
    section: str, key: str, value: str | int | float | bool | list
) -> Path:
    """Persist a single config value to the user config file.

    Uses simple line-based TOML editing (same pattern as save_engine_preference).
    Returns the path to the config file.
    """
    cfg_dir = _get_config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    config_path = cfg_dir / "config.toml"

    if config_path.exists():
        lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    else:
        lines = []

    val_str = _format_toml_value(value)

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
            elif re.match(rf"^#?\s*{re.escape(key)}\s*=", stripped):
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

    _write_config_atomically(config_path, "".join(lines))
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
        _merge_toml_safely(config, default_path)

    # Load user config
    if config_path:
        user_path = Path(config_path)
    else:
        user_path = _get_config_dir() / "config.toml"

    if user_path.exists():
        _merge_toml_safely(config, user_path)

    # Support NATSHELL_API_KEY environment variable as alternative to config file
    env_api_key = os.environ.get("NATSHELL_API_KEY")
    if env_api_key:
        config.remote.api_key = env_api_key

    # Warn if config file contains an API key and has permissive permissions
    # (Unix permission bits are meaningless on Windows — skip the check)
    if config.remote.api_key and user_path.exists():
        from natshell.platform import is_windows

        if not is_windows():
            try:
                perms = user_path.stat().st_mode & 0o777
                if perms & 0o077:
                    logger.warning(
                        "Config file %s has permissive permissions (%04o) "
                        "and contains an API key. Run: chmod 600 %s",
                        user_path,
                        perms,
                        user_path,
                    )
            except OSError:
                pass

    return config


_SECTIONS = (
    "model", "remote", "ollama", "agent", "safety",
    "ui", "backup", "engine", "mcp", "kiwix", "prompt", "memory", "skills",
)


def _merge_toml_safely(config: NatShellConfig, path: Path) -> None:
    """Merge *path* into *config*, logging and skipping it if it will not parse.

    An unparseable config used to propagate out of tomllib.load and stop
    NatShell from starting at all, which turned any malformed value into a
    lockout that had to be fixed by hand-editing the file.  Degrading to the
    values already loaded is the better failure: config.default.toml is merged
    first, and the classifier's blocked patterns are built in rather than
    configured, so skipping a broken file does not skip the safety rules.
    """
    try:
        _merge_toml(config, path)
    except tomllib.TOMLDecodeError as e:
        logger.error("Ignoring %s — it is not valid TOML: %s", path, e)
    except OSError as e:
        logger.error("Ignoring %s — could not be read: %s", path, e)


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


def save_skills_disabled(disabled: list[str]) -> Path:
    """Persist the skills.disabled list to the user config file.

    Delegates so that list items are escaped and the result is validated by the
    same code as every other written value; this used to be a near-copy of
    save_config_value that interpolated each name with no escaping.
    """
    return save_config_value("skills", "disabled", list(disabled))


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
