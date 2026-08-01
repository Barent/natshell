"""Update a NatShell configuration value at runtime and persist to disk."""

from __future__ import annotations

import logging

from natshell.config import (
    CONFIG_ENUMS,
    VALID_CONFIG_KEYS,
    NatShellConfig,
    save_config_value,
)
from natshell.tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# ── Read-only/writable key split ────────────────────────────────────────

# Keys the model is allowed to set via update_config.  Excluded are:
# safety settings (could disable user protections), remote inference URL/api_key
# (could redirect traffic / use a malicious backend), engine preference, MCP safety,
# prompt persona (could override system instructions), and skills.enabled toggles.
LLM_WRITABLE_KEYS: dict[str, dict[str, str]] = {
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
    "agent": {
        "max_steps": "int",
        "plan_max_steps": "int",
        "temperature": "float",
        "max_tokens": "int",
        "context_reserve": "int",
    },
    "ui": {
        "theme": "str",
    },
    "backup": {
        "enabled": "bool",
        "max_per_file": "int",
    },
    "kiwix": {
        "url": "str",
    },
    "memory": {
        "enabled": "bool",
        "max_chars": "int",
        "min_ctx": "int",
    },
    "skills": {
        "disabled": "list",
        "inject_in_compact": "bool",
    },
    "ollama": {
        "url": "str",
        "default_model": "str",
        "n_ctx": "int",
    },
    "prompt": {
        "extra_instructions": "str",
    },
}

# ── Live config injection ───────────────────────────────────────────────

_live_config: NatShellConfig | None = None


def set_live_config(config: NatShellConfig) -> None:
    """Inject the live NatShellConfig so updates take effect immediately."""
    global _live_config
    _live_config = config


# ── Tool definition ──────────────────────────────────────────────────────

DEFINITION = ToolDefinition(
    name="update_config",
    description=(
        "Update a NatShell configuration value. Changes are saved to "
        "~/.config/natshell/config.toml and take effect immediately. "
        "Use this when the user asks to change settings like temperature, "
        "max_steps, safety mode, GPU layers, etc."
    ),
    parameters={
        "type": "object",
        "properties": {
            "section": {
                "type": "string",
                "description": (
                    "The config section (e.g. 'agent', 'model', 'safety', 'engine', "
                    "'ui', 'backup', 'remote', 'ollama', 'mcp')"
                ),
            },
            "key": {
                "type": "string",
                "description": (
                    "The config key within the section "
                    "(e.g. 'temperature', 'n_gpu_layers')"
                ),
            },
            "value": {
                "type": "string",
                "description": "The new value (will be coerced to the correct type)",
            },
        },
        "required": ["section", "key", "value"],
    },
    requires_confirmation=True,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _coerce_value(value_str: str, type_str: str) -> int | float | bool | str:
    """Coerce a string value to the expected type.

    Raises ValueError on type mismatch.
    """
    if type_str == "int":
        try:
            return int(value_str)
        except ValueError:
            raise ValueError(f"Expected an integer, got: {value_str!r}")
    elif type_str == "float":
        try:
            return float(value_str)
        except ValueError:
            raise ValueError(f"Expected a float, got: {value_str!r}")
    elif type_str == "bool":
        low = value_str.lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"Expected a boolean, got: {value_str!r}")
    else:
        return value_str


def _apply_to_live_config(
    config: NatShellConfig, section: str, key: str, value: int | float | bool | str
) -> None:
    """Apply a value to the live config object."""
    section_obj = getattr(config, section, None)
    if section_obj is not None and hasattr(section_obj, key):
        setattr(section_obj, key, value)


# ── Handler ──────────────────────────────────────────────────────────────


async def update_config(section: str, key: str, value: str) -> ToolResult:
    """Validate, coerce, persist, and apply a config change."""
    # Validate section + key — model is NOT allowed to change security-sensitive keys
    if section not in LLM_WRITABLE_KEYS:
        return ToolResult(
            error=(
                f"Section [{section}] is read-only and cannot be changed by update_config. "
                f"(Security-sensitive sections require manual edits to config.toml.)"
            ),
            exit_code=1,
        )

    section_keys = LLM_WRITABLE_KEYS[section]
    if key not in section_keys:
        return ToolResult(
            error=f"Key {key!r} cannot be changed by update_config.",
            exit_code=1,
        )

    # Coerce value
    type_str = section_keys[key]
    try:
        coerced = _coerce_value(value, type_str)
    except ValueError as e:
        return ToolResult(error=str(e), exit_code=1)

    # Enum validation
    if section in CONFIG_ENUMS and key in CONFIG_ENUMS[section]:
        allowed = CONFIG_ENUMS[section][key]
        if coerced not in allowed:
            return ToolResult(
                error=f"Invalid value {coerced!r} for [{section}].{key}. "
                f"Allowed values: {', '.join(allowed)}",
                exit_code=1,
            )

    # Persist to disk
    try:
        config_path = save_config_value(section, key, coerced)
    except Exception as e:
        return ToolResult(
            error=f"Failed to save config: {e}",
            exit_code=1,
        )

    # Apply to live config
    if _live_config is not None:
        _apply_to_live_config(_live_config, section, key, coerced)

    # Sync kiwix URL to the running tool
    if section == "kiwix" and key == "url":
        from natshell.tools.kiwix_search import set_kiwix_url

        set_kiwix_url(str(coerced))

    return ToolResult(
        output=f"Updated [{section}].{key} = {coerced!r} (saved to {config_path})"
    )
